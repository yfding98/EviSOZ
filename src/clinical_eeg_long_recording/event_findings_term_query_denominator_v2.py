"""Independent term-query opportunity denominator for EEG event Findings.

This module is a strictly additive, public/synthetic-only research sidecar.
It consumes an already validated v1 item-scope denominator source inventory
and receipt as its structural host-trust root.  It never accepts a Findings
payload, payload-declared opportunities, pattern candidates, event outcomes,
private annotations, report text, or Qwen output.

The v2 denominator freezes 41 operational queries.  A query is more specific
than a term: it also fixes claim kind, temporal context, physical unit domain,
view/reference/bandwidth profile, and one primary atom-roster item.  Every
query is materialized even when its producer is unimplemented.  Such cells are
``not_evaluable``; they are never silently removed.

``negative_opportunity_eligible`` is only an independently replayable
precondition for a future explicit negative decision.  This module never
authorizes a clinical absence, report promotion, diagnostic correctness, or a
cortical SOZ/EZ claim.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .event_findings_atom_roster import (
    load_event_findings_atom_roster_policy,
    validate_event_findings_atom_roster_policy,
)
from .event_findings_denominator import (
    canonicalize_physical_interval_union,
    load_event_findings_denominator_policy,
    materialize_event_findings_denominator_receipt,
    physical_interval_union_seconds,
    validate_event_findings_denominator_policy,
    validate_event_findings_denominator_receipt,
    validate_event_findings_denominator_source_inventory,
)


FINDING_TERM_MANIFEST_SCHEMA_VERSION_V2 = (
    "clinical_eeg_finding_term_manifest_v2"
)
FINDING_TERM_MANIFEST_ID_V2 = "CLINICAL-EEG-FINDING-TERM-MANIFEST-V2"
EVENT_FINDINGS_TERM_QUERY_POLICY_SCHEMA_VERSION_V2 = (
    "clinical_eeg_event_findings_term_query_denominator_policy_v2"
)
EVENT_FINDINGS_TERM_QUERY_POLICY_ID_V2 = (
    "CLINICAL-EEG-EVENT-FINDINGS-TERM-QUERY-DENOMINATOR-POLICY-V2"
)
EVENT_FINDINGS_TERM_QUERY_SOURCE_INVENTORY_SCHEMA_VERSION_V2 = (
    "clinical_eeg_event_findings_term_query_source_inventory_v2"
)
EVENT_FINDINGS_TERM_QUERY_RECEIPT_SCHEMA_VERSION_V2 = (
    "clinical_eeg_event_findings_term_query_denominator_receipt_v2"
)
EVENT_FINDINGS_TERM_QUERY_METHOD_ID_V2 = (
    "EEG-ONLY-INDEPENDENT-EVENT-FINDINGS-TERM-QUERY-DENOMINATOR-V2"
)
TERM_QUERY_CAPABILITY_RECEIPT_SCHEMA_VERSION_V2 = (
    "clinical_eeg_term_query_capability_receipt_v2"
)
TERM_QUERY_SENSITIVITY_RECEIPT_SCHEMA_VERSION_V2 = (
    "clinical_eeg_term_query_sensitivity_receipt_v2"
)

DEFAULT_FINDING_TERM_MANIFEST_SHA256_V2 = (
    "22653eca0f2c32db45a4562efc7a72d799cb23b257f0972f43338efd0f971f7e"
)
DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_SHA256_V2 = (
    "8dd9eedbb081a92edcd40437110b10df57d2fa589a6f443d61bfc5130764a66a"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINDING_TERM_MANIFEST_PATH_V2 = (
    _ROOT / "configs" / "clinical_eeg_finding_term_manifest_v2.json"
)
FINDING_TERM_MANIFEST_SCHEMA_PATH_V2 = (
    _ROOT / "schemas" / "clinical_eeg_finding_term_manifest_v2.schema.json"
)
DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_PATH_V2 = (
    _ROOT
    / "configs"
    / "clinical_eeg_event_findings_term_query_denominator_policy_v2.json"
)
EVENT_FINDINGS_TERM_QUERY_POLICY_SCHEMA_PATH_V2 = (
    _ROOT
    / "schemas"
    / "clinical_eeg_event_findings_term_query_denominator_policy_v2.schema.json"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_EXPECTED_SCOPE_IDS = {
    "event_analysis_window",
    "background_context",
    "candidate_emergence_interval",
    "event_course_interval",
    "post_event_context",
    "quality_evaluable_interval",
}
_ALLOWED_PHYSICAL_UNIT_TYPES = {"electrode", "lead"}
_EVENT_UNIT = {
    "unit_type": "event",
    "unit_id": "GLOBAL",
    "unit_key": "event:GLOBAL",
}
_SOURCE_FIREWALL_KEYS = {
    "private_data_used",
    "event_findings_payload_used",
    "findings_candidates_used",
    "payload_evaluation_opportunities_used",
    "pattern_candidates_used",
    "event_outcome_used",
    "scalp_onset_hypothesis_used",
    "edf_annotations_used",
    "spreadsheet_used",
    "doctor_labels_used",
    "clinical_text_used",
    "patient_metadata_used",
    "video_used",
    "sleep_staging_used",
    "provocation_used",
    "ecg_emg_eog_used",
    "qwen_used",
    "production_route_used",
}
_QUALIFICATION_BASE_FIELDS = {
    "schema_version",
    "receipt_id",
    "term_query_id",
    "term_id",
    "query_cell_key",
    "unit_key",
    "claim_kind",
    "family",
    "temporal_context",
    "scope_id",
    "view_profile_id",
    "reference_profile_id",
    "bandwidth_profile_id",
    "effective_bandwidth_hz",
    "producer_id",
    "target_domain_id",
    "validation_scope",
    "patient_disjoint",
    "frozen_before_inference",
    "qualification_passed",
    "policy_sha256",
    "term_manifest_sha256",
    "receipt_sha256",
}

_EXPECTED_TERM_IDS_V2 = (
    "acns_derived_frequency_evolution_candidate",
    "acns_derived_location_evolution_candidate",
    "acns_derived_morphology_evolution_candidate",
    "artifact_interval_candidate",
    "dc_shift_candidate",
    "definite_evolution",
    "deterministic_background_spectral_profile",
    "deterministic_event_physical_amplitude_profile",
    "deterministic_event_rhythmicity_profile",
    "deterministic_event_spectral_profile",
    "deterministic_later_involvement_candidate",
    "deterministic_multifeature_change_point_candidate",
    "deterministic_recovery_context_profile",
    "deterministic_signal_usable_fraction",
    "electrographic_seizure",
    "event_amplitude_course_profile",
    "event_rhythmicity_course_profile",
    "high_frequency_oscillation",
    "ictal_sharp_contoured_component_candidate",
    "interictal_epileptiform_discharge",
    "low_voltage_fast_activity",
    "periodic_discharge",
    "periodic_element_candidate",
    "post_event_attenuation_candidate",
    "post_event_slowing_candidate",
    "reference_specific_spatial_change_candidate",
    "reference_specific_spatial_field_measurement",
    "research_scalp_visible_onset_hypothesis",
    "return_to_comparable_background_candidate",
    "rhythmic_run_candidate",
    "rhythmic_theta_activity",
    "sharp_wave",
    "spike",
    "very_slow_activity_candidate",
)

_EXPECTED_TERM_QUERY_IDS_V2 = (
    "TQ-ACQUISITION-DC-SHIFT",
    "TQ-ACQUISITION-HFO",
    "TQ-ACQUISITION-LVFA",
    "TQ-ACQUISITION-VERY-SLOW",
    "TQ-ARTIFACT-INTERVAL",
    "TQ-BACKGROUND-SPECTRAL-PROFILE",
    "TQ-DEFINITE-EVOLUTION-FREQUENCY",
    "TQ-DEFINITE-EVOLUTION-LOCATION",
    "TQ-DEFINITE-EVOLUTION-MORPHOLOGY",
    "TQ-ELECTROGRAPHIC-SEIZURE",
    "TQ-EVENT-AMPLITUDE-COURSE",
    "TQ-EVENT-FREQUENCY-SPECTRAL-PROFILE",
    "TQ-EVENT-RHYTHMICITY-COURSE",
    "TQ-EVOLUTION-FREQUENCY",
    "TQ-EVOLUTION-LOCATION",
    "TQ-EVOLUTION-MORPHOLOGY",
    "TQ-IED-RECORD-SUMMARY",
    "TQ-ONSET-BOUNDARY",
    "TQ-ONSET-CROSS-REFERENCE-RESOLUTION",
    "TQ-ONSET-EARLIEST-SET-MEMBERSHIP",
    "TQ-ONSET-INVOLVEMENT-ORDER-NEAR-SYNCHRONY",
    "TQ-ONSET-LATER-INVOLVEMENT",
    "TQ-ONSET-PER-UNIT-INVOLVEMENT",
    "TQ-ONSET-REFERENCE-FIELD-POLARITY",
    "TQ-ONSET-RESEARCH-SCALP-VISIBLE-HYPOTHESIS",
    "TQ-PERIODIC-DISCHARGE-SUMMARY",
    "TQ-PERIODIC-ELEMENT-INSTANCE",
    "TQ-PHYSICAL-AMPLITUDE-PROFILE",
    "TQ-POST-EVENT-ATTENUATION",
    "TQ-POST-EVENT-RETURN-COMPARABLE-BACKGROUND",
    "TQ-POST-EVENT-SLOWING",
    "TQ-RECOVERY-OFFSET",
    "TQ-RHYTHMIC-RUN-INSTANCE",
    "TQ-RHYTHMIC-THETA-SUMMARY",
    "TQ-RHYTHMICITY-PROFILE",
    "TQ-SHARP-CONTOURED-ICTAL-COMPONENT-INSTANCE",
    "TQ-SHARP-WAVE-INTERICTAL-INSTANCE",
    "TQ-SHARP-WAVE-INTERICTAL-SUMMARY",
    "TQ-SIGNAL-USABLE-FRACTION",
    "TQ-SPIKE-INTERICTAL-INSTANCE",
    "TQ-SPIKE-INTERICTAL-SUMMARY",
)
_EXPECTED_ACQUISITION_GATE_BINDINGS_V2 = {
    "TQ-ACQUISITION-DC-SHIFT": {
        "gate_id": "ACQ-GATE-DC-SHIFT-V1",
        "term_id": "dc_shift_candidate",
        "primary_roster_item_id": "a_dc_shift_pattern_instances",
    },
    "TQ-ACQUISITION-HFO": {
        "gate_id": "ACQ-GATE-HFO-V1",
        "term_id": "high_frequency_oscillation",
        "primary_roster_item_id": "a_hfo_pattern_instances",
    },
    "TQ-ACQUISITION-LVFA": {
        "gate_id": "ACQ-GATE-LVFA-V1",
        "term_id": "low_voltage_fast_activity",
        "primary_roster_item_id": "a_lvfa_pattern_instances",
    },
    "TQ-ACQUISITION-VERY-SLOW": {
        "gate_id": "ACQ-GATE-VERY-SLOW-V1",
        "term_id": "very_slow_activity_candidate",
        "primary_roster_item_id": "a_very_slow_activity_pattern_instances",
    },
}
_EXPECTED_ACQUISITION_POSITIVE_RECEIPT_KINDS_V2 = [
    "term_query_capability_receipt",
    "term_specific_artifact_qualification_receipt",
    "term_decision_receipt",
]
_EXPECTED_ACQUISITION_NEGATIVE_RECEIPT_KINDS_V2 = [
    "complete_term_specific_opportunity_receipt",
    "term_query_sensitivity_receipt",
]
_EXPECTED_TERM_ROWS_SHA256_V2 = (
    "4c6ee004f9cd129bec57a2383eb74e32dd55319c8501a53cb8b2014685995dfc"
)
_EXPECTED_QUERY_SPECS_SHA256_V2 = (
    "3b43d4695c132c67aa367e56a745e4735cbadd4e060bec1218ea8642170e351b"
)


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
    candidate = deepcopy(dict(value))
    candidate.pop(field, None)
    return _canonical_sha256(candidate)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if type(value) is not dict:
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _schema_errors(value: object, schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: list(item.path),
    )
    rendered: list[str] = []
    for error in errors[:16]:
        path = "/" + "/".join(str(part) for part in error.path)
        rendered.append(f"{path}: {error.message}")
    if len(errors) > 16:
        rendered.append(f"... {len(errors) - 16} more error(s)")
    return rendered


def _require_id(value: object, context: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical ID")
    return value


def _require_sha256(value: object, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _sorted_unique_ids(values: object, context: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be an ID array")
    result = sorted(_require_id(value, context) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must not contain duplicate IDs")
    return result


def _typed_unit(value: object, context: str, *, allow_event: bool) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    if set(value) != {"unit_type", "unit_id", "unit_key"}:
        raise ValueError(
            f"{context} must contain exactly unit_type, unit_id, and unit_key"
        )
    unit_type = _require_id(value["unit_type"], f"{context}.unit_type")
    unit_id = _require_id(value["unit_id"], f"{context}.unit_id")
    unit_key = _require_id(value["unit_key"], f"{context}.unit_key")
    allowed = set(_ALLOWED_PHYSICAL_UNIT_TYPES)
    if allow_event:
        allowed.add("event")
    if unit_type not in allowed:
        raise ValueError(f"{context} has an unsupported unit_type")
    if unit_key != f"{unit_type}:{unit_id}":
        raise ValueError(f"{context}.unit_key is not canonical")
    if unit_type == "event" and unit_key != "event:GLOBAL":
        raise ValueError("the event-global unit must be event:GLOBAL")
    if unit_type != "event" and unit_key == "event:GLOBAL":
        raise ValueError("a physical unit cannot use event:GLOBAL")
    return {"unit_type": unit_type, "unit_id": unit_id, "unit_key": unit_key}


def _bandwidth(
    value: object,
    context: str,
    *,
    nullable: bool,
) -> list[float] | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"{context} must be a two-number bandwidth")
    lower = _finite(value[0], f"{context}[0]")
    upper = _finite(value[1], f"{context}[1]")
    if lower < 0.0 or upper <= lower:
        raise ValueError(f"{context} must satisfy 0 <= lower < upper")
    return [lower, upper]


def _union_is_subset(
    subset: Sequence[Mapping[str, object]],
    superset: Sequence[Mapping[str, object]],
    tolerance: float,
) -> bool:
    inner = canonicalize_physical_interval_union(
        list(subset), tolerance_seconds=tolerance
    )
    outer = canonicalize_physical_interval_union(
        list(superset), tolerance_seconds=tolerance
    )
    outer_index = 0
    for segment in inner:
        while (
            outer_index < len(outer)
            and outer[outer_index]["stop"] < segment["start"] - tolerance
        ):
            outer_index += 1
        if outer_index >= len(outer):
            return False
        support = outer[outer_index]
        if (
            support["start"] > segment["start"] + tolerance
            or support["stop"] < segment["stop"] - tolerance
        ):
            return False
    return True


def _unions_equal(
    first: Sequence[Mapping[str, object]],
    second: Sequence[Mapping[str, object]],
    tolerance: float,
) -> bool:
    left = canonicalize_physical_interval_union(
        list(first), tolerance_seconds=tolerance
    )
    right = canonicalize_physical_interval_union(
        list(second), tolerance_seconds=tolerance
    )
    return len(left) == len(right) and all(
        abs(a["start"] - b["start"]) <= tolerance
        and abs(a["stop"] - b["stop"]) <= tolerance
        for a, b in zip(left, right)
    )


def event_findings_term_query_enumerator_code_sha256_v2() -> str:
    """Hash the exact additive v2 enumerator/runtime source."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_clinical_eeg_finding_term_manifest_v2(
    value: object,
    *,
    trusted_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("clinical EEG Finding term manifest v2 must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(candidate, FINDING_TERM_MANIFEST_SCHEMA_PATH_V2)
    if errors:
        raise ValueError("term manifest v2 schema validation failed: " + "; ".join(errors))
    expected_hash = _self_hash(candidate, "manifest_sha256")
    if candidate["manifest_sha256"] != expected_hash:
        raise ValueError("term manifest v2 SHA-256 mismatch")
    if trusted_manifest_sha256 is not None and expected_hash != _require_sha256(
        trusted_manifest_sha256, "trusted_manifest_sha256"
    ):
        raise ValueError("term manifest v2 is not host trusted")
    if candidate["manifest_id"] != FINDING_TERM_MANIFEST_ID_V2:
        raise ValueError("term manifest v2 ID mismatch")
    terms = list(candidate["terms"])
    term_ids = [str(row["term_id"]) for row in terms]
    if term_ids != sorted(term_ids):
        raise ValueError("term manifest v2 terms must use canonical term order")
    if len(term_ids) != len(set(term_ids)):
        raise ValueError("term manifest v2 has duplicate term IDs")
    if tuple(term_ids) != _EXPECTED_TERM_IDS_V2:
        raise ValueError("term manifest v2 does not exactly cover the frozen 34 terms")
    if _canonical_sha256(terms) != _EXPECTED_TERM_ROWS_SHA256_V2:
        raise ValueError("term manifest v2 frozen term semantics differ")
    canonical = set(term_ids)
    aliases: dict[str, str] = {}
    for row in terms:
        for alias in row["legacy_aliases"]:
            alias_id = _require_id(alias, "legacy_aliases")
            if alias_id in canonical or alias_id in aliases:
                raise ValueError("term manifest v2 legacy aliases collide")
            aliases[alias_id] = str(row["term_id"])
    if set(candidate["source_firewall"]) != {
        key for key in _SOURCE_FIREWALL_KEYS if key not in {
            "payload_evaluation_opportunities_used",
            "pattern_candidates_used",
            "event_outcome_used",
            "scalp_onset_hypothesis_used",
        }
    }:
        raise ValueError("term manifest v2 source firewall is incomplete")
    if any(candidate["source_firewall"].values()):
        raise ValueError("term manifest v2 used a forbidden source")
    return candidate


@lru_cache(maxsize=1)
def _default_term_manifest_v2() -> dict[str, Any]:
    return validate_clinical_eeg_finding_term_manifest_v2(
        _read_json(DEFAULT_FINDING_TERM_MANIFEST_PATH_V2),
        trusted_manifest_sha256=DEFAULT_FINDING_TERM_MANIFEST_SHA256_V2,
    )


def load_clinical_eeg_finding_term_manifest_v2(
    path: str | Path = DEFAULT_FINDING_TERM_MANIFEST_PATH_V2,
    *,
    trusted_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path)
    if trusted_manifest_sha256 is None:
        if resolved.resolve() != DEFAULT_FINDING_TERM_MANIFEST_PATH_V2.resolve():
            raise ValueError("a non-default term manifest requires a host trust anchor")
        trusted_manifest_sha256 = DEFAULT_FINDING_TERM_MANIFEST_SHA256_V2
    return validate_clinical_eeg_finding_term_manifest_v2(
        _read_json(resolved), trusted_manifest_sha256=trusted_manifest_sha256
    )


def _manifest(
    value: Mapping[str, object] | None,
    trusted_manifest_sha256: str | None,
) -> dict[str, Any]:
    if value is None:
        return deepcopy(_default_term_manifest_v2())
    if trusted_manifest_sha256 is None:
        trusted_manifest_sha256 = DEFAULT_FINDING_TERM_MANIFEST_SHA256_V2
    return validate_clinical_eeg_finding_term_manifest_v2(
        dict(value), trusted_manifest_sha256=trusted_manifest_sha256
    )


def _roster_specs(
    roster_policy: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in roster_policy["core_atom_specs"]:  # type: ignore[index]
        result[("core_atom", str(row["atom_id"]))] = row
    for row in roster_policy["child_roster_specs"]:  # type: ignore[index]
        result[("child_roster", str(row["child_roster_id"]))] = row
    return result


def _validated_roster_policy(
    value: Mapping[str, object] | None,
    trusted_sha256: str | None,
) -> dict[str, Any]:
    if value is None:
        if trusted_sha256 is None:
            return deepcopy(_default_roster_policy_v2())
        return load_event_findings_atom_roster_policy(trusted_policy_sha256=trusted_sha256)
    if trusted_sha256 is None:
        trusted_sha256 = str(load_event_findings_atom_roster_policy()["policy_sha256"])
    return validate_event_findings_atom_roster_policy(
        dict(value), trusted_policy_sha256=trusted_sha256
    )


def _validated_v1_policy(
    value: Mapping[str, object] | None,
    trusted_sha256: str | None,
) -> dict[str, Any]:
    if value is None:
        if trusted_sha256 is None:
            return deepcopy(_default_v1_policy_v2())
        return load_event_findings_denominator_policy(trusted_policy_sha256=trusted_sha256)
    if trusted_sha256 is None:
        trusted_sha256 = str(load_event_findings_denominator_policy()["policy_sha256"])
    return validate_event_findings_denominator_policy(
        dict(value), trusted_policy_sha256=trusted_sha256
    )


@lru_cache(maxsize=1)
def _default_roster_policy_v2() -> dict[str, Any]:
    return load_event_findings_atom_roster_policy()


@lru_cache(maxsize=1)
def _default_v1_policy_v2() -> dict[str, Any]:
    return load_event_findings_denominator_policy()


def validate_event_findings_term_query_denominator_policy_v2(
    value: object,
    *,
    trusted_policy_sha256: str | None = None,
    term_manifest: Mapping[str, object] | None = None,
    trusted_manifest_sha256: str | None = None,
    atom_roster_policy: Mapping[str, object] | None = None,
    trusted_atom_roster_policy_sha256: str | None = None,
    v1_denominator_policy: Mapping[str, object] | None = None,
    trusted_v1_denominator_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the closed 41-query contract and all parent-policy joins."""

    if type(value) is not dict:
        raise TypeError("event Findings term-query denominator policy v2 must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(candidate, EVENT_FINDINGS_TERM_QUERY_POLICY_SCHEMA_PATH_V2)
    if errors:
        raise ValueError(
            "term-query denominator policy v2 schema validation failed: "
            + "; ".join(errors)
        )
    expected_hash = _self_hash(candidate, "policy_sha256")
    if candidate["policy_sha256"] != expected_hash:
        raise ValueError("term-query denominator policy v2 SHA-256 mismatch")
    if trusted_policy_sha256 is not None and expected_hash != _require_sha256(
        trusted_policy_sha256, "trusted_policy_sha256"
    ):
        raise ValueError("term-query denominator policy v2 is not host trusted")
    if candidate["policy_id"] != EVENT_FINDINGS_TERM_QUERY_POLICY_ID_V2:
        raise ValueError("term-query denominator policy v2 ID mismatch")

    manifest = _manifest(term_manifest, trusted_manifest_sha256)
    roster = _validated_roster_policy(
        atom_roster_policy, trusted_atom_roster_policy_sha256
    )
    v1_policy = _validated_v1_policy(
        v1_denominator_policy, trusted_v1_denominator_policy_sha256
    )
    if (
        candidate["term_manifest_id"] != manifest["manifest_id"]
        or candidate["term_manifest_version"] != manifest["manifest_version"]
        or candidate["term_manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise ValueError("term-query policy is not bound to the trusted term manifest")
    if candidate["atom_roster_policy_sha256"] != roster["policy_sha256"]:
        raise ValueError("term-query policy is not bound to the trusted atom roster")
    if (
        candidate["v1_denominator_policy_id"] != v1_policy["policy_id"]
        or candidate["v1_denominator_policy_sha256"] != v1_policy["policy_sha256"]
    ):
        raise ValueError("term-query policy is not bound to the trusted v1 denominator")

    if list(candidate["scope_ids"]) != sorted(_EXPECTED_SCOPE_IDS):
        raise ValueError("term-query policy scope IDs do not match the frozen set")
    for key in (
        "view_profile_ids",
        "reference_profile_ids",
        "bandwidth_profile_ids",
    ):
        if list(candidate[key]) != _sorted_unique_ids(candidate[key], key):
            raise ValueError(f"term-query policy {key} must be canonical and unique")

    terms = {str(row["term_id"]): row for row in manifest["terms"]}
    roster_specs = _roster_specs(roster)
    v1_items = {
        (str(row["roster_item_kind"]), str(row["roster_item_id"])): row
        for row in v1_policy["item_scopes"]
    }
    queries = list(candidate["query_specs"])
    query_ids = tuple(str(row["term_query_id"]) for row in queries)
    if query_ids != _EXPECTED_TERM_QUERY_IDS_V2:
        raise ValueError("term-query policy does not exactly cover the frozen 41 queries")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("term-query policy contains duplicate query IDs")

    acquisition_policy = candidate[
        "acquisition_sensitive_qualification_gates"
    ]
    if (
        acquisition_policy["shared_generic_roster_or_receipt_allowed"]
        is not False
        or acquisition_policy["one_gate_status_may_satisfy_another"]
        is not False
        or acquisition_policy["all_gates_default_fail_closed"] is not True
    ):
        raise ValueError(
            "acquisition-sensitive qualification gates must remain "
            "independent and fail closed"
        )
    acquisition_gates = list(acquisition_policy["gates"])
    gate_ids = [str(row["gate_id"]) for row in acquisition_gates]
    gate_query_ids = [str(row["term_query_id"]) for row in acquisition_gates]
    gate_term_ids = [str(row["term_id"]) for row in acquisition_gates]
    gate_roster_ids = [
        str(row["primary_roster_item_id"])
        for row in acquisition_gates
    ]
    for values, label in (
        (gate_ids, "gate IDs"),
        (gate_query_ids, "query IDs"),
        (gate_term_ids, "term IDs"),
        (gate_roster_ids, "primary roster IDs"),
    ):
        if len(values) != len(set(values)):
            raise ValueError(
                "independent acquisition-sensitive gates have shared " + label
            )
    if set(gate_query_ids) != set(_EXPECTED_ACQUISITION_GATE_BINDINGS_V2):
        raise ValueError(
            "acquisition-sensitive gates do not cover the exact four queries"
        )
    gate_by_query = {
        str(row["term_query_id"]): row for row in acquisition_gates
    }
    query_by_id = {str(row["term_query_id"]): row for row in queries}
    for query_id, expected in _EXPECTED_ACQUISITION_GATE_BINDINGS_V2.items():
        gate = gate_by_query[query_id]
        query = query_by_id[query_id]
        if (
            gate["gate_id"] != expected["gate_id"]
            or gate["term_id"] != expected["term_id"]
            or gate["primary_roster_item_id"]
            != expected["primary_roster_item_id"]
            or gate["required_positive_receipt_kinds"]
            != _EXPECTED_ACQUISITION_POSITIVE_RECEIPT_KINDS_V2
            or gate["required_negative_receipt_kinds"]
            != _EXPECTED_ACQUISITION_NEGATIVE_RECEIPT_KINDS_V2
            or gate["default_status"] != "not_evaluable"
            or gate["report_promotion_authorized"] is not False
        ):
            raise ValueError(
                f"{query_id}: independent acquisition-sensitive gate "
                "contract differs"
            )
        primary = query["primary_roster_item"]
        secondary_ids = {
            str(row["roster_item_id"])
            for row in query["allowed_secondary_roster_items"]
        }
        if (
            query["term_id"] != expected["term_id"]
            or primary["roster_item_kind"] != "child_roster"
            or primary["roster_item_id"]
            != expected["primary_roster_item_id"]
            or query["implementation_status"]
            != "unimplemented_not_evaluable"
            or query["report_promotion_authorized"] is not False
            or query["temporal_context"] != "acquisition_sensitive_event"
            or {
                "q_view_reference_bandwidth_capability",
                "q_artifact_usable_support",
            }
            - secondary_ids
        ):
            raise ValueError(
                f"{query_id}: query is not bound to its independent "
                "acquisition and artifact gate"
            )
    if _canonical_sha256(queries) != _EXPECTED_QUERY_SPECS_SHA256_V2:
        raise ValueError("term-query policy frozen query semantics differ")

    used_views: set[str] = set()
    used_references: set[str] = set()
    used_bandwidths: set[str] = set()
    for row in queries:
        query_id = str(row["term_query_id"])
        term_id = str(row["term_id"])
        if term_id not in terms:
            raise ValueError(f"{query_id}: term is absent from the closed manifest")
        term = terms[term_id]
        if row["family"] != term["family"]:
            raise ValueError(f"{query_id}: query and term families differ")
        if row["temporal_context"] not in term["allowed_temporal_contexts"]:
            raise ValueError(f"{query_id}: temporal context is not authorized by the term")
        if row["scope_id"] not in _EXPECTED_SCOPE_IDS:
            raise ValueError(f"{query_id}: scope is not frozen")

        view_id = str(row["view_profile_id"])
        reference_id = str(row["reference_profile_id"])
        bandwidth_id = str(row["bandwidth_profile_id"])
        if view_id not in candidate["view_profile_ids"]:
            raise ValueError(f"{query_id}: undeclared view profile")
        if reference_id not in candidate["reference_profile_ids"]:
            raise ValueError(f"{query_id}: undeclared reference profile")
        if bandwidth_id not in candidate["bandwidth_profile_ids"]:
            raise ValueError(f"{query_id}: undeclared bandwidth profile")
        used_views.add(view_id)
        used_references.add(reference_id)
        used_bandwidths.add(bandwidth_id)

        required_bandwidth = _bandwidth(
            row["required_bandwidth_hz"],
            f"{query_id}.required_bandwidth_hz",
            nullable=True,
        )
        sample_rate = row["minimum_sample_rate_hz"]
        if sample_rate is not None:
            sample_rate_value = _finite(sample_rate, f"{query_id}.minimum_sample_rate_hz")
            if sample_rate_value <= 0.0:
                raise ValueError(f"{query_id}: minimum sample rate must be positive")
            if (
                required_bandwidth is not None
                and sample_rate_value + 1e-9 < 2.0 * required_bandwidth[1]
            ):
                raise ValueError(f"{query_id}: minimum sample rate violates Nyquist")
        if row["required_coupling"] == "dc" and (
            required_bandwidth is None or required_bandwidth[0] != 0.0
        ):
            raise ValueError(f"{query_id}: DC coupling requires a zero-Hz lower bound")

        primary = row["primary_roster_item"]
        primary_key = (
            str(primary["roster_item_kind"]),
            str(primary["roster_item_id"]),
        )
        if primary_key not in roster_specs or primary_key not in v1_items:
            raise ValueError(f"{query_id}: primary roster item is not frozen in v1")
        primary_spec = roster_specs[primary_key]
        binding_status = str(row["primary_binding_status"])
        if binding_status == "registered":
            if term_id not in primary_spec["allowed_term_ids"]:
                raise ValueError(f"{query_id}: registered primary item rejects the term")
            if row["family"] not in primary_spec["allowed_finding_families"]:
                raise ValueError(f"{query_id}: registered primary item rejects the family")
            if (
                row["intrinsic_evidence_role"]
                not in primary_spec["allowed_intrinsic_evidence_roles"]
            ):
                raise ValueError(f"{query_id}: registered primary item rejects the role")
        if binding_status == "requires_roster_extension" and (
            row["implementation_status"] != "unimplemented_not_evaluable"
        ):
            raise ValueError(f"{query_id}: an unregistered term must remain unimplemented")
        if (
            primary_spec["current_implementation_status"]
            == "unimplemented_not_evaluable"
            and row["implementation_status"] != "unimplemented_not_evaluable"
        ):
            raise ValueError(f"{query_id}: unimplemented primary item cannot activate a query")

        secondary_keys: list[tuple[str, str]] = []
        for secondary in row["allowed_secondary_roster_items"]:
            secondary_key = (
                str(secondary["roster_item_kind"]),
                str(secondary["roster_item_id"]),
            )
            if secondary_key not in roster_specs or secondary_key not in v1_items:
                raise ValueError(f"{query_id}: secondary roster item is not frozen in v1")
            secondary_keys.append(secondary_key)
        if len(secondary_keys) != len(set(secondary_keys)):
            raise ValueError(f"{query_id}: duplicate secondary roster item")
        if primary_key in secondary_keys:
            raise ValueError(f"{query_id}: primary item cannot be repeated as secondary")

        onset_role = row["intrinsic_evidence_role"] == "onset_eligible"
        causal_permission = (
            row["onset_support_permission"]
            == "required_future_free_causal_if_positive"
        )
        if onset_role != causal_permission:
            raise ValueError(f"{query_id}: onset role and causal-view permission differ")
        if causal_permission and not view_id.startswith("VIEW-ONSET-CAUSAL"):
            raise ValueError(f"{query_id}: onset-eligible query lacks a causal view")
        if row["report_promotion_authorized"] is not False:
            raise ValueError(f"{query_id}: report promotion must remain forbidden")

    if set(candidate["view_profile_ids"]) != used_views:
        raise ValueError("term-query policy declares an unused or missing view profile")
    if set(candidate["reference_profile_ids"]) != used_references:
        raise ValueError("term-query policy declares an unused or missing reference profile")
    if set(candidate["bandwidth_profile_ids"]) != used_bandwidths:
        raise ValueError("term-query policy declares an unused or missing bandwidth profile")
    if set(candidate["source_firewall"]) != _SOURCE_FIREWALL_KEYS:
        raise ValueError("term-query policy source firewall is incomplete")
    if any(value is not False for value in candidate["source_firewall"].values()):
        raise ValueError("term-query policy used a forbidden source")
    return candidate


@lru_cache(maxsize=1)
def _default_policy_v2() -> dict[str, Any]:
    return validate_event_findings_term_query_denominator_policy_v2(
        _read_json(DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_PATH_V2),
        trusted_policy_sha256=DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_SHA256_V2,
    )


def load_event_findings_term_query_denominator_policy_v2(
    path: str | Path = DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_PATH_V2,
    *,
    trusted_policy_sha256: str | None = None,
    **validation_kwargs: object,
) -> dict[str, Any]:
    resolved = Path(path)
    if trusted_policy_sha256 is None:
        if resolved.resolve() != DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_PATH_V2.resolve():
            raise ValueError("a non-default term-query policy requires a host trust anchor")
        trusted_policy_sha256 = DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_SHA256_V2
    return validate_event_findings_term_query_denominator_policy_v2(
        _read_json(resolved),
        trusted_policy_sha256=trusted_policy_sha256,
        **validation_kwargs,
    )


def _policy_v2(
    value: Mapping[str, object] | None,
    trusted_policy_sha256: str | None,
) -> dict[str, Any]:
    if value is None:
        if trusted_policy_sha256 is None:
            return deepcopy(_default_policy_v2())
        return load_event_findings_term_query_denominator_policy_v2(
            trusted_policy_sha256=trusted_policy_sha256
        )
    if trusted_policy_sha256 is None:
        trusted_policy_sha256 = DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_SHA256_V2
    if (
        trusted_policy_sha256 == DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_SHA256_V2
        and dict(value) == _default_policy_v2()
    ):
        return deepcopy(_default_policy_v2())
    return validate_event_findings_term_query_denominator_policy_v2(
        dict(value), trusted_policy_sha256=trusted_policy_sha256
    )


def _v1_item_rows(
    v1_policy: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    return {
        (str(row["roster_item_kind"]), str(row["roster_item_id"])): row
        for row in v1_policy["item_scopes"]  # type: ignore[index]
    }


def _static_query_cell(
    query: Mapping[str, object],
    unit: Mapping[str, str],
    v1_items: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, Any]:
    primary = query["primary_roster_item"]
    primary_kind = str(primary["roster_item_kind"])  # type: ignore[index]
    primary_id = str(primary["roster_item_id"])  # type: ignore[index]
    v1_item = v1_items[(primary_kind, primary_id)]
    primary_unit_key = (
        str(unit["unit_key"])
        if v1_item["granularity"] == "unit"
        else "event:GLOBAL"
    )
    if v1_item["granularity"] == "unit" and unit["unit_type"] == "event":
        raise ValueError(
            f"{query['term_query_id']}: event-global query cannot bind a per-unit primary"
        )
    query_id = str(query["term_query_id"])
    unit_key = str(unit["unit_key"])
    return {
        "query_cell_key": f"{query_id}:{unit_key}",
        "term_query_id": query_id,
        "term_id": str(query["term_id"]),
        "claim_kind": str(query["claim_kind"]),
        "family": str(query["family"]),
        "temporal_context": str(query["temporal_context"]),
        "intrinsic_evidence_role": str(query["intrinsic_evidence_role"]),
        "scope_id": str(query["scope_id"]),
        "view_profile_id": str(query["view_profile_id"]),
        "reference_profile_id": str(query["reference_profile_id"]),
        "bandwidth_profile_id": str(query["bandwidth_profile_id"]),
        "required_bandwidth_hz": deepcopy(query["required_bandwidth_hz"]),
        "minimum_sample_rate_hz": query["minimum_sample_rate_hz"],
        "required_coupling": str(query["required_coupling"]),
        "implementation_status": str(query["implementation_status"]),
        "negative_semantics": str(query["negative_semantics"]),
        "unit": deepcopy(dict(unit)),
        "primary_roster_item_kind": primary_kind,
        "primary_roster_item_id": primary_id,
        "primary_v1_cell_key": f"{primary_kind}:{primary_id}:{primary_unit_key}",
    }


def enumerate_event_findings_term_query_cells_v2(
    typed_expected_units: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Enumerate every frozen query independently of any Finding candidate."""

    denominator_policy = _policy_v2(policy, trusted_policy_sha256)
    units = sorted(
        (
            _typed_unit(row, "typed_expected_units", allow_event=False)
            for row in typed_expected_units
        ),
        key=lambda row: row["unit_key"],
    )
    if not units:
        raise ValueError("typed expected-unit inventory must be non-empty")
    unit_keys = [row["unit_key"] for row in units]
    if len(unit_keys) != len(set(unit_keys)):
        raise ValueError("typed expected-unit inventory has duplicate canonical keys")
    v1_policy = _validated_v1_policy(None, None)
    if denominator_policy["v1_denominator_policy_sha256"] != v1_policy["policy_sha256"]:
        raise ValueError("term-query policy and v1 denominator policy differ")
    v1_items = _v1_item_rows(v1_policy)
    result: list[dict[str, Any]] = []
    for query in denominator_policy["query_specs"]:
        targets = [_EVENT_UNIT] if query["unit_domain"] == "event_global" else units
        result.extend(_static_query_cell(query, unit, v1_items) for unit in targets)
    result.sort(key=lambda row: str(row["query_cell_key"]))
    keys = [str(row["query_cell_key"]) for row in result]
    if len(keys) != len(set(keys)):
        raise ValueError("term-query enumeration produced duplicate cells")
    return result


def _expected_static_query_cell(
    value: object,
    policy: Mapping[str, object],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("query_cell must be an enumerated query-cell object")
    unit = _typed_unit(value.get("unit"), "query_cell.unit", allow_event=True)
    query_id = _require_id(value.get("term_query_id"), "query_cell.term_query_id")
    query = next(
        (row for row in policy["query_specs"] if row["term_query_id"] == query_id),
        None,
    )
    if query is None:
        raise ValueError("query_cell references an unregistered term query")
    v1_items = _v1_item_rows(_validated_v1_policy(None, None))
    expected = _static_query_cell(query, unit, v1_items)
    if dict(value) != expected:
        raise ValueError("query_cell does not exactly match candidate-blind enumeration")
    return expected


def _qualification_receipt_v2(
    kind: str,
    *,
    query_cell: Mapping[str, object],
    effective_bandwidth_hz: Sequence[object] | None,
    producer_id: str,
    target_domain_id: str,
    validation_scope: str,
    patient_disjoint: bool,
    frozen_before_inference: bool,
    qualification_passed: bool,
    opportunity_policy_sha256: str | None,
    policy: Mapping[str, object] | None,
    trusted_policy_sha256: str | None,
    term_manifest: Mapping[str, object] | None,
    trusted_manifest_sha256: str | None,
) -> dict[str, Any]:
    denominator_policy = _policy_v2(policy, trusted_policy_sha256)
    manifest = _manifest(term_manifest, trusted_manifest_sha256)
    cell = _expected_static_query_cell(query_cell, denominator_policy)
    bandwidth = _bandwidth(
        effective_bandwidth_hz, "effective_bandwidth_hz", nullable=True
    )
    if type(patient_disjoint) is not bool or type(frozen_before_inference) is not bool:
        raise TypeError("qualification isolation flags must be booleans")
    if type(qualification_passed) is not bool:
        raise TypeError("qualification_passed must be a boolean")
    schema_version = (
        TERM_QUERY_CAPABILITY_RECEIPT_SCHEMA_VERSION_V2
        if kind == "capability"
        else TERM_QUERY_SENSITIVITY_RECEIPT_SCHEMA_VERSION_V2
    )
    seed: dict[str, Any] = {
        "schema_version": schema_version,
        "term_query_id": cell["term_query_id"],
        "term_id": cell["term_id"],
        "query_cell_key": cell["query_cell_key"],
        "unit_key": cell["unit"]["unit_key"],
        "claim_kind": cell["claim_kind"],
        "family": cell["family"],
        "temporal_context": cell["temporal_context"],
        "scope_id": cell["scope_id"],
        "view_profile_id": cell["view_profile_id"],
        "reference_profile_id": cell["reference_profile_id"],
        "bandwidth_profile_id": cell["bandwidth_profile_id"],
        "effective_bandwidth_hz": bandwidth,
        "producer_id": _require_id(producer_id, "producer_id"),
        "target_domain_id": _require_id(target_domain_id, "target_domain_id"),
        "validation_scope": _require_id(validation_scope, "validation_scope"),
        "patient_disjoint": patient_disjoint,
        "frozen_before_inference": frozen_before_inference,
        "qualification_passed": qualification_passed,
        "policy_sha256": denominator_policy["policy_sha256"],
        "term_manifest_sha256": manifest["manifest_sha256"],
    }
    if kind == "sensitivity":
        seed["opportunity_policy_sha256"] = _require_sha256(
            opportunity_policy_sha256, "opportunity_policy_sha256"
        )
    prefix = "TQCAP" if kind == "capability" else "TQSENS"
    receipt: dict[str, Any] = {
        **seed,
        "receipt_id": f"{prefix}-{_canonical_sha256(seed)[:24]}",
        "receipt_sha256": "0" * 64,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return _validate_qualification_receipt_v2(
        receipt, kind=kind, policy=denominator_policy, manifest=manifest
    )


def build_term_query_capability_receipt_v2(
    *,
    query_cell: Mapping[str, object],
    effective_bandwidth_hz: Sequence[object] | None,
    producer_id: str,
    target_domain_id: str = "PUBLIC-SYNTHETIC-EEG",
    validation_scope: str = "PATIENT-DISJOINT-TERM-QUERY",
    patient_disjoint: bool = True,
    frozen_before_inference: bool = True,
    qualification_passed: bool = True,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    term_manifest: Mapping[str, object] | None = None,
    trusted_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    return _qualification_receipt_v2(
        "capability",
        query_cell=query_cell,
        effective_bandwidth_hz=effective_bandwidth_hz,
        producer_id=producer_id,
        target_domain_id=target_domain_id,
        validation_scope=validation_scope,
        patient_disjoint=patient_disjoint,
        frozen_before_inference=frozen_before_inference,
        qualification_passed=qualification_passed,
        opportunity_policy_sha256=None,
        policy=policy,
        trusted_policy_sha256=trusted_policy_sha256,
        term_manifest=term_manifest,
        trusted_manifest_sha256=trusted_manifest_sha256,
    )


def build_term_query_sensitivity_receipt_v2(
    *,
    query_cell: Mapping[str, object],
    effective_bandwidth_hz: Sequence[object] | None,
    opportunity_policy_sha256: str,
    producer_id: str,
    target_domain_id: str = "PUBLIC-SYNTHETIC-EEG",
    validation_scope: str = "PATIENT-DISJOINT-TERM-QUERY-SENSITIVITY",
    patient_disjoint: bool = True,
    frozen_before_inference: bool = True,
    qualification_passed: bool = True,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    term_manifest: Mapping[str, object] | None = None,
    trusted_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    return _qualification_receipt_v2(
        "sensitivity",
        query_cell=query_cell,
        effective_bandwidth_hz=effective_bandwidth_hz,
        producer_id=producer_id,
        target_domain_id=target_domain_id,
        validation_scope=validation_scope,
        patient_disjoint=patient_disjoint,
        frozen_before_inference=frozen_before_inference,
        qualification_passed=qualification_passed,
        opportunity_policy_sha256=opportunity_policy_sha256,
        policy=policy,
        trusted_policy_sha256=trusted_policy_sha256,
        term_manifest=term_manifest,
        trusted_manifest_sha256=trusted_manifest_sha256,
    )


def _validate_qualification_receipt_v2(
    value: object,
    *,
    kind: str,
    policy: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"term-query {kind} receipt must be an object")
    candidate = deepcopy(value)
    expected_fields = set(_QUALIFICATION_BASE_FIELDS)
    if kind == "sensitivity":
        expected_fields.add("opportunity_policy_sha256")
    if set(candidate) != expected_fields:
        raise ValueError(f"term-query {kind} receipt fields do not match v2")
    expected_schema = (
        TERM_QUERY_CAPABILITY_RECEIPT_SCHEMA_VERSION_V2
        if kind == "capability"
        else TERM_QUERY_SENSITIVITY_RECEIPT_SCHEMA_VERSION_V2
    )
    if candidate["schema_version"] != expected_schema:
        raise ValueError(f"term-query {kind} receipt schema mismatch")
    if candidate["receipt_sha256"] != _self_hash(candidate, "receipt_sha256"):
        raise ValueError(f"term-query {kind} receipt SHA-256 mismatch")
    prefix = "TQCAP" if kind == "capability" else "TQSENS"
    seed = {
        key: candidate[key]
        for key in candidate
        if key not in {"receipt_id", "receipt_sha256"}
    }
    if candidate["receipt_id"] != f"{prefix}-{_canonical_sha256(seed)[:24]}":
        raise ValueError(f"term-query {kind} receipt ID is not content addressed")
    for key in (
        "receipt_id",
        "producer_id",
        "target_domain_id",
        "validation_scope",
    ):
        _require_id(candidate[key], f"{kind}.{key}")
    if candidate["policy_sha256"] != policy["policy_sha256"]:
        raise ValueError(f"term-query {kind} receipt policy mismatch")
    if candidate["term_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError(f"term-query {kind} receipt manifest mismatch")
    if type(candidate["patient_disjoint"]) is not bool:
        raise TypeError(f"term-query {kind} patient_disjoint must be boolean")
    if type(candidate["frozen_before_inference"]) is not bool:
        raise TypeError(f"term-query {kind} frozen_before_inference must be boolean")
    if type(candidate["qualification_passed"]) is not bool:
        raise TypeError(f"term-query {kind} qualification_passed must be boolean")
    bandwidth = _bandwidth(
        candidate["effective_bandwidth_hz"],
        f"{kind}.effective_bandwidth_hz",
        nullable=True,
    )
    if candidate["effective_bandwidth_hz"] != bandwidth:
        raise ValueError(f"term-query {kind} bandwidth is not canonical")
    if kind == "sensitivity":
        _require_sha256(
            candidate["opportunity_policy_sha256"],
            "sensitivity.opportunity_policy_sha256",
        )
    query_id = _require_id(candidate["term_query_id"], f"{kind}.term_query_id")
    query = next(
        (row for row in policy["query_specs"] if row["term_query_id"] == query_id),
        None,
    )
    if query is None:
        raise ValueError(f"term-query {kind} receipt references an unknown query")
    unit_key = _require_id(candidate["unit_key"], f"{kind}.unit_key")
    if ":" not in unit_key:
        raise ValueError(f"term-query {kind} unit key is not typed")
    unit_type, unit_id = unit_key.split(":", 1)
    _typed_unit(
        {"unit_type": unit_type, "unit_id": unit_id, "unit_key": unit_key},
        f"{kind}.unit",
        allow_event=True,
    )
    exact = {
        "term_query_id": query["term_query_id"],
        "term_id": query["term_id"],
        "query_cell_key": f"{query_id}:{unit_key}",
        "unit_key": unit_key,
        "claim_kind": query["claim_kind"],
        "family": query["family"],
        "temporal_context": query["temporal_context"],
        "scope_id": query["scope_id"],
        "view_profile_id": query["view_profile_id"],
        "reference_profile_id": query["reference_profile_id"],
        "bandwidth_profile_id": query["bandwidth_profile_id"],
    }
    if any(candidate[key] != expected for key, expected in exact.items()):
        raise ValueError(f"term-query {kind} receipt has a non-exact query binding")
    return candidate


def _validate_v1_chain(
    *,
    v1_source_inventory: object,
    trusted_v1_source_inventory_sha256: str,
    v1_denominator_receipt: object,
    policy_v2: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    v1_policy = _validated_v1_policy(None, None)
    if policy_v2["v1_denominator_policy_sha256"] != v1_policy["policy_sha256"]:
        raise ValueError("v2 policy is not bound to the active trusted v1 policy")
    v1_source = validate_event_findings_denominator_source_inventory(
        v1_source_inventory,
        trusted_source_inventory_sha256=trusted_v1_source_inventory_sha256,
        policy=v1_policy,
        trusted_policy_sha256=str(v1_policy["policy_sha256"]),
    )
    v1_receipt = validate_event_findings_denominator_receipt(
        v1_denominator_receipt,
        source_inventory=v1_source,
        trusted_source_inventory_sha256=trusted_v1_source_inventory_sha256,
        policy=v1_policy,
        trusted_policy_sha256=str(v1_policy["policy_sha256"]),
    )
    if v1_receipt["source_inventory_sha256"] != v1_source["source_inventory_sha256"]:
        raise ValueError("v1 receipt and source inventory trust roots differ")
    if v1_receipt["policy_sha256"] != policy_v2["v1_denominator_policy_sha256"]:
        raise ValueError("v1 receipt is bound to another denominator policy")
    return v1_source, v1_receipt, v1_policy


def _opportunity_policy_sha256(
    query_cell: Mapping[str, object],
    primary_v1_cell: Mapping[str, object],
    *,
    policy_sha256: str,
    term_manifest_sha256: str,
    v1_denominator_receipt_sha256: str,
) -> str:
    seed = {
        "term_query_id": query_cell["term_query_id"],
        "term_id": query_cell["term_id"],
        "query_cell_key": query_cell["query_cell_key"],
        "claim_kind": query_cell["claim_kind"],
        "family": query_cell["family"],
        "temporal_context": query_cell["temporal_context"],
        "scope_id": query_cell["scope_id"],
        "view_profile_id": query_cell["view_profile_id"],
        "reference_profile_id": query_cell["reference_profile_id"],
        "bandwidth_profile_id": query_cell["bandwidth_profile_id"],
        "unit_key": query_cell["unit"]["unit_key"],  # type: ignore[index]
        "primary_v1_cell_key": query_cell["primary_v1_cell_key"],
        "primary_v1_cell_sha256": primary_v1_cell["cell_sha256"],
        "required_interval_union": primary_v1_cell["required_interval_union"],
        "policy_sha256": policy_sha256,
        "term_manifest_sha256": term_manifest_sha256,
        "v1_denominator_receipt_sha256": v1_denominator_receipt_sha256,
    }
    return _canonical_sha256(seed)


def term_query_opportunity_policy_sha256_v2(
    query_cell: Mapping[str, object],
    *,
    primary_v1_cell: Mapping[str, object],
    v1_denominator_receipt_sha256: str,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    term_manifest: Mapping[str, object] | None = None,
    trusted_manifest_sha256: str | None = None,
) -> str:
    """Build the event-specific opportunity-policy identity for sensitivity."""

    denominator_policy = _policy_v2(policy, trusted_policy_sha256)
    manifest = _manifest(term_manifest, trusted_manifest_sha256)
    cell = _expected_static_query_cell(query_cell, denominator_policy)
    if not isinstance(primary_v1_cell, Mapping):
        raise TypeError("primary_v1_cell must be a validated-v1 receipt cell")
    if primary_v1_cell.get("cell_key") != cell["primary_v1_cell_key"]:
        raise ValueError("primary_v1_cell does not match the query primary binding")
    if primary_v1_cell.get("cell_sha256") != _self_hash(
        primary_v1_cell, "cell_sha256"
    ):
        raise ValueError("primary_v1_cell SHA-256 mismatch")
    return _opportunity_policy_sha256(
        cell,
        primary_v1_cell,
        policy_sha256=str(denominator_policy["policy_sha256"]),
        term_manifest_sha256=str(manifest["manifest_sha256"]),
        v1_denominator_receipt_sha256=_require_sha256(
            v1_denominator_receipt_sha256,
            "v1_denominator_receipt_sha256",
        ),
    )


def _normalize_query_availability(
    value: object,
    *,
    tolerance: float,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    expected_fields = {
        "query_cell_key",
        "source_available_interval_union",
        "effective_bandwidth_hz",
        "sample_rate_hz",
        "effective_coupling",
        "processing_disposition",
        "positive_capability_receipt_ids",
        "negative_sensitivity_receipt_ids",
        "technical_failure_receipt_ids",
    }
    if set(value) != expected_fields:
        raise ValueError(f"{context} fields do not match the v2 contract")
    source_union = canonicalize_physical_interval_union(
        value["source_available_interval_union"],  # type: ignore[arg-type]
        tolerance_seconds=tolerance,
    )
    bandwidth = _bandwidth(
        value["effective_bandwidth_hz"],
        f"{context}.effective_bandwidth_hz",
        nullable=True,
    )
    sample_rate_raw = value["sample_rate_hz"]
    sample_rate = (
        None
        if sample_rate_raw is None
        else _finite(sample_rate_raw, f"{context}.sample_rate_hz")
    )
    if sample_rate is not None and sample_rate <= 0.0:
        raise ValueError(f"{context}.sample_rate_hz must be positive")
    coupling = value["effective_coupling"]
    if coupling not in {"dc", "ac", "unknown", "not_applicable"}:
        raise ValueError(f"{context}.effective_coupling is unsupported")
    processing = value["processing_disposition"]
    if processing not in {
        "completed",
        "technical_failure",
        "not_run_unimplemented",
    }:
        raise ValueError(f"{context}.processing_disposition is unsupported")
    return {
        "query_cell_key": _require_id(
            value["query_cell_key"], f"{context}.query_cell_key"
        ),
        "source_available_interval_union": source_union,
        "effective_bandwidth_hz": bandwidth,
        "sample_rate_hz": sample_rate,
        "effective_coupling": coupling,
        "processing_disposition": processing,
        "positive_capability_receipt_ids": _sorted_unique_ids(
            value["positive_capability_receipt_ids"],
            f"{context}.positive_capability_receipt_ids",
        ),
        "negative_sensitivity_receipt_ids": _sorted_unique_ids(
            value["negative_sensitivity_receipt_ids"],
            f"{context}.negative_sensitivity_receipt_ids",
        ),
        "technical_failure_receipt_ids": _sorted_unique_ids(
            value["technical_failure_receipt_ids"],
            f"{context}.technical_failure_receipt_ids",
        ),
    }


def _source_inventory_identifier_seed(value: Mapping[str, object]) -> dict[str, Any]:
    return {
        key: deepcopy(value[key])
        for key in value
        if key not in {
            "schema_version",
            "source_inventory_id",
            "source_inventory_sha256",
        }
    }


def build_event_findings_term_query_source_inventory_v2(
    *,
    v1_source_inventory: object,
    trusted_v1_source_inventory_sha256: str,
    v1_denominator_receipt: object,
    query_availability: Sequence[Mapping[str, object]],
    capability_receipts: Sequence[Mapping[str, object]] = (),
    sensitivity_receipts: Sequence[Mapping[str, object]] = (),
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    term_manifest: Mapping[str, object] | None = None,
    trusted_manifest_sha256: str | None = None,
    source_firewall: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a candidate-blind v2 inventory rooted in an externally trusted v1 chain."""

    denominator_policy = _policy_v2(policy, trusted_policy_sha256)
    manifest = _manifest(term_manifest, trusted_manifest_sha256)
    v1_source, v1_receipt, _ = _validate_v1_chain(
        v1_source_inventory=v1_source_inventory,
        trusted_v1_source_inventory_sha256=trusted_v1_source_inventory_sha256,
        v1_denominator_receipt=v1_denominator_receipt,
        policy_v2=denominator_policy,
    )
    tolerance = float(
        denominator_policy["interval_union_policy"]["comparison_tolerance_seconds"]
    )
    availability = sorted(
        (
            _normalize_query_availability(
                row, tolerance=tolerance, context="query_availability"
            )
            for row in query_availability
        ),
        key=lambda row: str(row["query_cell_key"]),
    )
    capabilities = sorted(
        (
            _validate_qualification_receipt_v2(
                row,
                kind="capability",
                policy=denominator_policy,
                manifest=manifest,
            )
            for row in capability_receipts
        ),
        key=lambda row: str(row["receipt_id"]),
    )
    sensitivities = sorted(
        (
            _validate_qualification_receipt_v2(
                row,
                kind="sensitivity",
                policy=denominator_policy,
                manifest=manifest,
            )
            for row in sensitivity_receipts
        ),
        key=lambda row: str(row["receipt_id"]),
    )
    firewall = (
        {key: False for key in sorted(_SOURCE_FIREWALL_KEYS)}
        if source_firewall is None
        else deepcopy(dict(source_firewall))
    )
    replay_binding = {
        "enumerator_code_sha256": event_findings_term_query_enumerator_code_sha256_v2(),
        "v1_source_inventory_id": v1_source["source_inventory_id"],
        "v1_source_inventory_sha256": v1_source["source_inventory_sha256"],
        "v1_denominator_receipt_id": v1_receipt["receipt_id"],
        "v1_denominator_receipt_sha256": v1_receipt["receipt_sha256"],
        "v1_denominator_policy_sha256": v1_receipt["policy_sha256"],
        "host_trust_required": True,
        "exact_replay_required": True,
        "candidate_blind": True,
    }
    inventory: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_TERM_QUERY_SOURCE_INVENTORY_SCHEMA_VERSION_V2,
        "source_inventory_id": "PENDING",
        "record_id": v1_receipt["record_id"],
        "event_id": v1_receipt["event_id"],
        "canonical_signal_sha256": v1_receipt["canonical_signal_sha256"],
        "policy_id": denominator_policy["policy_id"],
        "policy_sha256": denominator_policy["policy_sha256"],
        "term_manifest_id": manifest["manifest_id"],
        "term_manifest_sha256": manifest["manifest_sha256"],
        "atom_roster_policy_sha256": denominator_policy["atom_roster_policy_sha256"],
        "v1_source_inventory_id": v1_source["source_inventory_id"],
        "v1_source_inventory_sha256": v1_source["source_inventory_sha256"],
        "v1_denominator_receipt_id": v1_receipt["receipt_id"],
        "v1_denominator_receipt_sha256": v1_receipt["receipt_sha256"],
        "typed_expected_units": deepcopy(v1_receipt["typed_expected_units"]),
        "query_availability": availability,
        "capability_receipts": capabilities,
        "sensitivity_receipts": sensitivities,
        "replay_binding": replay_binding,
        "source_firewall": firewall,
        "source_inventory_sha256": "0" * 64,
    }
    inventory["source_inventory_id"] = (
        "EEGTQDENOMSRC-"
        + _canonical_sha256(_source_inventory_identifier_seed(inventory))[:24]
    )
    inventory["source_inventory_sha256"] = _self_hash(
        inventory, "source_inventory_sha256"
    )
    return validate_event_findings_term_query_source_inventory_v2(
        inventory,
        trusted_source_inventory_sha256=str(inventory["source_inventory_sha256"]),
        v1_source_inventory=v1_source,
        trusted_v1_source_inventory_sha256=trusted_v1_source_inventory_sha256,
        v1_denominator_receipt=v1_receipt,
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
        term_manifest=manifest,
        trusted_manifest_sha256=str(manifest["manifest_sha256"]),
    )


def validate_event_findings_term_query_source_inventory_v2(
    value: object,
    *,
    trusted_source_inventory_sha256: str,
    v1_source_inventory: object,
    trusted_v1_source_inventory_sha256: str,
    v1_denominator_receipt: object,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    term_manifest: Mapping[str, object] | None = None,
    trusted_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate inventory closure against externally retained v1 and v2 anchors."""

    if type(value) is not dict:
        raise TypeError("event Findings term-query source inventory v2 must be an object")
    candidate = deepcopy(value)
    allowed_fields = {
        "schema_version",
        "source_inventory_id",
        "record_id",
        "event_id",
        "canonical_signal_sha256",
        "policy_id",
        "policy_sha256",
        "term_manifest_id",
        "term_manifest_sha256",
        "atom_roster_policy_sha256",
        "v1_source_inventory_id",
        "v1_source_inventory_sha256",
        "v1_denominator_receipt_id",
        "v1_denominator_receipt_sha256",
        "typed_expected_units",
        "query_availability",
        "capability_receipts",
        "sensitivity_receipts",
        "replay_binding",
        "source_firewall",
        "source_inventory_sha256",
    }
    if set(candidate) != allowed_fields:
        raise ValueError("term-query source inventory fields do not match v2")
    if candidate["schema_version"] != EVENT_FINDINGS_TERM_QUERY_SOURCE_INVENTORY_SCHEMA_VERSION_V2:
        raise ValueError("term-query source inventory schema mismatch")
    expected_hash = _self_hash(candidate, "source_inventory_sha256")
    if candidate["source_inventory_sha256"] != expected_hash:
        raise ValueError("term-query source inventory SHA-256 mismatch")
    if expected_hash != _require_sha256(
        trusted_source_inventory_sha256, "trusted_source_inventory_sha256"
    ):
        raise ValueError("term-query source inventory is not host trusted")

    denominator_policy = _policy_v2(policy, trusted_policy_sha256)
    manifest = _manifest(term_manifest, trusted_manifest_sha256)
    v1_source, v1_receipt, _ = _validate_v1_chain(
        v1_source_inventory=v1_source_inventory,
        trusted_v1_source_inventory_sha256=trusted_v1_source_inventory_sha256,
        v1_denominator_receipt=v1_denominator_receipt,
        policy_v2=denominator_policy,
    )
    exact_identity = {
        "record_id": v1_receipt["record_id"],
        "event_id": v1_receipt["event_id"],
        "canonical_signal_sha256": v1_receipt["canonical_signal_sha256"],
        "policy_id": denominator_policy["policy_id"],
        "policy_sha256": denominator_policy["policy_sha256"],
        "term_manifest_id": manifest["manifest_id"],
        "term_manifest_sha256": manifest["manifest_sha256"],
        "atom_roster_policy_sha256": denominator_policy["atom_roster_policy_sha256"],
        "v1_source_inventory_id": v1_source["source_inventory_id"],
        "v1_source_inventory_sha256": v1_source["source_inventory_sha256"],
        "v1_denominator_receipt_id": v1_receipt["receipt_id"],
        "v1_denominator_receipt_sha256": v1_receipt["receipt_sha256"],
        "typed_expected_units": v1_receipt["typed_expected_units"],
    }
    if any(candidate[key] != expected for key, expected in exact_identity.items()):
        raise ValueError("term-query source inventory does not match the exact v1 chain")
    _require_id(candidate["source_inventory_id"], "source_inventory_id")
    _require_id(candidate["record_id"], "record_id")
    _require_id(candidate["event_id"], "event_id")
    _require_sha256(candidate["canonical_signal_sha256"], "canonical_signal_sha256")

    expected_cells = enumerate_event_findings_term_query_cells_v2(
        candidate["typed_expected_units"],
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
    )
    expected_by_key = {str(row["query_cell_key"]): row for row in expected_cells}
    tolerance = float(
        denominator_policy["interval_union_policy"]["comparison_tolerance_seconds"]
    )
    availability = [
        _normalize_query_availability(
            row, tolerance=tolerance, context="query_availability"
        )
        for row in candidate["query_availability"]
    ]
    if candidate["query_availability"] != availability:
        raise ValueError("term-query availability rows are not canonical")
    if availability != sorted(availability, key=lambda row: row["query_cell_key"]):
        raise ValueError("term-query availability rows must be canonically sorted")
    availability_keys = [str(row["query_cell_key"]) for row in availability]
    if len(availability_keys) != len(set(availability_keys)):
        raise ValueError("term-query availability has duplicate query cells")
    if set(availability_keys) != set(expected_by_key):
        raise ValueError("term-query availability does not exactly cover expected cells")

    capabilities = [
        _validate_qualification_receipt_v2(
            row, kind="capability", policy=denominator_policy, manifest=manifest
        )
        for row in candidate["capability_receipts"]
    ]
    sensitivities = [
        _validate_qualification_receipt_v2(
            row, kind="sensitivity", policy=denominator_policy, manifest=manifest
        )
        for row in candidate["sensitivity_receipts"]
    ]
    for name, rows in (
        ("capability", capabilities),
        ("sensitivity", sensitivities),
    ):
        if rows != sorted(rows, key=lambda row: row["receipt_id"]):
            raise ValueError(f"term-query {name} receipts must be canonically sorted")
        ids = [str(row["receipt_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"term-query {name} receipt IDs must be unique")
    capability_by_id = {str(row["receipt_id"]): row for row in capabilities}
    sensitivity_by_id = {str(row["receipt_id"]): row for row in sensitivities}
    v1_cell_by_key = {
        str(row["cell_key"]): row for row in v1_receipt["cells"]
    }
    referenced_capabilities: set[str] = set()
    referenced_sensitivities: set[str] = set()
    for row in availability:
        cell_key = str(row["query_cell_key"])
        expected = expected_by_key[cell_key]
        primary = v1_cell_by_key[str(expected["primary_v1_cell_key"])]
        if not _union_is_subset(
            row["source_available_interval_union"],
            primary["evaluable_interval_union"],
            tolerance,
        ):
            raise ValueError(
                f"{cell_key}: query opportunity exceeds the primary v1 cell"
            )
        capability_ids = list(row["positive_capability_receipt_ids"])
        sensitivity_ids = list(row["negative_sensitivity_receipt_ids"])
        failure_ids = list(row["technical_failure_receipt_ids"])
        if expected["implementation_status"] == "unimplemented_not_evaluable":
            if (
                row["source_available_interval_union"]
                or row["effective_bandwidth_hz"] is not None
                or row["sample_rate_hz"] is not None
                or row["effective_coupling"] != "not_applicable"
                or row["processing_disposition"] != "not_run_unimplemented"
                or capability_ids
                or sensitivity_ids
                or failure_ids
            ):
                raise ValueError(
                    f"{cell_key}: unimplemented query must remain empty and not evaluable"
                )
            continue
        primary_failures = list(primary["technical_failure_receipt_ids"])
        if primary_failures:
            if (
                row["processing_disposition"] != "technical_failure"
                or failure_ids != primary_failures
                or row["source_available_interval_union"]
            ):
                raise ValueError(
                    f"{cell_key}: query failure must exactly inherit the v1 failure ledger"
                )
        elif failure_ids or row["processing_disposition"] == "technical_failure":
            raise ValueError(
                f"{cell_key}: query cannot invent a technical-failure receipt"
            )
        elif row["processing_disposition"] != "completed":
            raise ValueError(f"{cell_key}: implemented query processing must complete")

        opportunity_hash = _opportunity_policy_sha256(
            expected,
            primary,
            policy_sha256=str(denominator_policy["policy_sha256"]),
            term_manifest_sha256=str(manifest["manifest_sha256"]),
            v1_denominator_receipt_sha256=str(v1_receipt["receipt_sha256"]),
        )
        exact_common = {
            "term_query_id": expected["term_query_id"],
            "term_id": expected["term_id"],
            "query_cell_key": expected["query_cell_key"],
            "unit_key": expected["unit"]["unit_key"],
            "claim_kind": expected["claim_kind"],
            "family": expected["family"],
            "temporal_context": expected["temporal_context"],
            "scope_id": expected["scope_id"],
            "view_profile_id": expected["view_profile_id"],
            "reference_profile_id": expected["reference_profile_id"],
            "bandwidth_profile_id": expected["bandwidth_profile_id"],
            "effective_bandwidth_hz": row["effective_bandwidth_hz"],
            "policy_sha256": denominator_policy["policy_sha256"],
            "term_manifest_sha256": manifest["manifest_sha256"],
        }
        for receipt_id in capability_ids:
            if receipt_id not in capability_by_id:
                raise ValueError(f"{cell_key}: unknown capability receipt")
            receipt = capability_by_id[receipt_id]
            if any(receipt[key] != expected_value for key, expected_value in exact_common.items()):
                raise ValueError(f"{cell_key}: capability receipt does not exactly match")
            referenced_capabilities.add(receipt_id)
        for receipt_id in sensitivity_ids:
            if receipt_id not in sensitivity_by_id:
                raise ValueError(f"{cell_key}: unknown sensitivity receipt")
            receipt = sensitivity_by_id[receipt_id]
            if any(receipt[key] != expected_value for key, expected_value in exact_common.items()):
                raise ValueError(f"{cell_key}: sensitivity receipt does not exactly match")
            if receipt["opportunity_policy_sha256"] != opportunity_hash:
                raise ValueError(
                    f"{cell_key}: sensitivity receipt opportunity-policy mismatch"
                )
            referenced_sensitivities.add(receipt_id)
        if expected["negative_semantics"] == "not_applicable" and sensitivity_ids:
            raise ValueError(f"{cell_key}: a measurement-only query cannot claim sensitivity")

    if referenced_capabilities != set(capability_by_id):
        raise ValueError("term-query capability receipt collection is not exactly referenced")
    if referenced_sensitivities != set(sensitivity_by_id):
        raise ValueError("term-query sensitivity receipt collection is not exactly referenced")

    expected_replay = {
        "enumerator_code_sha256": event_findings_term_query_enumerator_code_sha256_v2(),
        "v1_source_inventory_id": v1_source["source_inventory_id"],
        "v1_source_inventory_sha256": v1_source["source_inventory_sha256"],
        "v1_denominator_receipt_id": v1_receipt["receipt_id"],
        "v1_denominator_receipt_sha256": v1_receipt["receipt_sha256"],
        "v1_denominator_policy_sha256": v1_receipt["policy_sha256"],
        "host_trust_required": True,
        "exact_replay_required": True,
        "candidate_blind": True,
    }
    if candidate["replay_binding"] != expected_replay:
        raise ValueError("term-query source inventory replay binding is not exact")
    firewall = candidate["source_firewall"]
    if not isinstance(firewall, Mapping) or set(firewall) != _SOURCE_FIREWALL_KEYS:
        raise ValueError("term-query source inventory firewall is incomplete")
    if any(value is not False for value in firewall.values()):
        raise ValueError("term-query source inventory used a forbidden source")
    expected_id = (
        "EEGTQDENOMSRC-"
        + _canonical_sha256(_source_inventory_identifier_seed(candidate))[:24]
    )
    if candidate["source_inventory_id"] != expected_id:
        raise ValueError("term-query source inventory ID is not content addressed")
    return candidate


def _bandwidth_covers(
    actual: Sequence[float] | None,
    required: Sequence[object] | None,
    tolerance: float,
) -> bool:
    if required is None:
        return True
    if actual is None:
        return False
    required_bandwidth = _bandwidth(required, "required_bandwidth", nullable=False)
    assert required_bandwidth is not None
    return (
        actual[0] <= required_bandwidth[0] + tolerance
        and actual[1] >= required_bandwidth[1] - tolerance
    )


def _coupling_covers(actual: str, required: str) -> bool:
    if required == "not_applicable":
        return True
    if required == "dc":
        return actual == "dc"
    return actual in {"ac", "dc"}


def materialize_event_findings_term_query_denominator_receipt_v2(
    source_inventory: object,
    *,
    trusted_source_inventory_sha256: str,
    v1_source_inventory: object,
    trusted_v1_source_inventory_sha256: str,
    v1_denominator_receipt: object,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    term_manifest: Mapping[str, object] | None = None,
    trusted_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize the complete, gap-preserving term-query opportunity grid."""

    denominator_policy = _policy_v2(policy, trusted_policy_sha256)
    manifest = _manifest(term_manifest, trusted_manifest_sha256)
    source = validate_event_findings_term_query_source_inventory_v2(
        source_inventory,
        trusted_source_inventory_sha256=trusted_source_inventory_sha256,
        v1_source_inventory=v1_source_inventory,
        trusted_v1_source_inventory_sha256=trusted_v1_source_inventory_sha256,
        v1_denominator_receipt=v1_denominator_receipt,
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
        term_manifest=manifest,
        trusted_manifest_sha256=str(manifest["manifest_sha256"]),
    )
    _, v1_receipt, _ = _validate_v1_chain(
        v1_source_inventory=v1_source_inventory,
        trusted_v1_source_inventory_sha256=trusted_v1_source_inventory_sha256,
        v1_denominator_receipt=v1_denominator_receipt,
        policy_v2=denominator_policy,
    )
    expected_cells = enumerate_event_findings_term_query_cells_v2(
        source["typed_expected_units"],
        policy=denominator_policy,
        trusted_policy_sha256=str(denominator_policy["policy_sha256"]),
    )
    availability_by_key = {
        str(row["query_cell_key"]): row for row in source["query_availability"]
    }
    v1_cell_by_key = {
        str(row["cell_key"]): row for row in v1_receipt["cells"]
    }
    capability_by_id = {
        str(row["receipt_id"]): row for row in source["capability_receipts"]
    }
    sensitivity_by_id = {
        str(row["receipt_id"]): row for row in source["sensitivity_receipts"]
    }
    tolerance = float(
        denominator_policy["interval_union_policy"]["comparison_tolerance_seconds"]
    )

    cells: list[dict[str, Any]] = []
    for expected in expected_cells:
        key = str(expected["query_cell_key"])
        availability = availability_by_key[key]
        primary = v1_cell_by_key[str(expected["primary_v1_cell_key"])]
        required_union = deepcopy(primary["required_interval_union"])
        source_union = deepcopy(availability["source_available_interval_union"])
        required_seconds = physical_interval_union_seconds(
            required_union, tolerance_seconds=tolerance
        )
        source_seconds = physical_interval_union_seconds(
            source_union, tolerance_seconds=tolerance
        )
        capability_ids = list(availability["positive_capability_receipt_ids"])
        sensitivity_ids = list(availability["negative_sensitivity_receipt_ids"])
        failure_ids = list(availability["technical_failure_receipt_ids"])
        qualified_capability_ids = [
            receipt_id
            for receipt_id in capability_ids
            if capability_by_id[receipt_id]["qualification_passed"] is True
            and capability_by_id[receipt_id]["patient_disjoint"] is True
            and capability_by_id[receipt_id]["frozen_before_inference"] is True
        ]
        qualified_sensitivity_ids = [
            receipt_id
            for receipt_id in sensitivity_ids
            if sensitivity_by_id[receipt_id]["qualification_passed"] is True
            and sensitivity_by_id[receipt_id]["patient_disjoint"] is True
            and sensitivity_by_id[receipt_id]["frozen_before_inference"] is True
        ]
        bandwidth_ok = _bandwidth_covers(
            availability["effective_bandwidth_hz"],
            expected["required_bandwidth_hz"],
            tolerance,
        )
        minimum_rate = expected["minimum_sample_rate_hz"]
        sample_rate_ok = (
            minimum_rate is None
            or (
                availability["sample_rate_hz"] is not None
                and availability["sample_rate_hz"] + tolerance
                >= float(minimum_rate)
            )
        )
        coupling_ok = _coupling_covers(
            str(availability["effective_coupling"]),
            str(expected["required_coupling"]),
        )
        reasons: list[str] = []
        if expected["implementation_status"] == "unimplemented_not_evaluable":
            status = "not_evaluable"
            reasons.append("frozen_query_unimplemented")
        elif failure_ids:
            status = "not_evaluable"
            reasons.append("trusted_v1_technical_failure")
        elif availability["processing_disposition"] != "completed":
            status = "not_evaluable"
            reasons.append("query_processing_not_completed")
        elif not qualified_capability_ids:
            status = "not_evaluable"
            reasons.append("matching_capability_receipt_missing")
        elif not bandwidth_ok:
            status = "not_evaluable"
            reasons.append("effective_bandwidth_insufficient")
        elif not sample_rate_ok:
            status = "not_evaluable"
            reasons.append("sample_rate_insufficient")
        elif not coupling_ok:
            status = "not_evaluable"
            reasons.append("acquisition_coupling_insufficient")
        elif required_seconds <= tolerance:
            status = "not_evaluable"
            reasons.append("required_scope_empty_or_censored")
        elif source_seconds <= tolerance:
            status = "not_evaluable"
            reasons.append("no_source_available_signal_support")
        elif _unions_equal(source_union, required_union, tolerance):
            status = "sufficient"
            reasons.append("complete_gap_preserving_query_opportunity")
        else:
            status = "limited"
            reasons.append("partial_gap_preserving_query_opportunity")

        if status == "not_evaluable":
            evaluable_union: list[dict[str, float]] = []
            evaluable_seconds = 0.0
            coverage = 0.0
        else:
            evaluable_union = source_union
            evaluable_seconds = source_seconds
            coverage = min(1.0, source_seconds / required_seconds)
        negative_eligible = bool(
            status == "sufficient"
            and availability["processing_disposition"] == "completed"
            and expected["negative_semantics"] != "not_applicable"
            and qualified_sensitivity_ids
        )
        if status == "sufficient" and expected["negative_semantics"] != "not_applicable":
            reasons.append(
                "matching_sensitivity_receipt_available"
                if negative_eligible
                else "matching_sensitivity_receipt_missing"
            )
        reasons.append("clinical_absence_never_authorized_by_denominator")

        cell: dict[str, Any] = {
            "query_cell_key": key,
            "term_query_id": expected["term_query_id"],
            "term_id": expected["term_id"],
            "claim_kind": expected["claim_kind"],
            "family": expected["family"],
            "temporal_context": expected["temporal_context"],
            "view_profile_id": expected["view_profile_id"],
            "reference_profile_id": expected["reference_profile_id"],
            "bandwidth_profile_id": expected["bandwidth_profile_id"],
            "scope_id": expected["scope_id"],
            "unit": deepcopy(expected["unit"]),
            "primary_roster_item_kind": expected["primary_roster_item_kind"],
            "primary_roster_item_id": expected["primary_roster_item_id"],
            "primary_v1_cell_key": expected["primary_v1_cell_key"],
            "required_interval_union": required_union,
            "source_available_interval_union": source_union,
            "evaluable_interval_union": evaluable_union,
            "required_seconds": required_seconds,
            "source_available_seconds": source_seconds,
            "evaluable_seconds": evaluable_seconds,
            "coverage_fraction": coverage,
            "effective_bandwidth_hz": deepcopy(
                availability["effective_bandwidth_hz"]
            ),
            "sample_rate_hz": availability["sample_rate_hz"],
            "effective_coupling": availability["effective_coupling"],
            "opportunity_status": status,
            "processing_disposition": availability["processing_disposition"],
            "positive_capability_receipt_ids": capability_ids,
            "negative_sensitivity_receipt_ids": sensitivity_ids,
            "technical_failure_receipt_ids": failure_ids,
            "negative_opportunity_eligible": negative_eligible,
            "reason_codes": sorted(set(reasons)),
            "cell_sha256": "0" * 64,
        }
        cell["cell_sha256"] = _self_hash(cell, "cell_sha256")
        cells.append(cell)

    cells.sort(key=lambda row: str(row["query_cell_key"]))
    expected_query_ids = list(_EXPECTED_TERM_QUERY_IDS_V2)
    materialized_query_ids = sorted({str(row["term_query_id"]) for row in cells})
    expected_cell_keys = [str(row["query_cell_key"]) for row in expected_cells]
    materialized_cell_keys = [str(row["query_cell_key"]) for row in cells]
    event_global_count = sum(row["unit"]["unit_type"] == "event" for row in cells)
    physical_count = len(cells) - event_global_count
    summary = {
        "expected_base_query_count": 41,
        "materialized_base_query_count": len(materialized_query_ids),
        "expected_cell_count": len(expected_cell_keys),
        "materialized_cell_count": len(materialized_cell_keys),
        "expected_event_global_cell_count": event_global_count,
        "materialized_event_global_cell_count": event_global_count,
        "expected_physical_unit_cell_count": physical_count,
        "materialized_physical_unit_cell_count": physical_count,
        "expected_unit_count": len(source["typed_expected_units"]),
        "sufficient_count": sum(row["opportunity_status"] == "sufficient" for row in cells),
        "limited_count": sum(row["opportunity_status"] == "limited" for row in cells),
        "not_evaluable_count": sum(
            row["opportunity_status"] == "not_evaluable" for row in cells
        ),
        "technical_failure_count": sum(
            row["processing_disposition"] == "technical_failure" for row in cells
        ),
        "negative_opportunity_eligible_count": sum(
            bool(row["negative_opportunity_eligible"]) for row in cells
        ),
        "expected_query_ids_sha256": _canonical_sha256(expected_query_ids),
        "materialized_query_ids_sha256": _canonical_sha256(materialized_query_ids),
        "expected_cell_keys_sha256": _canonical_sha256(expected_cell_keys),
        "materialized_cell_keys_sha256": _canonical_sha256(materialized_cell_keys),
        "all_expected_queries_materialized": expected_query_ids == materialized_query_ids,
        "all_expected_cells_materialized": expected_cell_keys == materialized_cell_keys,
        "gap_preserving": True,
        "double_counting_allowed": False,
        "clinical_absence_authorized": False,
        "report_promotion_authorized": False,
        "clinical_correctness_claimed": False,
    }
    replay_binding = {
        "enumerator_code_sha256": source["replay_binding"]["enumerator_code_sha256"],
        "trusted_source_inventory_sha256": source["source_inventory_sha256"],
        "v1_source_inventory_id": source["v1_source_inventory_id"],
        "v1_source_inventory_sha256": source["v1_source_inventory_sha256"],
        "v1_denominator_receipt_id": source["v1_denominator_receipt_id"],
        "v1_denominator_receipt_sha256": source["v1_denominator_receipt_sha256"],
        "host_trust_required": True,
        "exact_replay_required": True,
        "candidate_blind": True,
    }
    receipt_seed = {
        "method_id": EVENT_FINDINGS_TERM_QUERY_METHOD_ID_V2,
        "record_id": source["record_id"],
        "event_id": source["event_id"],
        "canonical_signal_sha256": source["canonical_signal_sha256"],
        "policy_sha256": source["policy_sha256"],
        "term_manifest_sha256": source["term_manifest_sha256"],
        "source_inventory_sha256": source["source_inventory_sha256"],
        "v1_source_inventory_sha256": source["v1_source_inventory_sha256"],
        "v1_denominator_receipt_sha256": source["v1_denominator_receipt_sha256"],
    }
    receipt: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_TERM_QUERY_RECEIPT_SCHEMA_VERSION_V2,
        "receipt_id": f"EEGTQDENOM-{_canonical_sha256(receipt_seed)[:24]}",
        "method_id": EVENT_FINDINGS_TERM_QUERY_METHOD_ID_V2,
        "record_id": source["record_id"],
        "event_id": source["event_id"],
        "canonical_signal_sha256": source["canonical_signal_sha256"],
        "policy_id": source["policy_id"],
        "policy_sha256": source["policy_sha256"],
        "term_manifest_id": source["term_manifest_id"],
        "term_manifest_sha256": source["term_manifest_sha256"],
        "atom_roster_policy_sha256": source["atom_roster_policy_sha256"],
        "v1_source_inventory_id": source["v1_source_inventory_id"],
        "v1_source_inventory_sha256": source["v1_source_inventory_sha256"],
        "v1_denominator_receipt_id": source["v1_denominator_receipt_id"],
        "v1_denominator_receipt_sha256": source["v1_denominator_receipt_sha256"],
        "source_inventory_id": source["source_inventory_id"],
        "source_inventory_sha256": source["source_inventory_sha256"],
        "typed_expected_units": deepcopy(source["typed_expected_units"]),
        "replay_binding": replay_binding,
        "source_firewall": deepcopy(source["source_firewall"]),
        "cells": cells,
        "summary": summary,
        "clinical_absence_authorized": False,
        "report_promotion_authorized": False,
        "clinical_correctness_claimed": False,
        "receipt_sha256": "0" * 64,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return receipt


def validate_event_findings_term_query_denominator_receipt_v2(
    value: object,
    *,
    source_inventory: object,
    trusted_source_inventory_sha256: str,
    v1_source_inventory: object,
    trusted_v1_source_inventory_sha256: str,
    v1_denominator_receipt: object,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    term_manifest: Mapping[str, object] | None = None,
    trusted_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate self-hashes and exact replay from external v1/v2 trust roots."""

    if type(value) is not dict:
        raise TypeError("event Findings term-query denominator receipt v2 must be an object")
    candidate = deepcopy(value)
    allowed_fields = {
        "schema_version",
        "receipt_id",
        "method_id",
        "record_id",
        "event_id",
        "canonical_signal_sha256",
        "policy_id",
        "policy_sha256",
        "term_manifest_id",
        "term_manifest_sha256",
        "atom_roster_policy_sha256",
        "v1_source_inventory_id",
        "v1_source_inventory_sha256",
        "v1_denominator_receipt_id",
        "v1_denominator_receipt_sha256",
        "source_inventory_id",
        "source_inventory_sha256",
        "typed_expected_units",
        "replay_binding",
        "source_firewall",
        "cells",
        "summary",
        "clinical_absence_authorized",
        "report_promotion_authorized",
        "clinical_correctness_claimed",
        "receipt_sha256",
    }
    if set(candidate) != allowed_fields:
        raise ValueError("term-query denominator receipt fields do not match v2")
    if candidate["schema_version"] != EVENT_FINDINGS_TERM_QUERY_RECEIPT_SCHEMA_VERSION_V2:
        raise ValueError("term-query denominator receipt schema mismatch")
    if candidate["method_id"] != EVENT_FINDINGS_TERM_QUERY_METHOD_ID_V2:
        raise ValueError("term-query denominator receipt method mismatch")
    if candidate["receipt_sha256"] != _self_hash(candidate, "receipt_sha256"):
        raise ValueError("term-query denominator receipt SHA-256 mismatch")
    for cell in candidate["cells"]:
        if not isinstance(cell, Mapping) or cell.get("cell_sha256") != _self_hash(
            cell, "cell_sha256"
        ):
            raise ValueError("term-query denominator receipt has a cell SHA mismatch")
    if (
        candidate["clinical_absence_authorized"] is not False
        or candidate["report_promotion_authorized"] is not False
        or candidate["clinical_correctness_claimed"] is not False
    ):
        raise ValueError("term-query denominator cannot authorize clinical/report claims")
    expected = materialize_event_findings_term_query_denominator_receipt_v2(
        source_inventory,
        trusted_source_inventory_sha256=trusted_source_inventory_sha256,
        v1_source_inventory=v1_source_inventory,
        trusted_v1_source_inventory_sha256=trusted_v1_source_inventory_sha256,
        v1_denominator_receipt=v1_denominator_receipt,
        policy=policy,
        trusted_policy_sha256=trusted_policy_sha256,
        term_manifest=term_manifest,
        trusted_manifest_sha256=trusted_manifest_sha256,
    )
    if candidate != expected:
        raise ValueError(
            "term-query denominator receipt does not match exact host-side replay"
        )
    return candidate


__all__ = [
    "DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_PATH_V2",
    "DEFAULT_EVENT_FINDINGS_TERM_QUERY_POLICY_SHA256_V2",
    "DEFAULT_FINDING_TERM_MANIFEST_PATH_V2",
    "DEFAULT_FINDING_TERM_MANIFEST_SHA256_V2",
    "EVENT_FINDINGS_TERM_QUERY_METHOD_ID_V2",
    "build_event_findings_term_query_source_inventory_v2",
    "build_term_query_capability_receipt_v2",
    "build_term_query_sensitivity_receipt_v2",
    "enumerate_event_findings_term_query_cells_v2",
    "event_findings_term_query_enumerator_code_sha256_v2",
    "load_clinical_eeg_finding_term_manifest_v2",
    "load_event_findings_term_query_denominator_policy_v2",
    "materialize_event_findings_term_query_denominator_receipt_v2",
    "term_query_opportunity_policy_sha256_v2",
    "validate_clinical_eeg_finding_term_manifest_v2",
    "validate_event_findings_term_query_denominator_policy_v2",
    "validate_event_findings_term_query_denominator_receipt_v2",
    "validate_event_findings_term_query_source_inventory_v2",
]
