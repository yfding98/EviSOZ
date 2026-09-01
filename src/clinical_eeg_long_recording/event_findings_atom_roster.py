"""Closed, source-bound atom roster for one EEG event Findings graph.

``event_eeg_findings_v3`` is intentionally a frozen evidence wire.  It is
strict about rows that are present, but it does not define the denominator of
clinical questions that a producer was expected to answer.  This additive
shadow sidecar closes that denominator without changing the v2/v3 wire.

The receipt proves structural accounting only.  It cannot qualify a clinical
term, establish clinical correctness, or authorize cortical SOZ/EZ language.
It is not connected to the private report route.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .event_findings_v3_validation import (
    validate_event_eeg_findings_v3_payload,
)


EVENT_FINDINGS_ATOM_ROSTER_POLICY_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_atom_roster_policy_v1"
)
EVENT_FINDINGS_ATOM_ROSTER_RECEIPT_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_atom_roster_receipt_v1"
)
EVENT_FINDINGS_ATOM_ROSTER_METHOD_ID = (
    "EEG-ONLY-CLOSED-EVENT-FINDINGS-ATOM-ROSTER-V1"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_FINDINGS_ATOM_ROSTER_POLICY_PATH = (
    _REPOSITORY_ROOT
    / "configs"
    / "clinical_eeg_event_findings_atom_roster_v1.json"
)
EVENT_FINDINGS_ATOM_ROSTER_POLICY_SCHEMA_PATH = (
    _REPOSITORY_ROOT
    / "schemas"
    / "clinical_eeg_event_findings_atom_roster_policy_v1.schema.json"
)
EVENT_FINDINGS_ATOM_ROSTER_RECEIPT_SCHEMA_PATH = (
    _REPOSITORY_ROOT
    / "schemas"
    / "clinical_eeg_event_findings_atom_roster_receipt_v1.schema.json"
)

_STATUS_PRIORITY = {
    "not_evaluable": 0,
    "absent_with_opportunity": 1,
    "uncertain": 2,
    "present": 3,
}
_ASSERTION_PRIORITY = {
    None: -1,
    "measured": 0,
    "model_candidate": 1,
    "report_eligible_automated": 2,
}
_VALIDATION_KWARGS = {
    "trusted_producer_receipts",
    "trusted_calibration_receipts",
    "trusted_capability_qualification_receipts",
    "trusted_sensitivity_receipts",
    "trusted_term_decision_receipts",
    "trusted_registry_bindings",
}

_RETIRED_AMBIGUOUS_SHARP_TERMS = frozenset({"sharp_transient_candidate"})
_INTERICTAL_SHARP_TERMS = frozenset(
    {"interictal_epileptiform_discharge", "sharp_wave", "spike"}
)
_ICTAL_SHARP_COMPONENT_TERMS = frozenset(
    {"ictal_sharp_contoured_component_candidate"}
)
_LEGACY_SHARED_ACQUISITION_ROSTER_ID = (
    "a_acquisition_sensitive_pattern_instances"
)
_ACQUISITION_SENSITIVE_ROSTER_SPECS = {
    "a_dc_shift_pattern_instances": {
        "term_id": "dc_shift_candidate",
        "family": "spectral",
        "surface_frame_id": "FRAME-DC-SHIFT-PATTERN-QUALIFIED-V1",
    },
    "a_hfo_pattern_instances": {
        "term_id": "high_frequency_oscillation",
        "family": "high_frequency",
        "surface_frame_id": "FRAME-HFO-PATTERN-QUALIFIED-V1",
    },
    "a_lvfa_pattern_instances": {
        "term_id": "low_voltage_fast_activity",
        "family": "morphology",
        "surface_frame_id": "FRAME-LVFA-PATTERN-QUALIFIED-V1",
    },
    "a_very_slow_activity_pattern_instances": {
        "term_id": "very_slow_activity_candidate",
        "family": "spectral",
        "surface_frame_id": "FRAME-VERY-SLOW-ACTIVITY-PATTERN-QUALIFIED-V1",
    },
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
    validator = Draft202012Validator(_read_json(schema_path))
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
    rendered: list[str] = []
    for error in errors[:12]:
        path = "/" + "/".join(str(part) for part in error.path)
        rendered.append(f"{path}: {error.message}")
    if len(errors) > 12:
        rendered.append(f"... {len(errors) - 12} more error(s)")
    return rendered


def _validated_findings(
    value: object,
    validation_kwargs: Mapping[str, object] | None,
) -> dict[str, Any]:
    kwargs = dict(validation_kwargs or {})
    unexpected = sorted(set(kwargs) - _VALIDATION_KWARGS)
    if unexpected:
        raise ValueError(
            "unsupported event Findings validation kwargs: "
            + ", ".join(unexpected)
        )
    return validate_event_eeg_findings_v3_payload(value, **kwargs)


def validate_event_findings_atom_roster_policy(
    value: object,
    *,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a roster policy, its self-hash, and semantic invariants."""

    if type(value) is not dict:
        raise TypeError("event Findings atom-roster policy must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate,
        EVENT_FINDINGS_ATOM_ROSTER_POLICY_SCHEMA_PATH,
    )
    if errors:
        raise ValueError(
            "event Findings atom-roster policy schema validation failed: "
            + "; ".join(errors)
        )

    expected_hash = _self_hash(candidate, "policy_sha256")
    if candidate["policy_sha256"] != expected_hash:
        raise ValueError("event Findings atom-roster policy_sha256 mismatch")
    if trusted_policy_sha256 is not None and expected_hash != trusted_policy_sha256:
        raise ValueError("event Findings atom-roster policy is not host trusted")

    core_ids = [str(row["atom_id"]) for row in candidate["core_atom_specs"]]
    child_ids = [
        str(row["child_roster_id"])
        for row in candidate["child_roster_specs"]
    ]
    if len(core_ids) != len(set(core_ids)):
        raise ValueError("event Findings atom-roster policy has duplicate core atom IDs")
    if len(child_ids) != len(set(child_ids)):
        raise ValueError("event Findings atom-roster policy has duplicate child roster IDs")
    if set(core_ids) & set(child_ids):
        raise ValueError("core atom and child roster IDs must be disjoint")

    partition = candidate["sharp_context_partition_policy"]
    if set(partition["retired_ambiguous_term_ids"]) != set(
        _RETIRED_AMBIGUOUS_SHARP_TERMS
    ):
        raise ValueError("sharp context partition retired-term set differs")
    if set(partition["interictal_term_ids"]) != set(_INTERICTAL_SHARP_TERMS):
        raise ValueError("sharp context partition interictal-term set differs")
    if set(partition["ictal_term_ids"]) != set(_ICTAL_SHARP_COMPONENT_TERMS):
        raise ValueError("sharp context partition ictal-term set differs")
    if set(_INTERICTAL_SHARP_TERMS) & set(_ICTAL_SHARP_COMPONENT_TERMS):
        raise AssertionError("interictal and ictal sharp term sets must be disjoint")

    core_by_id = {
        str(row["atom_id"]): row for row in candidate["core_atom_specs"]
    }
    child_by_id = {
        str(row["child_roster_id"]): row
        for row in candidate["child_roster_specs"]
    }
    if _LEGACY_SHARED_ACQUISITION_ROSTER_ID in child_by_id:
        raise ValueError(
            "a shared generic acquisition-sensitive roster is forbidden"
        )
    missing_acquisition_gates = sorted(
        set(_ACQUISITION_SENSITIVE_ROSTER_SPECS) - set(child_by_id)
    )
    if missing_acquisition_gates:
        raise ValueError(
            "event Findings atom-roster policy lacks independent "
            f"acquisition-sensitive gates: {missing_acquisition_gates}"
        )
    for roster_id, expected in _ACQUISITION_SENSITIVE_ROSTER_SPECS.items():
        row = child_by_id[roster_id]
        exact_values = {
            "allowed_term_ids": [expected["term_id"]],
            "allowed_finding_families": [expected["family"]],
            "allowed_surface_frame_ids": [expected["surface_frame_id"]],
            "structural_scopes": [
                "event_mandatory",
                "instance_dependent",
                "acquisition_sensitive",
            ],
            "allowed_assertion_levels": [
                "model_candidate",
                "report_eligible_automated",
            ],
            "allowed_intrinsic_evidence_roles": [
                "early_context",
                "limitation",
            ],
            "source_paths": ["/acquisition_capabilities"],
            "activation_policy": "acquisition_sensitive",
            "absence_requires_complete_opportunity": True,
            "absence_requires_sensitivity_receipt": True,
            "onset_support_permission": "forbidden",
            "current_implementation_status": "unimplemented_not_evaluable",
            "instance_semantics": "acquisition_sensitive_pattern_candidate",
            "enumeration_scope": "event_analysis_window",
            "deduplication_required": True,
            "opportunity_denominator_source": (
                "independent_policy_enumerator_required_for_absence_or_recall"
            ),
        }
        if any(row[key] != value for key, value in exact_values.items()):
            raise ValueError(
                f"{roster_id}: independent acquisition-sensitive gate "
                "semantics differ"
            )
    if "c1_sharp_transient_instances" in child_by_id:
        raise ValueError("retired mixed sharp-transient child roster is forbidden")
    interictal_core_id = str(partition["interictal_core_atom_id"])
    interictal_child_id = str(partition["interictal_child_roster_id"])
    ictal_core_id = str(partition["ictal_core_atom_id"])
    ictal_child_id = str(partition["ictal_child_roster_id"])
    if interictal_core_id not in core_by_id or ictal_core_id not in core_by_id:
        raise ValueError("sharp context partition references an unknown core atom")
    if interictal_child_id not in child_by_id or ictal_child_id not in child_by_id:
        raise ValueError("sharp context partition references an unknown child roster")

    interictal_specs = (
        core_by_id[interictal_core_id],
        child_by_id[interictal_child_id],
    )
    for row in interictal_specs:
        if set(row["allowed_term_ids"]) != set(_INTERICTAL_SHARP_TERMS):
            raise ValueError(
                "interictal sharp roster must contain exactly the protected "
                "interictal terms"
            )
        if set(row["allowed_intrinsic_evidence_roles"]) != {"non_event_context"}:
            raise ValueError(
                "interictal sharp roster must be restricted to non_event_context"
            )

    ictal_child = child_by_id[ictal_child_id]
    if set(ictal_child["allowed_term_ids"]) != set(_ICTAL_SHARP_COMPONENT_TERMS):
        raise ValueError(
            "ictal sharp child roster must contain exactly the ictal candidate term"
        )
    if set(ictal_child["allowed_intrinsic_evidence_roles"]) != {"early_context"}:
        raise ValueError(
            "ictal sharp child roster must be restricted to early_context"
        )
    ictal_core = core_by_id[ictal_core_id]
    if not set(_ICTAL_SHARP_COMPONENT_TERMS).issubset(
        set(ictal_core["allowed_term_ids"])
    ) or "early_context" not in ictal_core["allowed_intrinsic_evidence_roles"]:
        raise ValueError("ictal sharp core atom does not bind the ictal candidate")

    designated_interictal_ids = {interictal_core_id, interictal_child_id}
    designated_ictal_ids = {ictal_core_id, ictal_child_id}
    for collection_name, id_key in (
        ("core_atom_specs", "atom_id"),
        ("child_roster_specs", "child_roster_id"),
    ):
        for row in candidate[collection_name]:
            context = str(row[id_key])
            terms = set(row["allowed_term_ids"])
            if terms & set(_RETIRED_AMBIGUOUS_SHARP_TERMS):
                raise ValueError(f"{context}: retired ambiguous sharp term is forbidden")
            if terms & set(_INTERICTAL_SHARP_TERMS):
                if context not in designated_interictal_ids:
                    raise ValueError(
                        f"{context}: interictal protected term escaped its partition"
                    )
                if set(row["allowed_intrinsic_evidence_roles"]) != {
                    "non_event_context"
                }:
                    raise ValueError(
                        f"{context}: interictal protected term has a non-interictal role"
                    )
            if terms & set(_ICTAL_SHARP_COMPONENT_TERMS):
                if context not in designated_ictal_ids:
                    raise ValueError(
                        f"{context}: ictal sharp candidate escaped its partition"
                    )
                if "early_context" not in row["allowed_intrinsic_evidence_roles"]:
                    raise ValueError(
                        f"{context}: ictal sharp candidate lacks early_context"
                    )
    for collection_name, id_key in (
        ("core_atom_specs", "atom_id"),
        ("child_roster_specs", "child_roster_id"),
    ):
        for row in candidate[collection_name]:
            context = str(row[id_key])
            if (
                row["absence_requires_sensitivity_receipt"]
                and not row["absence_requires_complete_opportunity"]
            ):
                raise ValueError(
                    f"{context}: sensitivity-qualified absence also requires "
                    "a complete evaluation opportunity"
                )
            if row["onset_support_permission"] == (
                "required_future_free_causal_if_positive"
            ) and "onset_eligible" not in row["allowed_intrinsic_evidence_roles"]:
                raise ValueError(
                    f"{context}: causal onset permission requires onset_eligible role"
                )
            if row["onset_support_permission"] in {
                "forbidden",
                "not_applicable",
            } and "onset_eligible" in row["allowed_intrinsic_evidence_roles"]:
                raise ValueError(
                    f"{context}: onset-ineligible slot cannot allow onset_eligible role"
                )
            if row["current_implementation_status"] == (
                "unimplemented_not_evaluable"
            ) and "report_eligible_automated" not in row[
                "allowed_assertion_levels"
            ]:
                # An unimplemented row may still be measurement-only.  No extra
                # restriction is required; this branch documents the deliberate
                # absence of an implicit promotion rule.
                pass
    return candidate


def _validate_sharp_context_partition_findings(
    source: Mapping[str, object],
    policy: Mapping[str, object],
) -> None:
    """Reject legacy or cross-context sharp terms before roster matching."""

    partition = policy["sharp_context_partition_policy"]
    retired = set(str(value) for value in partition["retired_ambiguous_term_ids"])
    interictal = set(str(value) for value in partition["interictal_term_ids"])
    ictal = set(str(value) for value in partition["ictal_term_ids"])
    for index, raw in enumerate(source["findings"]):  # type: ignore[index]
        finding = dict(raw)
        term_id = str(finding["term"]["term_id"])
        role = str(finding["intrinsic_evidence_role"])
        if term_id in retired:
            raise ValueError(
                f"findings[{index}] uses retired ambiguous sharp term {term_id!r}"
            )
        if term_id in interictal and role != "non_event_context":
            raise ValueError(
                f"findings[{index}] interictal term {term_id!r} requires "
                "non_event_context"
            )
        if term_id in ictal and role != "early_context":
            raise ValueError(
                f"findings[{index}] ictal sharp candidate requires early_context"
            )


def load_event_findings_atom_roster_policy(
    path: str | Path = DEFAULT_EVENT_FINDINGS_ATOM_ROSTER_POLICY_PATH,
    *,
    trusted_policy_sha256: str | None = None,
) -> dict[str, Any]:
    return validate_event_findings_atom_roster_policy(
        _read_json(Path(path)),
        trusted_policy_sha256=trusted_policy_sha256,
    )


def _policy(
    value: Mapping[str, object] | None,
    trusted_policy_sha256: str | None,
) -> dict[str, Any]:
    if value is None:
        return load_event_findings_atom_roster_policy(
            trusted_policy_sha256=trusted_policy_sha256
        )
    if trusted_policy_sha256 is None:
        # A payload must not be able to author and self-hash its own smaller or
        # more permissive denominator.  The checked-in policy is the default
        # host trust anchor; experiments with another policy must pass its hash
        # explicitly from outside the payload.
        trusted_policy_sha256 = str(
            load_event_findings_atom_roster_policy()["policy_sha256"]
        )
    return validate_event_findings_atom_roster_policy(
        dict(value),
        trusted_policy_sha256=trusted_policy_sha256,
    )


def _resolve_path(root: Mapping[str, object], path: str) -> tuple[bool, object]:
    current: object = root
    for part in path.lstrip("/").split("/"):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _matching_findings(
    source: Mapping[str, object],
    spec: Mapping[str, object],
) -> list[dict[str, Any]]:
    families = set(str(value) for value in spec["allowed_finding_families"])
    term_ids = set(str(value) for value in spec["allowed_term_ids"])
    roles = set(str(value) for value in spec["allowed_intrinsic_evidence_roles"])
    assertion_levels = set(
        str(value) for value in spec["allowed_assertion_levels"]
    )
    # ``allowed_term_ids`` is a closed allowlist.  An empty list belongs to a
    # structural adapter (for example occurrence/burden or composite-pattern
    # containers); it must never act as a wildcard over every Finding family.
    if not term_ids:
        return []
    result: list[dict[str, Any]] = []
    for raw in source["findings"]:  # type: ignore[index]
        row = dict(raw)
        family_ok = not families or str(row["family"]) in families
        term_ok = str(row["term"]["term_id"]) in term_ids
        role_ok = not roles or str(row["intrinsic_evidence_role"]) in roles
        assertion_ok = str(row["assertion_level"]) in assertion_levels
        if family_ok and term_ok and role_ok and assertion_ok:
            result.append(row)
    return sorted(result, key=lambda row: str(row["evidence_id"]))


def _matching_opportunities(
    source: Mapping[str, object],
    spec: Mapping[str, object],
) -> list[dict[str, Any]]:
    families = set(str(value) for value in spec["allowed_finding_families"])
    term_ids = set(str(value) for value in spec["allowed_term_ids"])
    # As above, an empty term allowlist is a deny-by-default structural slot,
    # not permission to consume every opportunity in the listed families.
    if not term_ids:
        return []
    result: list[dict[str, Any]] = []
    for raw in source["evaluation_opportunities"]:  # type: ignore[index]
        row = dict(raw)
        family_ok = not families or str(row["family"]) in families
        term_ok = str(row["term_id"]) in term_ids
        if family_ok and term_ok:
            result.append(row)
    return sorted(
        result,
        key=lambda row: str(row["evaluation_opportunity_id"]),
    )


def _finding_receipt_ids(
    findings: Sequence[Mapping[str, object]],
    key: str,
) -> list[str]:
    values = {
        str(row[key])
        for row in findings
        if row.get(key) is not None
    }
    return sorted(values)


def _aggregate_status(findings: Sequence[Mapping[str, object]]) -> str:
    if not findings:
        return "not_evaluable"
    return max(
        (str(row["status"]) for row in findings),
        key=lambda value: _STATUS_PRIORITY[value],
    )


def _aggregate_assertion(
    findings: Sequence[Mapping[str, object]],
) -> str | None:
    if not findings:
        return None
    return max(
        (str(row["assertion_level"]) for row in findings),
        key=lambda value: _ASSERTION_PRIORITY[value],
    )


def _spatial_unit_ids(finding: Mapping[str, object]) -> set[str]:
    result: set[str] = set()
    for raw in finding.get("spatial_support", []):
        row = dict(raw)
        if row.get("unit_type") in {"lead", "electrode"}:
            result.add(str(row["id"]))
    return result


def _opportunity_unit_ids(opportunity: Mapping[str, object]) -> set[str]:
    result: set[str] = set()
    for value in opportunity.get("spatial_unit_keys", []):
        text = str(value)
        result.add(text.split(":", 1)[-1])
    return result


def _source_path_unit_statuses(
    source: Mapping[str, object],
) -> dict[str, dict[str, tuple[str, list[str], list[str]]]]:
    """Return path family -> unit -> (status, evidence IDs, opportunity IDs)."""

    result: dict[str, dict[str, tuple[str, list[str], list[str]]]] = {
        "montage": {},
        "quality": {},
        "involvement": {},
    }
    montage = dict(source["montage"])
    for raw in montage["input_units"]:
        row = dict(raw)
        unit_id = str(row["unit_id"])
        status = (
            "present"
            if row["observation_status"] == "observed" and row["evidence_eligible"]
            else "not_evaluable"
        )
        result["montage"][unit_id] = (status, [], [])

    quality = dict(source["quality"])
    for raw in quality["per_unit"]:
        row = dict(raw)
        unit_id = str(row["unit_id"])
        status = "present" if row["status"] == "usable" else "not_evaluable"
        result["quality"][unit_id] = (status, [], [])

    hypothesis = dict(source["scalp_onset_hypothesis"])
    for raw in hypothesis["per_unit_involvement"]:
        row = dict(raw)
        unit_id = str(row["unit_id"])
        opportunity_ids = (
            []
            if row["evaluation_opportunity_id"] is None
            else [str(row["evaluation_opportunity_id"])]
        )
        result["involvement"][unit_id] = (
            str(row["status"]),
            sorted(str(value) for value in row["evidence_ids"]),
            opportunity_ids,
        )
    return result


def _unit_dispositions(
    *,
    source: Mapping[str, object],
    spec: Mapping[str, object],
    expected_unit_ids: Sequence[str],
    findings: Sequence[Mapping[str, object]],
    opportunities: Sequence[Mapping[str, object]],
    technical_failure: bool,
) -> list[dict[str, Any]]:
    if "unit_mandatory" not in spec["structural_scopes"]:
        return []

    path_units = _source_path_unit_statuses(source)
    source_paths = set(str(path) for path in spec["source_paths"])
    implementation = str(spec["current_implementation_status"])
    result: list[dict[str, Any]] = []
    for unit_id in expected_unit_ids:
        matched_findings = [
            row for row in findings if unit_id in _spatial_unit_ids(row)
        ]
        matched_opportunities = [
            row for row in opportunities if unit_id in _opportunity_unit_ids(row)
        ]
        reason_codes: list[str] = []
        if technical_failure:
            status = "not_evaluable"
            processing = "technical_failure"
            reason_codes.append("producer_technical_failure")
        elif implementation == "unimplemented_not_evaluable":
            status = "not_evaluable"
            processing = "completed"
            reason_codes.append("registered_module_not_implemented")
        elif matched_findings:
            status = _aggregate_status(matched_findings)
            processing = "completed"
        else:
            processing = "completed"
            structural_status: str | None = None
            for path, family in (
                ("/montage", "montage"),
                ("/quality/per_unit", "quality"),
                (
                    "/scalp_onset_hypothesis/per_unit_involvement",
                    "involvement",
                ),
            ):
                if path in source_paths and unit_id in path_units[family]:
                    structural_status = path_units[family][unit_id][0]
                    break
            if structural_status is not None:
                status = structural_status
            elif matched_opportunities:
                status = "uncertain"
                reason_codes.append("unit_has_opportunity_but_no_explicit_finding")
            else:
                status = "not_evaluable"
                reason_codes.append("unit_not_materialized_by_source")

        if status == "absent_with_opportunity":
            sufficient = any(row["status"] == "sufficient" for row in matched_opportunities)
            has_sensitivity = bool(
                _finding_receipt_ids(matched_findings, "sensitivity_receipt_id")
            )
            if not sufficient or (
                spec["absence_requires_sensitivity_receipt"] and not has_sensitivity
            ):
                status = "uncertain"
                reason_codes.append("unit_absence_not_fully_qualified")

        result.append(
            {
                "unit_id": unit_id,
                "status": status,
                "processing_disposition": processing,
                "finding_ids": sorted(
                    str(row["evidence_id"]) for row in matched_findings
                ),
                "evaluation_opportunity_ids": sorted(
                    str(row["evaluation_opportunity_id"])
                    for row in matched_opportunities
                ),
                "reason_codes": sorted(set(reason_codes)),
                "accounted_for": True,
            }
        )
    return result


def _dependency_permissions(
    source: Mapping[str, object],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for finding in source["findings"]:  # type: ignore[index]
        for measurement in finding["measurements"]:
            dependency = measurement["source_binding"]["raw_sample_dependency"]
            if dependency is not None:
                result[str(dependency["dependency_id"])] = dependency
    for waveform in source["waveform_evidence"]:  # type: ignore[index]
        dependency = waveform["raw_sample_dependency"]
        if dependency is not None:
            result[str(dependency["dependency_id"])] = dependency
    return result


def _require_onset_causal_permissions(
    source: Mapping[str, object],
    findings: Sequence[Mapping[str, object]],
    context: str,
) -> None:
    dependency_map = _dependency_permissions(source)
    for finding in findings:
        if finding["status"] != "present":
            continue
        if finding["intrinsic_evidence_role"] != "onset_eligible":
            continue
        dependency_ids = [
            str(value) for value in finding["raw_sample_dependency_ids"]
        ]
        if not dependency_ids:
            raise ValueError(f"{context}: positive onset evidence has no raw dependency")
        for dependency_id in dependency_ids:
            if dependency_id not in dependency_map:
                raise ValueError(
                    f"{context}: missing onset raw dependency {dependency_id}"
                )
            dependency = dependency_map[dependency_id]
            allowed = (
                dependency["view_role"] == "onset_causal"
                and dependency["future_sample_access"] is False
                and dependency["onset_evidence_authorized"] is True
                and dependency["onset_support_eligible"] is True
            )
            if not allowed:
                raise ValueError(
                    f"{context}: positive onset evidence is not future-free and causal"
                )


def _event_onset_claim_authorization(
    source: Mapping[str, object],
    findings: Sequence[Mapping[str, object]],
) -> tuple[bool, str | None]:
    """Authorize onset semantics only for a supported cerebral ictal event.

    A detector/onset-like signal change remains auditable when this gate is
    closed, but it cannot be surfaced as an onset-topology hypothesis.
    """

    onset_ids = {
        str(row["evidence_id"])
        for row in findings
        if row["status"] == "present"
        and row["intrinsic_evidence_role"] == "onset_eligible"
    }
    if not onset_ids:
        return False, "positive_onset_evidence_not_materialized"

    qualification = dict(source["event_qualification"])
    if qualification["status"] not in {
        "qualified_electrographic_event",
        "qualified_electrographic_seizure",
    }:
        return False, "event_not_qualified_for_onset_claim"

    outcome = dict(source["event_outcome"])
    if outcome["outcome"] not in {
        "qualified_electrographic_event",
        "qualified_electrographic_seizure",
    }:
        return False, "event_outcome_not_qualified_for_onset_claim"

    hypotheses = dict(source["competing_hypotheses"])
    selected_id = hypotheses["selected_hypothesis_id"]
    if hypotheses["status"] != "available" or selected_id is None:
        return False, "supported_cerebral_ictal_hypothesis_not_selected"
    selected = next(
        (
            dict(row)
            for row in hypotheses["hypotheses"]
            if row["hypothesis_id"] == selected_id
        ),
        None,
    )
    if selected is None or not (
        selected["category"] == "cerebral_ictal"
        and selected["disposition"] == "supported"
        and selected["onset_claim_eligible"] is True
    ):
        return False, "selected_hypothesis_not_onset_claim_eligible"
    supporting_ids = {
        str(value) for value in selected["supporting_evidence_ids"]
    }
    if not onset_ids.issubset(supporting_ids):
        return False, "onset_evidence_not_bound_to_selected_hypothesis"
    return True, None


def _qualified_absence(
    status: str,
    *,
    spec: Mapping[str, object],
    opportunities: Sequence[Mapping[str, object]],
    sensitivity_ids: Sequence[str],
) -> tuple[str, list[str]]:
    if status != "absent_with_opportunity":
        return status, []
    reasons: list[str] = []
    if spec["absence_requires_complete_opportunity"] and not any(
        row["status"] == "sufficient" for row in opportunities
    ):
        reasons.append("absence_missing_complete_opportunity")
    if spec["absence_requires_sensitivity_receipt"] and not sensitivity_ids:
        reasons.append("absence_missing_sensitivity_receipt")
    if reasons:
        return "uncertain", reasons
    return status, reasons


def _claim_disposition(
    *,
    policy: Mapping[str, object],
    spec: Mapping[str, object],
    status: str,
    assertion_level: str | None,
    onset_claim_authorized: bool = True,
) -> str:
    if status == "not_evaluable":
        return (
            "limitation_only"
            if spec["salience_tier"] in {"critical", "major"}
            else "withheld"
        )
    if status != "present":
        return "withheld"
    # Structural source availability is deliberately weaker than an
    # assertion.  In particular, a populated v3 block may itself say that an
    # onset hypothesis is not evaluable.  Never turn a source path (or any
    # other assertion-less row) into a clinical/research claim merely because
    # the enclosing object exists.
    if assertion_level is None:
        return "withheld"
    if (
        spec["term_semantic_class"] == "research_onset_hypothesis"
        and not onset_claim_authorized
    ):
        return "withheld"
    if (
        policy["policy_status"] == "clinically_qualified"
        and assertion_level == "report_eligible_automated"
    ):
        if spec["term_semantic_class"] == "research_onset_hypothesis":
            return "research_hypothesis_candidate"
        return "report_eligible_qualified_claim"
    if assertion_level == "measured":
        return "descriptive_measurement_candidate"
    if spec["term_semantic_class"] == "research_onset_hypothesis":
        return "research_hypothesis_candidate"
    return "withheld"


def _core_entry(
    *,
    source: Mapping[str, object],
    policy: Mapping[str, object],
    spec: Mapping[str, object],
    expected_unit_ids: Sequence[str],
    technical_failure: bool,
) -> dict[str, Any]:
    findings = _matching_findings(source, spec)
    opportunities = _matching_opportunities(source, spec)
    source_paths = [
        str(path)
        for path in spec["source_paths"]
        if _resolve_path(source, str(path))[0]
    ]
    reason_codes: list[str] = []
    atom_id = str(spec["atom_id"])
    if technical_failure:
        status = "not_evaluable"
        processing = "technical_failure"
        assertion_level = None
        findings = []
        opportunities = []
        reason_codes.append("producer_technical_failure")
    elif atom_id == "q_background_comparability":
        # A pre-event spectral profile or a context interval does not by
        # itself prove comparability.  The dedicated source-bound baseline
        # sidecar is intentionally not embedded in frozen v3.
        status = "not_evaluable"
        processing = "completed"
        assertion_level = None
        reason_codes.append("independent_background_comparability_receipt_required")
    elif atom_id == "c2_cross_reference_stability_resolution" and len(
        source["montage"]["reference_perturbations_evaluated"]  # type: ignore[index]
    ) < 2:
        status = "not_evaluable"
        processing = "completed"
        assertion_level = None
        reason_codes.append("fewer_than_two_reference_views_evaluated")
    elif spec["current_implementation_status"] == "unimplemented_not_evaluable":
        status = "not_evaluable"
        processing = "completed"
        assertion_level = None
        reason_codes.append("registered_module_not_implemented")
    elif findings:
        status = _aggregate_status(findings)
        processing = "completed"
        assertion_level = _aggregate_assertion(findings)
    elif source_paths:
        # Source-path presence closes structural accounting only.  It does not
        # prove that the represented clinical atom is present: the referenced
        # block can explicitly be uncertain/not-evaluable, and some blocks are
        # containers rather than assertions.  A dedicated, source-bound
        # Finding (or a future qualified structural adapter) is required to
        # assign a four-state clinical status.
        status = "not_evaluable"
        processing = "completed"
        assertion_level = None
        reason_codes.append(
            "structural_source_accounted_without_qualified_finding"
        )
    else:
        status = "not_evaluable"
        processing = "completed"
        assertion_level = None
        reason_codes.append("required_atom_source_not_materialized")

    capability_ids = _finding_receipt_ids(findings, "capability_receipt_id")
    sensitivity_ids = _finding_receipt_ids(findings, "sensitivity_receipt_id")
    term_decision_ids = _finding_receipt_ids(findings, "term_decision_receipt_id")
    status, absence_reasons = _qualified_absence(
        status,
        spec=spec,
        opportunities=opportunities,
        sensitivity_ids=sensitivity_ids,
    )
    reason_codes.extend(absence_reasons)

    if (
        spec["onset_support_permission"]
        == "required_future_free_causal_if_positive"
    ):
        _require_onset_causal_permissions(source, findings, str(spec["atom_id"]))
        onset_claim_authorized, onset_reason = _event_onset_claim_authorization(
            source, findings
        )
        if onset_reason is not None:
            reason_codes.append(onset_reason)
    else:
        onset_claim_authorized = (
            spec["term_semantic_class"] != "research_onset_hypothesis"
        )

    unit_dispositions = _unit_dispositions(
        source=source,
        spec=spec,
        expected_unit_ids=expected_unit_ids,
        findings=findings,
        opportunities=opportunities,
        technical_failure=technical_failure,
    )
    scoped_units = (
        list(expected_unit_ids)
        if "unit_mandatory" in spec["structural_scopes"]
        else []
    )
    disposition = _claim_disposition(
        policy=policy,
        spec=spec,
        status=status,
        assertion_level=assertion_level,
        onset_claim_authorized=onset_claim_authorized,
    )
    surface_frames = (
        sorted(str(value) for value in spec["allowed_surface_frame_ids"])
        if disposition == "report_eligible_qualified_claim"
        else []
    )
    entry: dict[str, Any] = {
        "atom_id": str(spec["atom_id"]),
        "group": str(spec["group"]),
        "status": status,
        "processing_disposition": processing,
        "assertion_level": assertion_level,
        "finding_ids": sorted(str(row["evidence_id"]) for row in findings),
        "evaluation_opportunity_ids": sorted(
            str(row["evaluation_opportunity_id"]) for row in opportunities
        ),
        "capability_receipt_ids": capability_ids,
        "sensitivity_receipt_ids": sensitivity_ids,
        "term_decision_receipt_ids": term_decision_ids,
        "source_paths": sorted(source_paths),
        "expected_unit_ids": scoped_units,
        "accounted_unit_ids": scoped_units,
        "unit_dispositions": unit_dispositions,
        "surface_frame_ids": surface_frames,
        "claim_plan_disposition": disposition,
        "reason_codes": sorted(set(reason_codes)),
        "accounted_for": True,
        "expert_correctness_claimed": False,
        "entry_binding_sha256": "0" * 64,
    }
    entry["entry_binding_sha256"] = _self_hash(entry, "entry_binding_sha256")
    return entry


def _interval_from_finding(finding: Mapping[str, object]) -> dict[str, float] | None:
    raw = finding.get("time_interval")
    if raw is None:
        return None
    row = dict(raw)
    return {
        "start": float(row["start"]),
        "stop": float(row["stop"]),
        "resolution_seconds": float(row["resolution_seconds"]),
    }


def _interval_from_boundary(value: Mapping[str, object]) -> dict[str, float] | None:
    if value.get("interval") is None:
        return None
    interval = dict(value["interval"])
    return {
        "start": float(interval["lower"]),
        "stop": float(interval["upper"]),
        "resolution_seconds": float(interval["resolution_seconds"]),
    }


def _enumeration_scope_interval(
    source: Mapping[str, object],
    spec: Mapping[str, object],
    opportunities: Sequence[Mapping[str, object]],
) -> dict[str, float] | None:
    intervals = [
        dict(row["interval"])
        for row in opportunities
        if row.get("interval") is not None
    ]
    if intervals:
        return {
            "start": min(float(row["start"]) for row in intervals),
            "stop": max(float(row["stop"]) for row in intervals),
            "resolution_seconds": max(
                float(row["resolution_seconds"]) for row in intervals
            ),
        }
    scope = str(spec["enumeration_scope"])
    window = dict(source["window"])
    if scope in {"event_analysis_window", "event_course_interval"}:
        start, stop = window["final_interval"]
        return {"start": float(start), "stop": float(stop)}
    if scope == "candidate_emergence_interval":
        return _interval_from_boundary(dict(window["onset_boundary"]))
    if scope == "post_event_context":
        return _interval_from_boundary(dict(window["offset_boundary"]))
    if scope == "quality_evaluable_interval":
        start, stop = window["final_interval"]
        return {"start": float(start), "stop": float(stop)}
    return None


def _instance(
    *,
    instance_id: str,
    status: str,
    finding_ids: Sequence[str],
    time_interval: Mapping[str, object] | None,
    unit_ids: Sequence[str],
    source_paths: Sequence[str],
    waveform_evidence_ids: Sequence[str],
    reason_codes: Sequence[str],
    pattern_candidate_disposition: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "instance_id": instance_id,
        "status": status,
        "finding_ids": sorted(set(finding_ids)),
        "time_interval": None if time_interval is None else dict(time_interval),
        "unit_ids": sorted(set(unit_ids)),
        "source_paths": sorted(set(source_paths)),
        "waveform_evidence_ids": sorted(set(waveform_evidence_ids)),
        "reason_codes": sorted(set(reason_codes)),
        "pattern_candidate_disposition": (
            None
            if pattern_candidate_disposition is None
            else deepcopy(dict(pattern_candidate_disposition))
        ),
        "instance_binding_sha256": "0" * 64,
    }
    result["instance_binding_sha256"] = _self_hash(
        result, "instance_binding_sha256"
    )
    return result


def _generic_finding_instances(
    findings: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for finding in findings:
        if finding["status"] not in {"present", "uncertain"}:
            continue
        interval = _interval_from_finding(finding)
        units = sorted(_spatial_unit_ids(finding))
        semantic_key = _canonical_sha256(
            {
                "term_id": finding["term"]["term_id"],
                "time_interval": interval,
                "unit_ids": units,
                "status": finding["status"],
            }
        )
        if semantic_key not in grouped:
            grouped[semantic_key] = {
                "status": str(finding["status"]),
                "finding_ids": [],
                "time_interval": interval,
                "unit_ids": units,
                "waveform_evidence_ids": [],
            }
        grouped[semantic_key]["finding_ids"].append(str(finding["evidence_id"]))
        grouped[semantic_key]["waveform_evidence_ids"].extend(
            str(value) for value in finding["waveform_evidence_ids"]
        )
    result: list[dict[str, Any]] = []
    for semantic_key, row in sorted(grouped.items()):
        result.append(
            _instance(
                instance_id=f"ATOMINST-{semantic_key[:24]}",
                status=row["status"],
                finding_ids=row["finding_ids"],
                time_interval=row["time_interval"],
                unit_ids=row["unit_ids"],
                source_paths=[],
                waveform_evidence_ids=row["waveform_evidence_ids"],
                reason_codes=[],
            )
        )
    return result


def _spatial_involvement_instances(
    source: Mapping[str, object],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    hypothesis = dict(source["scalp_onset_hypothesis"])
    for raw in hypothesis["per_unit_involvement"]:
        row = dict(raw)
        if row["status"] not in {"present", "uncertain"}:
            continue
        interval = _interval_from_boundary(row)
        semantic_key = _canonical_sha256(
            {
                "unit_id": row["unit_id"],
                "status": row["status"],
                "interval": interval,
                "evidence_ids": row["evidence_ids"],
            }
        )
        result.append(
            _instance(
                instance_id=f"ATOMINST-{semantic_key[:24]}",
                status=str(row["status"]),
                finding_ids=[str(value) for value in row["evidence_ids"]],
                time_interval=interval,
                unit_ids=[str(row["unit_id"])],
                source_paths=[
                    "/scalp_onset_hypothesis/per_unit_involvement"
                ],
                waveform_evidence_ids=[],
                reason_codes=[],
            )
        )
    return sorted(result, key=lambda row: str(row["instance_id"]))


def _pattern_instances(source: Mapping[str, object]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    finding_map = {
        str(row["evidence_id"]): row for row in source["findings"]  # type: ignore[index]
    }
    for raw in source["pattern_candidates"]:  # type: ignore[index]
        row = dict(raw)
        # An absent/not-evaluable semantic assertion is not an activated
        # pattern occurrence.  It remains fully preserved in the source v3
        # graph and source hash, but it cannot be upgraded to an uncertain
        # positive child instance.
        if row["status"] not in {"present", "uncertain"}:
            continue
        physical_instance_id = str(row["pattern_instance_id"])
        candidate_id = str(row["pattern_candidate_id"])
        finding_ids = [str(value) for value in row["required_atom_ids"]]
        members = [finding_map[value] for value in finding_ids if value in finding_map]
        intervals = [
            _interval_from_finding(member)
            for member in members
            if _interval_from_finding(member) is not None
        ]
        interval = None
        if intervals:
            interval = {
                "start": min(float(value["start"]) for value in intervals),
                "stop": max(float(value["stop"]) for value in intervals),
                "resolution_seconds": max(
                    float(value.get("resolution_seconds", 0.0))
                    for value in intervals
                ),
            }
        semantic_instance_key = {
            "physical_pattern_instance_id": physical_instance_id,
            "pattern_candidate_id": candidate_id,
        }
        result.append(
            _instance(
                instance_id=(
                    "ATOMINST-"
                    + _canonical_sha256(semantic_instance_key)[:24]
                ),
                status=str(row["status"]),
                finding_ids=finding_ids,
                time_interval=interval,
                unit_ids=sorted(
                    set().union(
                        *(_spatial_unit_ids(member) for member in members)
                    )
                    if members
                    else set()
                ),
                source_paths=["/pattern_candidates"],
                waveform_evidence_ids=sorted(
                    {
                        str(value)
                        for member in members
                        for value in member["waveform_evidence_ids"]
                    }
                ),
                reason_codes=[str(value) for value in row["reason_codes"]],
                pattern_candidate_disposition={
                    "physical_pattern_instance_id": physical_instance_id,
                    "pattern_candidate_id": candidate_id,
                    "term_id": str(row["term"]["term_id"]),
                    "source_domain_scope": str(row["source_domain_scope"]),
                    "assertion_level": str(row["assertion_level"]),
                    "status": str(row["status"]),
                    "counterevidence_ids": sorted(
                        str(value) for value in row["counterevidence_ids"]
                    ),
                    "qualification_rule_receipt_id": (
                        None
                        if row["qualification_rule_receipt_id"] is None
                        else str(row["qualification_rule_receipt_id"])
                    ),
                },
            )
        )
    return sorted(result, key=lambda row: str(row["instance_id"]))


def _artifact_instances(source: Mapping[str, object]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in source["quality"]["artifact_intervals"]:  # type: ignore[index]
        row = dict(raw)
        interval_raw = row.get("interval")
        interval = None
        if (
            isinstance(interval_raw, Sequence)
            and not isinstance(interval_raw, (str, bytes))
            and len(interval_raw) == 2
        ):
            interval = {
                "start": float(interval_raw[0]),
                "stop": float(interval_raw[1]),
            }
        elif isinstance(interval_raw, Mapping) and {
            "start",
            "stop",
        }.issubset(interval_raw):
            interval = {
                "start": float(interval_raw["start"]),
                "stop": float(interval_raw["stop"]),
            }
            if "resolution_seconds" in interval_raw:
                interval["resolution_seconds"] = float(
                    interval_raw["resolution_seconds"]
                )
        units = sorted(
            str(value)
            for value in row.get(
                "affected_unit_ids",
                row.get("unit_ids", []),
            )
        )
        identity_row = deepcopy(row)
        if "affected_unit_ids" in identity_row:
            identity_row["affected_unit_ids"] = units
        elif "unit_ids" in identity_row:
            identity_row["unit_ids"] = units
        semantic_key = _canonical_sha256(identity_row)
        result.append(
            _instance(
                instance_id=f"ARTIFACT-{semantic_key[:24]}",
                status="present",
                finding_ids=[],
                time_interval=interval,
                unit_ids=units,
                source_paths=["/quality/artifact_intervals"],
                waveform_evidence_ids=[],
                reason_codes=["signal_artifact_candidate_not_behavior_inference"],
            )
        )
    return sorted(result, key=lambda row: str(row["instance_id"]))


def _child_instances(
    source: Mapping[str, object],
    spec: Mapping[str, object],
    findings: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    semantics = str(spec["instance_semantics"])
    if semantics == "spatial_involvement_candidate":
        return _spatial_involvement_instances(source)
    if semantics == "composite_pattern_candidate":
        return _pattern_instances(source)
    if semantics == "artifact_interval_candidate":
        return _artifact_instances(source)
    return _generic_finding_instances(findings)


def _child_roster(
    *,
    source: Mapping[str, object],
    policy: Mapping[str, object],
    spec: Mapping[str, object],
    expected_unit_ids: Sequence[str],
    technical_failure: bool,
) -> dict[str, Any]:
    findings = _matching_findings(source, spec)
    opportunities = _matching_opportunities(source, spec)
    source_paths = [
        str(path)
        for path in spec["source_paths"]
        if _resolve_path(source, str(path))[0]
    ]
    capability_ids = _finding_receipt_ids(findings, "capability_receipt_id")
    sensitivity_ids = _finding_receipt_ids(findings, "sensitivity_receipt_id")
    term_decision_ids = _finding_receipt_ids(findings, "term_decision_receipt_id")
    reason_codes: list[str] = []

    if technical_failure:
        activation = "not_evaluable"
        finding_status = "not_evaluable"
        processing = "technical_failure"
        assertion_level = None
        instances: list[dict[str, Any]] = []
        completeness = "not_evaluable"
        reason_codes.append("producer_technical_failure")
    elif spec["current_implementation_status"] == "unimplemented_not_evaluable":
        activation = "not_evaluable"
        finding_status = "not_evaluable"
        processing = "completed"
        assertion_level = None
        instances = []
        completeness = "not_evaluable"
        reason_codes.append("registered_instance_enumerator_not_implemented")
    else:
        processing = "completed"
        assertion_level = _aggregate_assertion(findings)
        instances = _child_instances(source, spec, findings)
        instance_ids = [str(row["instance_id"]) for row in instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError(
                f"{spec['child_roster_id']}: child-instance identity is not unique"
            )
        if instances:
            activation = "triggered"
            finding_status = max(
                (str(row["status"]) for row in instances),
                key=lambda value: _STATUS_PRIORITY[value],
            )
            completeness = (
                "closed_present_roster"
                if finding_status == "present"
                else "candidate_roster_uncertain"
            )
        elif opportunities:
            activation = "not_triggered"
            if (
                any(row["status"] == "sufficient" for row in opportunities)
                and (
                    not spec["absence_requires_sensitivity_receipt"]
                    or sensitivity_ids
                )
            ):
                finding_status = "absent_with_opportunity"
                completeness = "closed_absent_with_opportunity"
            else:
                finding_status = "uncertain"
                completeness = "candidate_roster_uncertain"
                reason_codes.append("no_instance_without_qualified_absence")
        else:
            activation = "not_evaluable"
            finding_status = "not_evaluable"
            completeness = "not_evaluable"
            reason_codes.append("independent_instance_opportunity_not_materialized")

    if (
        spec["onset_support_permission"]
        == "required_future_free_causal_if_positive"
    ):
        _require_onset_causal_permissions(source, findings, str(spec["child_roster_id"]))

    unit_dispositions = _unit_dispositions(
        source=source,
        spec=spec,
        expected_unit_ids=expected_unit_ids,
        findings=findings,
        opportunities=opportunities,
        technical_failure=technical_failure,
    )
    scoped_units = (
        list(expected_unit_ids)
        if "unit_mandatory" in spec["structural_scopes"]
        else []
    )
    if instances:
        deduplication_policy_id: str | None = (
            "CLOSED-ATOM-INSTANCE-SEMANTIC-IDENTITY-V1"
        )
        deduplication_sha256: str | None = _canonical_sha256(
            [
                {
                    "instance_id": row["instance_id"],
                    "binding": row["instance_binding_sha256"],
                }
                for row in instances
            ]
        )
    else:
        deduplication_policy_id = None
        deduplication_sha256 = None

    roster: dict[str, Any] = {
        "child_roster_id": str(spec["child_roster_id"]),
        "group": str(spec["group"]),
        "activation_status": activation,
        "finding_status": finding_status,
        "processing_disposition": processing,
        "assertion_level": assertion_level,
        "finding_ids": sorted(str(row["evidence_id"]) for row in findings),
        "evaluation_opportunity_ids": sorted(
            str(row["evaluation_opportunity_id"]) for row in opportunities
        ),
        "capability_receipt_ids": capability_ids,
        "sensitivity_receipt_ids": sensitivity_ids,
        "term_decision_receipt_ids": term_decision_ids,
        "source_paths": sorted(source_paths),
        "expected_unit_ids": scoped_units,
        "accounted_unit_ids": scoped_units,
        "unit_dispositions": unit_dispositions,
        "enumeration_scope_interval": _enumeration_scope_interval(
            source, spec, opportunities
        ),
        "instances": instances,
        "deduplication_policy_id": deduplication_policy_id,
        "deduplication_sha256": deduplication_sha256,
        "completeness_disposition": completeness,
        "reason_codes": sorted(set(reason_codes)),
        "accounted_for": True,
        "expert_correctness_claimed": False,
        "roster_binding_sha256": "0" * 64,
    }
    roster["roster_binding_sha256"] = _self_hash(
        roster, "roster_binding_sha256"
    )
    return roster


def _validate_unit_inventory(
    source: Mapping[str, object],
    trusted_expected_unit_ids: Sequence[str] | None,
) -> tuple[list[str], str]:
    source_unit_ids = sorted(
        str(row["unit_id"])
        for row in source["montage"]["input_units"]  # type: ignore[index]
    )
    if len(source_unit_ids) != len(set(source_unit_ids)):
        raise ValueError("source montage contains duplicate input unit IDs")
    if trusted_expected_unit_ids is None:
        expected = source_unit_ids
        inventory_source = "source_v3_shadow"
    else:
        expected = sorted(str(value) for value in trusted_expected_unit_ids)
        if not expected or len(expected) != len(set(expected)):
            raise ValueError("trusted expected-unit inventory must be non-empty and unique")
        if expected != source_unit_ids:
            raise ValueError(
                "source montage input units do not match the trusted expected-unit inventory"
            )
        inventory_source = "trusted_canonical_montage_receipt"
    if not expected:
        raise ValueError("event Findings atom roster requires at least one expected unit")
    return expected, inventory_source


def materialize_event_findings_atom_roster_receipt(
    event_findings_v3: object,
    *,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    trusted_expected_unit_ids: Sequence[str] | None = None,
    technical_failure_atom_ids: Sequence[str] = (),
    technical_failure_child_roster_ids: Sequence[str] = (),
    findings_validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Materialize one deterministic structural-completeness receipt.

    Technical failures are host-supplied and explicit.  They close the
    structural denominator as ``not_evaluable`` but can never create a medical
    negative or a report-eligible claim.
    """

    roster_policy = _policy(policy, trusted_policy_sha256)
    source = _validated_findings(event_findings_v3, findings_validation_kwargs)
    _validate_sharp_context_partition_findings(source, roster_policy)
    expected_unit_ids, inventory_source = _validate_unit_inventory(
        source, trusted_expected_unit_ids
    )

    core_specs = {
        str(row["atom_id"]): row for row in roster_policy["core_atom_specs"]
    }
    child_specs = {
        str(row["child_roster_id"]): row
        for row in roster_policy["child_roster_specs"]
    }
    technical_atoms = set(str(value) for value in technical_failure_atom_ids)
    technical_children = set(
        str(value) for value in technical_failure_child_roster_ids
    )
    unexpected_atoms = sorted(technical_atoms - set(core_specs))
    unexpected_children = sorted(technical_children - set(child_specs))
    if unexpected_atoms or unexpected_children:
        raise ValueError(
            "technical-failure identifiers are not registered: "
            + ", ".join(unexpected_atoms + unexpected_children)
        )

    core_entries = [
        _core_entry(
            source=source,
            policy=roster_policy,
            spec=core_specs[atom_id],
            expected_unit_ids=expected_unit_ids,
            technical_failure=atom_id in technical_atoms,
        )
        for atom_id in sorted(core_specs)
    ]
    child_rosters = [
        _child_roster(
            source=source,
            policy=roster_policy,
            spec=child_specs[roster_id],
            expected_unit_ids=expected_unit_ids,
            technical_failure=roster_id in technical_children,
        )
        for roster_id in sorted(child_specs)
    ]

    status_rows: list[Mapping[str, object]] = core_entries + child_rosters
    critical_ids = sorted(
        atom_id
        for atom_id, row in core_specs.items()
        if row["salience_tier"] == "critical"
    )
    major_ids = sorted(
        atom_id
        for atom_id, row in core_specs.items()
        if row["salience_tier"] == "major"
    )
    core_ids = sorted(core_specs)
    child_ids = sorted(child_specs)
    unit_disposition_count = sum(
        len(row["unit_dispositions"]) for row in status_rows
    )
    summary = {
        "expected_core_atom_count": len(core_ids),
        "accounted_core_atom_count": len(core_entries),
        "expected_child_roster_count": len(child_ids),
        "accounted_child_roster_count": len(child_rosters),
        "critical_expected_count": len(critical_ids),
        "critical_accounted_count": sum(
            row["atom_id"] in critical_ids for row in core_entries
        ),
        "major_expected_count": len(major_ids),
        "major_accounted_count": sum(
            row["atom_id"] in major_ids for row in core_entries
        ),
        "expected_unit_count": len(expected_unit_ids),
        "accounted_unit_count": len(expected_unit_ids),
        "unit_disposition_count": unit_disposition_count,
        "present_count": sum(row["status"] == "present" for row in core_entries)
        + sum(row["finding_status"] == "present" for row in child_rosters),
        "absent_with_opportunity_count": sum(
            row["status"] == "absent_with_opportunity" for row in core_entries
        )
        + sum(
            row["finding_status"] == "absent_with_opportunity"
            for row in child_rosters
        ),
        "uncertain_count": sum(
            row["status"] == "uncertain" for row in core_entries
        )
        + sum(row["finding_status"] == "uncertain" for row in child_rosters),
        "not_evaluable_count": sum(
            row["status"] == "not_evaluable" for row in core_entries
        )
        + sum(
            row["finding_status"] == "not_evaluable" for row in child_rosters
        ),
        "technical_failure_count": sum(
            row["processing_disposition"] == "technical_failure"
            for row in status_rows
        ),
        "evidence_bound_core_count": sum(
            bool(row["finding_ids"] or row["source_paths"])
            for row in core_entries
        ),
        "expected_core_keys_sha256": _canonical_sha256(core_ids),
        "materialized_core_keys_sha256": _canonical_sha256(
            [row["atom_id"] for row in core_entries]
        ),
        "expected_child_keys_sha256": _canonical_sha256(child_ids),
        "materialized_child_keys_sha256": _canonical_sha256(
            [row["child_roster_id"] for row in child_rosters]
        ),
        "all_core_atoms_accounted_for": True,
        "all_child_rosters_accounted_for": True,
        "structural_completeness_only": True,
        "clinical_correctness_claimed": False,
        "full_report_factuality_claimed": False,
    }

    findings_hash = _canonical_sha256(source)
    inventory_hash = _canonical_sha256(expected_unit_ids)
    receipt_seed = {
        "event_id": source["event_id"],
        "findings_payload_sha256": findings_hash,
        "roster_policy_sha256": roster_policy["policy_sha256"],
        "expected_unit_inventory_sha256": inventory_hash,
        "method_id": EVENT_FINDINGS_ATOM_ROSTER_METHOD_ID,
    }
    receipt: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_ATOM_ROSTER_RECEIPT_SCHEMA_VERSION,
        "receipt_id": f"ATOMROSTER-{_canonical_sha256(receipt_seed)[:24]}",
        "event_id": str(source["event_id"]),
        "record_id": str(source["provenance"]["record_id"]),
        "canonical_signal_sha256": str(
            source["provenance"]["canonical_signal_sha256"]
        ),
        "expected_unit_inventory_source": inventory_source,
        "expected_unit_ids": expected_unit_ids,
        "expected_unit_inventory_sha256": inventory_hash,
        "findings_schema_version": "event_eeg_findings_v3",
        "findings_payload_sha256": findings_hash,
        "roster_id": str(roster_policy["roster_id"]),
        "roster_policy_sha256": str(roster_policy["policy_sha256"]),
        "core_entries": core_entries,
        "child_rosters": child_rosters,
        "summary": summary,
        "receipt_sha256": "0" * 64,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")

    errors = _schema_errors(
        receipt,
        EVENT_FINDINGS_ATOM_ROSTER_RECEIPT_SCHEMA_PATH,
    )
    if errors:
        raise ValueError(
            "materialized event Findings atom-roster receipt is invalid: "
            + "; ".join(errors)
        )
    return receipt


def validate_event_findings_atom_roster_receipt(
    value: object,
    *,
    event_findings_v3: object,
    policy: Mapping[str, object] | None = None,
    trusted_policy_sha256: str | None = None,
    trusted_expected_unit_ids: Sequence[str] | None = None,
    require_trusted_unit_inventory: bool = False,
    technical_failure_atom_ids: Sequence[str] = (),
    technical_failure_child_roster_ids: Sequence[str] = (),
    findings_validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate a receipt by exact, source-bound deterministic replay."""

    if type(value) is not dict:
        raise TypeError("event Findings atom-roster receipt must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(
        candidate,
        EVENT_FINDINGS_ATOM_ROSTER_RECEIPT_SCHEMA_PATH,
    )
    if errors:
        raise ValueError(
            "event Findings atom-roster receipt schema validation failed: "
            + "; ".join(errors)
        )
    if candidate["receipt_sha256"] != _self_hash(candidate, "receipt_sha256"):
        raise ValueError("event Findings atom-roster receipt_sha256 mismatch")
    if (
        require_trusted_unit_inventory
        and candidate["expected_unit_inventory_source"]
        != "trusted_canonical_montage_receipt"
    ):
        raise ValueError(
            "downstream promotion requires a trusted canonical expected-unit inventory"
        )

    expected = materialize_event_findings_atom_roster_receipt(
        event_findings_v3,
        policy=policy,
        trusted_policy_sha256=trusted_policy_sha256,
        trusted_expected_unit_ids=trusted_expected_unit_ids,
        technical_failure_atom_ids=technical_failure_atom_ids,
        technical_failure_child_roster_ids=technical_failure_child_roster_ids,
        findings_validation_kwargs=findings_validation_kwargs,
    )
    if _canonical_json(candidate) != _canonical_json(expected):
        raise ValueError(
            "event Findings atom-roster receipt does not exactly replay from "
            "the trusted policy, source Findings, inventory, and failure ledger"
        )
    return candidate


__all__ = [
    "DEFAULT_EVENT_FINDINGS_ATOM_ROSTER_POLICY_PATH",
    "EVENT_FINDINGS_ATOM_ROSTER_METHOD_ID",
    "EVENT_FINDINGS_ATOM_ROSTER_POLICY_SCHEMA_PATH",
    "EVENT_FINDINGS_ATOM_ROSTER_POLICY_SCHEMA_VERSION",
    "EVENT_FINDINGS_ATOM_ROSTER_RECEIPT_SCHEMA_PATH",
    "EVENT_FINDINGS_ATOM_ROSTER_RECEIPT_SCHEMA_VERSION",
    "load_event_findings_atom_roster_policy",
    "materialize_event_findings_atom_roster_receipt",
    "validate_event_findings_atom_roster_policy",
    "validate_event_findings_atom_roster_receipt",
]
