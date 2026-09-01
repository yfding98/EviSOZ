"""Fail-closed S07/S08 to native S03--S06 trigger attribution.

This additive module implements the missing *content binding* portion of the
v1.5 Findings freeze.  It accepts only caller-trusted, content-addressed
candidate rows and deterministic native-measurement atoms.  The module never
reads EEG files, annotations, spreadsheets, clinical text, detector scores,
attention, saliency, or late-course features.

``allowed`` in this wire means only that a native measurement atom is legally
bound to one research onset candidate under the frozen permission rules.  It
does not authorize a clinical term, an SOZ claim, a rank contribution, or a
report sentence.  Rank deletion/insertion rows are deliberately emitted as
unexecuted requests: no ranker or threshold registry is present in v1.5, so
no counterfactual delta or minimal trigger set is manufactured.

Whole bipolar leads remain directed ``lead`` units.  Their source-channel
order must exactly match the lead ID and no endpoint attribution field exists
in either the input or output schema.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Final

from src.soz.geometry import STANDARD_19


ONSET_TRIGGER_ATTRIBUTION_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_onset_trigger_attribution_v1_5_1"
)
ONSET_TRIGGER_ATTRIBUTION_CONTEXT_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_onset_trigger_attribution_context_v1_5_1"
)
ONSET_TRIGGER_ATTRIBUTION_METHOD_ID: Final[str] = (
    "EEG-NATIVE-ONSET-TRIGGER-ATTRIBUTION-V1.5.1"
)
PARENT_FINDINGS_FREEZE_SHA256: Final[str] = (
    "d460dafdfe5e76a90369ec0939becb7f92e9beb28ee40e12b72eccdfbcdc1ed1"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 1e-9

_CANDIDATE_SLOTS = frozenset({"S07", "S08"})
_TRIGGER_SLOT_DOMAINS: Final[dict[str, str]] = {
    "S03": "frequency_spectrum",
    "S04": "physical_amplitude",
    "S05": "waveform_numeric_primitive",
    "S06": "component_cycle_interval",
}
_NAMESPACES = frozenset({"measurement", "model_candidate"})
_MEASUREMENT_STATES = frozenset(
    {"present", "uncertain", "not_evaluable", "absent_with_opportunity"}
)
_CANDIDATE_STATES = frozenset({"present", "uncertain", "not_evaluable"})
_PERMISSION_LANES = frozenset(
    {"onset_causal", "course_offline", "matched_context", "limitation"}
)
_REFERENCE_FAMILIES = frozenset(
    {"referential", "bipolar", "common_average", "laplacian"}
)
_RANK_ELECTRODE_ONTOLOGY: Final[frozenset[str]] = frozenset(STANDARD_19)
_QUERY_TRANSITIONS = frozenset(
    {
        "first_observed",
        "first_observed_and_stabilized",
        "updated_unstable",
        "stabilized",
        "changed_after_stabilization",
        "invalidated",
    }
)
_STABLE_QUERY_TRANSITIONS = frozenset(
    {"first_observed_and_stabilized", "stabilized"}
)
_UNCERTAIN_QUERY_TRANSITIONS = frozenset(
    {"first_observed", "updated_unstable", "changed_after_stabilization"}
)

_CONTEXT_FIREWALL: Final[dict[str, bool]] = {
    "EEG_samples_used": True,
    "allowlisted_acquisition_metadata_used": True,
    "EDF_annotations_used": False,
    "spreadsheet_or_Excel_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_history_used": False,
    "patient_identity_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_or_activation_used": False,
    "ECG_EMG_EOG_used": False,
    "LLM_used": False,
}

_ATOM_SOURCE_FIREWALL: Final[dict[str, bool]] = {
    "deterministic_native_EEG_remeasurement_used": True,
    "attention_used": False,
    "saliency_used": False,
    "detector_posterior_used": False,
    "detector_score_used": False,
    "late_course_feature_used_as_trigger": False,
    "EDF_annotations_used": False,
    "spreadsheet_or_Excel_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_history_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_or_activation_used": False,
    "ECG_EMG_EOG_used": False,
    "LLM_used": False,
}

_AUTHORIZATION: Final[dict[str, bool | str | list[str]]] = {
    "scope": "research_native_trigger_content_binding_only",
    "automated_clinical_term_allowlist": [],
    "clinical_term_authorized": False,
    "clinical_or_production_use": False,
    "positive_rank_contribution_authorized_by_this_module": False,
    "SOZ_EZ_or_surgical_target_claim_authorized": False,
    "report_text_authorized": False,
    "attention_saliency_or_detector_output_accepted_as_trigger": False,
    "late_course_may_promote_positive_onset": False,
    "whole_bipolar_lead_identity_preserved": True,
    "bipolar_endpoint_attribution_authorized": False,
    "rank_counterfactual_execution_status": "interface_only_not_executed",
}

_PROTECTED_MEASUREMENT_NAMES = (
    "spike",
    "sharp_wave",
    "interictal_epileptiform",
    "ied",
    "pathological_theta",
    "rhythmic_theta",
    "periodic_discharge",
    "definite_evolution",
    "spread",
    "propagation",
    "postictal",
    "lvfa",
    "low_voltage_fast_activity",
    "electrodecrement",
    "hfo",
    "dc_shift",
    "electrographic_seizure",
    "phase_reversal",
    "cortical_soz",
    "epileptogenic_zone",
    "surgical_target",
)

_CONTEXT_FIELDS = {
    "schema_version",
    "recording_id",
    "occurrence_id",
    "query_index",
    "locked_prefix_query_index",
    "final_left_edge_s",
    "locked_causal_prefix_interval_s",
    "k3_interval_s",
    "temporal_tolerance_s",
    "candidate_generator_receipt_sha256",
    "native_operator_registry_receipt_sha256",
    "final_left_closure_receipt_sha256",
    "final_left_support_union_sha256",
    "locked_causal_prefix_receipt_sha256",
    "k3_gate_receipt_sha256",
    "reference_policy_receipt_sha256",
    "temporal_tolerance_registry_receipt_sha256",
    "onset_trigger_threshold_registry_receipt_sha256",
    "threshold_registry_admitted",
    "source_firewall",
    "context_content_sha256",
}

_CANDIDATE_FIELDS = {
    "candidate_id",
    "source_slot_id",
    "recording_id",
    "occurrence_id",
    "query_index",
    "recording_relative_earliest_interval_s",
    "typed_unit",
    "reference_family",
    "namespace",
    "candidate_state",
    "temporal_role",
    "future_sample_access",
    "onset_evidence_authorized",
    "whole_bipolar_lead_identity_preserved",
    "bipolar_endpoint_attribution_authorized",
    "candidate_generator_receipt_sha256",
    "reference_transform_receipt_sha256",
    "candidate_validation_receipt_sha256",
    "candidate_content_sha256",
}

_ATOM_FIELDS = {
    "measurement_atom_id",
    "source_proposal_ids",
    "source_slot_id",
    "measurement_domain",
    "namespace",
    "recording_id",
    "occurrence_id",
    "query_index",
    "recording_relative_half_open_interval_s",
    "change_interval_s",
    "raw_dependency_sha256s",
    "raw_dependency_interval_union_s",
    "typed_unit",
    "canonical_source_channels",
    "reference_family",
    "sample_rate_hz",
    "physical_unit",
    "effective_bandwidth_hz",
    "required_bandwidth_hz",
    "qc_opportunity_censor",
    "qc_opportunity_state",
    "bandwidth_state",
    "measurement_opportunity_state",
    "effect_threshold_state",
    "minimum_persistence_state",
    "query_transition_state",
    "operator_id",
    "operator_version",
    "effect_size_and_unit",
    "uncertainty",
    "permission_lane",
    "native_remeasurement_verified",
    "future_sample_access",
    "late_course_feature_used",
    "whole_bipolar_lead_identity_preserved",
    "bipolar_endpoint_attribution_authorized",
    "trigger_source_firewall",
    "transform_receipt_sha256",
    "operator_parameter_receipt_sha256",
    "native_measurement_validation_receipt_sha256",
    "raw_dependency_receipt_sha256",
    "reference_transform_receipt_sha256",
    "qc_opportunity_receipt_sha256",
    "bandwidth_receipt_sha256",
    "effect_threshold_decision_receipt_sha256",
    "minimum_persistence_decision_receipt_sha256",
    "query_closure_receipt_sha256",
    "producer_receipt_sha256",
    "permission_receipt_sha256",
    "measurement_state",
    "measurement_content_sha256",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
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


def _exact_mapping(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    result = deepcopy(dict(value))
    if set(result) != fields:
        missing = sorted(fields - set(result))
        extra = sorted(set(result) - fields)
        raise ValueError(f"{context} fields drifted; missing={missing}, extra={extra}")
    return result


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _unit_string(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 64
        or any(character.isspace() for character in value)
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a compact physical-unit string")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a non-negative integer")
    return value


def _finite(
    value: object,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{context} must be <= {maximum}")
    return result


def _interval(value: object, context: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"{context} must be a two-number interval")
    start = _finite(value[0], f"{context}[0]", minimum=0.0)
    stop = _finite(value[1], f"{context}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{context} must be a non-empty half-open interval")
    return [start, stop]


def _contains(carrier: Sequence[float], item: Sequence[float], tolerance: float) -> bool:
    return (
        float(item[0]) >= float(carrier[0]) - tolerance
        and float(item[1]) <= float(carrier[1]) + tolerance
    )


def _canonical_interval_union(value: object, context: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an interval array")
    rows = [_interval(row, f"{context}[{index}]") for index, row in enumerate(value)]
    expected = sorted(rows, key=lambda row: (row[0], row[1]))
    if rows != expected:
        raise ValueError(f"{context} must already be sorted")
    for left, right in zip(rows, rows[1:]):
        if right[0] <= left[1] + _TOL:
            raise ValueError(f"{context} must already be canonical and disjoint")
    return rows


def _union_inside_interval(
    union: Sequence[Sequence[float]], carrier: Sequence[float], tolerance: float
) -> bool:
    return all(_contains(carrier, row, tolerance) for row in union)


def _interval_inside_union(
    interval: Sequence[float], union: Sequence[Sequence[float]], tolerance: float
) -> bool:
    return any(_contains(row, interval, tolerance) for row in union)


def _validated_trust_set(
    value: Collection[str], context: str, *, allow_empty: bool = False
) -> set[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a collection of SHA-256 values")
    result = {_sha256(item, context) for item in value}
    if not result and not allow_empty:
        raise ValueError(f"{context} cannot be empty")
    return result


def _require_trusted(value: str, trusted: set[str], context: str) -> None:
    if value not in trusted:
        raise ValueError(f"{context} is not in the caller-trusted receipt set")


def _typed_unit(value: object, context: str) -> dict[str, str]:
    row = _exact_mapping(value, {"unit_type", "unit_id", "unit_key"}, context)
    unit_type = _identifier(row["unit_type"], f"{context}.unit_type")
    unit_id = _identifier(row["unit_id"], f"{context}.unit_id")
    unit_key = _identifier(row["unit_key"], f"{context}.unit_key")
    if unit_type not in {"electrode", "lead"}:
        raise ValueError(f"{context}.unit_type is unsupported")
    if unit_key != f"{unit_type}:{unit_id}":
        raise ValueError(f"{context}.unit_key is not type-safe")
    if unit_type == "electrode" and unit_id not in _RANK_ELECTRODE_ONTOLOGY:
        raise ValueError(f"{context}.unit_id is outside the frozen rank-electrode ontology")
    if unit_type == "lead":
        endpoints = unit_id.split("-")
        if (
            len(endpoints) != 2
            or endpoints[0] == endpoints[1]
            or any(endpoint not in _RANK_ELECTRODE_ONTOLOGY for endpoint in endpoints)
        ):
            raise ValueError(
                f"{context}.unit_id is outside the frozen directed whole-lead ontology"
            )
    return {"unit_type": unit_type, "unit_id": unit_id, "unit_key": unit_key}


def _lead_endpoints(unit: Mapping[str, str], context: str) -> list[str]:
    parts = unit["unit_id"].split("-")
    if len(parts) != 2 or any(_ID_RE.fullmatch(part) is None for part in parts):
        raise ValueError(f"{context} lead must encode exactly two directed endpoints")
    if parts[0] == parts[1]:
        raise ValueError(f"{context} lead cannot repeat one endpoint")
    return parts


def _validate_reference_identity(
    unit: Mapping[str, str], reference_family: str, context: str
) -> None:
    if reference_family not in _REFERENCE_FAMILIES:
        raise ValueError(f"{context}.reference_family is unsupported")
    expected = {
        "electrode": {"referential"},
        "lead": {"bipolar"},
    }[unit["unit_type"]]
    if reference_family not in expected:
        raise ValueError(f"{context} typed unit/reference family crossed")
    if unit["unit_type"] == "lead":
        _lead_endpoints(unit, context)


def _validate_exact_bool_mapping(
    value: object, expected: Mapping[str, bool], context: str
) -> dict[str, bool]:
    row = _exact_mapping(value, set(expected), context)
    for key, required in expected.items():
        if row[key] is not required:
            raise ValueError(f"{context}.{key} must remain {required}")
    return dict(expected)


def _validated_context(value: object, trusted_receipts: set[str]) -> dict[str, Any]:
    row = _exact_mapping(value, _CONTEXT_FIELDS, "context")
    if row["schema_version"] != ONSET_TRIGGER_ATTRIBUTION_CONTEXT_SCHEMA_VERSION:
        raise ValueError("context schema version drifted")
    row["recording_id"] = _identifier(row["recording_id"], "context.recording_id")
    row["occurrence_id"] = _identifier(row["occurrence_id"], "context.occurrence_id")
    row["query_index"] = _nonnegative_int(row["query_index"], "context.query_index")
    row["locked_prefix_query_index"] = _nonnegative_int(
        row["locked_prefix_query_index"], "context.locked_prefix_query_index"
    )
    if row["locked_prefix_query_index"] > row["query_index"]:
        raise ValueError("locked prefix query cannot be in the future")
    row["final_left_edge_s"] = _finite(
        row["final_left_edge_s"], "context.final_left_edge_s", minimum=0.0
    )
    row["locked_causal_prefix_interval_s"] = _interval(
        row["locked_causal_prefix_interval_s"],
        "context.locked_causal_prefix_interval_s",
    )
    row["k3_interval_s"] = _interval(row["k3_interval_s"], "context.k3_interval_s")
    row["temporal_tolerance_s"] = _finite(
        row["temporal_tolerance_s"],
        "context.temporal_tolerance_s",
        minimum=0.0,
        maximum=5.0,
    )
    # These are two serialized boundaries in the same physical-time ledger,
    # not two uncertain physiological estimates.  They must agree exactly up
    # to numerical serialization tolerance; the candidate-overlap tolerance
    # must not move the locked prefix or admit an out-of-prefix K3 interval.
    if (
        abs(
            row["locked_causal_prefix_interval_s"][0]
            - row["final_left_edge_s"]
        )
        > _TOL
    ):
        raise ValueError("locked causal prefix is not bound to the final-left edge")
    if not _contains(
        row["locked_causal_prefix_interval_s"], row["k3_interval_s"], _TOL
    ):
        raise ValueError("K3 interval must remain inside the locked causal prefix")
    receipt_fields = (
        "candidate_generator_receipt_sha256",
        "native_operator_registry_receipt_sha256",
        "final_left_closure_receipt_sha256",
        "final_left_support_union_sha256",
        "locked_causal_prefix_receipt_sha256",
        "k3_gate_receipt_sha256",
        "reference_policy_receipt_sha256",
        "temporal_tolerance_registry_receipt_sha256",
        "onset_trigger_threshold_registry_receipt_sha256",
    )
    for field in receipt_fields:
        row[field] = _sha256(row[field], f"context.{field}")
        _require_trusted(row[field], trusted_receipts, f"context.{field}")
    if row["threshold_registry_admitted"] is not False:
        raise ValueError("v1.5 has no admitted real onset-trigger threshold registry")
    row["source_firewall"] = _validate_exact_bool_mapping(
        row["source_firewall"], _CONTEXT_FIREWALL, "context.source_firewall"
    )
    row["context_content_sha256"] = _sha256(
        row["context_content_sha256"], "context.context_content_sha256"
    )
    if row["context_content_sha256"] != _self_hash(row, "context_content_sha256"):
        raise ValueError("context content hash does not replay")
    _require_trusted(
        row["context_content_sha256"], trusted_receipts, "context content receipt"
    )
    return row


def _validated_candidate(
    value: object,
    *,
    index: int,
    context: Mapping[str, Any],
    trusted_receipts: set[str],
    trusted_content: set[str],
) -> dict[str, Any]:
    prefix = f"candidates[{index}]"
    row = _exact_mapping(value, _CANDIDATE_FIELDS, prefix)
    row["candidate_id"] = _identifier(row["candidate_id"], f"{prefix}.candidate_id")
    if row["source_slot_id"] not in _CANDIDATE_SLOTS:
        raise ValueError(f"{prefix}.source_slot_id must be S07 or S08")
    for field in ("recording_id", "occurrence_id"):
        row[field] = _identifier(row[field], f"{prefix}.{field}")
        if row[field] != context[field]:
            raise ValueError(f"{prefix}.{field} crossed the attribution context")
    row["query_index"] = _nonnegative_int(row["query_index"], f"{prefix}.query_index")
    row["recording_relative_earliest_interval_s"] = _interval(
        row["recording_relative_earliest_interval_s"],
        f"{prefix}.recording_relative_earliest_interval_s",
    )
    row["typed_unit"] = _typed_unit(row["typed_unit"], f"{prefix}.typed_unit")
    row["reference_family"] = _identifier(
        row["reference_family"], f"{prefix}.reference_family"
    )
    _validate_reference_identity(row["typed_unit"], row["reference_family"], prefix)
    if row["namespace"] != "model_candidate":
        raise ValueError(f"{prefix}.namespace must remain model_candidate")
    if row["candidate_state"] not in _CANDIDATE_STATES:
        raise ValueError(f"{prefix}.candidate_state is unsupported")
    if row["temporal_role"] != "onset_causal":
        raise ValueError(f"{prefix}.temporal_role must remain onset_causal")
    for field, required in (
        ("future_sample_access", False),
        ("onset_evidence_authorized", True),
        ("whole_bipolar_lead_identity_preserved", True),
        ("bipolar_endpoint_attribution_authorized", False),
    ):
        if row[field] is not required:
            raise ValueError(f"{prefix}.{field} must remain {required}")
    for field in (
        "candidate_generator_receipt_sha256",
        "reference_transform_receipt_sha256",
        "candidate_validation_receipt_sha256",
    ):
        row[field] = _sha256(row[field], f"{prefix}.{field}")
        _require_trusted(row[field], trusted_receipts, f"{prefix}.{field}")
    if (
        row["candidate_generator_receipt_sha256"]
        != context["candidate_generator_receipt_sha256"]
    ):
        raise ValueError(f"{prefix} generator receipt crossed the context")
    row["candidate_content_sha256"] = _sha256(
        row["candidate_content_sha256"], f"{prefix}.candidate_content_sha256"
    )
    if row["candidate_content_sha256"] != _self_hash(row, "candidate_content_sha256"):
        raise ValueError(f"{prefix} content hash does not replay")
    _require_trusted(
        row["candidate_content_sha256"], trusted_content, f"{prefix} content"
    )
    return row


def _validated_effect(value: object, context: str) -> dict[str, Any]:
    row = _exact_mapping(
        value,
        {"measurement_name_id", "value", "unit", "semantics"},
        context,
    )
    row["measurement_name_id"] = _identifier(
        row["measurement_name_id"], f"{context}.measurement_name_id"
    )
    lowered = row["measurement_name_id"].lower()
    if any(term in lowered for term in _PROTECTED_MEASUREMENT_NAMES):
        raise ValueError(f"{context} attempts to open a protected clinical term")
    row["value"] = _finite(row["value"], f"{context}.value")
    row["unit"] = _unit_string(row["unit"], f"{context}.unit")
    if row["semantics"] != "native_measurement_effect_not_rank_delta_or_clinical_causality":
        raise ValueError(f"{context}.semantics drifted")
    return row


def _validated_uncertainty(
    value: object, *, effect: Mapping[str, Any], context: str
) -> dict[str, Any]:
    row = _exact_mapping(value, {"status", "lower", "upper", "unit"}, context)
    if row["status"] not in {"bounded", "not_established"}:
        raise ValueError(f"{context}.status is unsupported")
    row["unit"] = _unit_string(row["unit"], f"{context}.unit")
    if row["unit"] != effect["unit"]:
        raise ValueError(f"{context} unit differs from the native effect")
    if row["status"] == "not_established":
        if row["lower"] is not None or row["upper"] is not None:
            raise ValueError(f"{context} unestablished bounds must remain null")
    else:
        row["lower"] = _finite(row["lower"], f"{context}.lower")
        row["upper"] = _finite(row["upper"], f"{context}.upper")
        if row["lower"] > row["upper"] + _TOL:
            raise ValueError(f"{context} bounds are reversed")
        if not row["lower"] - _TOL <= effect["value"] <= row["upper"] + _TOL:
            raise ValueError(f"{context} does not contain the effect estimate")
    return row


def _validated_atom(
    value: object,
    *,
    index: int,
    context: Mapping[str, Any],
    trusted_receipts: set[str],
    trusted_content: set[str],
) -> dict[str, Any]:
    prefix = f"measurement_atoms[{index}]"
    row = _exact_mapping(value, _ATOM_FIELDS, prefix)
    row["measurement_atom_id"] = _identifier(
        row["measurement_atom_id"], f"{prefix}.measurement_atom_id"
    )
    if not isinstance(row["source_proposal_ids"], Sequence) or isinstance(
        row["source_proposal_ids"], (str, bytes)
    ):
        raise TypeError(f"{prefix}.source_proposal_ids must be an ID array")
    proposal_ids = [
        _identifier(item, f"{prefix}.source_proposal_ids")
        for item in row["source_proposal_ids"]
    ]
    if not proposal_ids or proposal_ids != sorted(set(proposal_ids)):
        raise ValueError(f"{prefix}.source_proposal_ids must be non-empty, unique, sorted")
    row["source_proposal_ids"] = proposal_ids
    if row["source_slot_id"] not in _TRIGGER_SLOT_DOMAINS:
        raise ValueError(f"{prefix}.source_slot_id must be S03--S06")
    if row["measurement_domain"] != _TRIGGER_SLOT_DOMAINS[row["source_slot_id"]]:
        raise ValueError(f"{prefix} slot and numeric measurement domain crossed")
    if row["namespace"] not in _NAMESPACES:
        raise ValueError(f"{prefix}.namespace is unsupported")
    for field in ("recording_id", "occurrence_id"):
        row[field] = _identifier(row[field], f"{prefix}.{field}")
        if row[field] != context[field]:
            raise ValueError(f"{prefix}.{field} crossed the attribution context")
    row["query_index"] = _nonnegative_int(row["query_index"], f"{prefix}.query_index")
    row["recording_relative_half_open_interval_s"] = _interval(
        row["recording_relative_half_open_interval_s"],
        f"{prefix}.recording_relative_half_open_interval_s",
    )
    row["change_interval_s"] = _interval(
        row["change_interval_s"], f"{prefix}.change_interval_s"
    )
    if not _contains(
        row["recording_relative_half_open_interval_s"],
        row["change_interval_s"],
        _TOL,
    ):
        raise ValueError(f"{prefix}.change_interval_s leaves its measurement interval")
    if not isinstance(row["raw_dependency_sha256s"], Sequence) or isinstance(
        row["raw_dependency_sha256s"], (str, bytes)
    ):
        raise TypeError(f"{prefix}.raw_dependency_sha256s must be an array")
    raw_hashes = [
        _sha256(item, f"{prefix}.raw_dependency_sha256s")
        for item in row["raw_dependency_sha256s"]
    ]
    if raw_hashes != sorted(set(raw_hashes)):
        raise ValueError(f"{prefix}.raw_dependency_sha256s must be unique and sorted")
    row["raw_dependency_sha256s"] = raw_hashes
    row["raw_dependency_interval_union_s"] = _canonical_interval_union(
        row["raw_dependency_interval_union_s"],
        f"{prefix}.raw_dependency_interval_union_s",
    )
    row["typed_unit"] = _typed_unit(row["typed_unit"], f"{prefix}.typed_unit")
    row["reference_family"] = _identifier(
        row["reference_family"], f"{prefix}.reference_family"
    )
    _validate_reference_identity(row["typed_unit"], row["reference_family"], prefix)
    if not isinstance(row["canonical_source_channels"], Sequence) or isinstance(
        row["canonical_source_channels"], (str, bytes)
    ):
        raise TypeError(f"{prefix}.canonical_source_channels must be an array")
    source_channels = [
        _identifier(item, f"{prefix}.canonical_source_channels")
        for item in row["canonical_source_channels"]
    ]
    if not source_channels or len(source_channels) != len(set(source_channels)):
        raise ValueError(f"{prefix}.canonical_source_channels must be non-empty and unique")
    if row["typed_unit"]["unit_type"] == "lead" and source_channels != _lead_endpoints(
        row["typed_unit"], prefix
    ):
        raise ValueError(f"{prefix} changed directed whole-bipolar-lead identity")
    if row["typed_unit"]["unit_type"] == "electrode" and source_channels != [
        row["typed_unit"]["unit_id"]
    ]:
        raise ValueError(f"{prefix} referential electrode source identity drifted")
    row["canonical_source_channels"] = source_channels
    row["sample_rate_hz"] = _finite(
        row["sample_rate_hz"], f"{prefix}.sample_rate_hz", minimum=_TOL
    )
    row["physical_unit"] = _unit_string(row["physical_unit"], f"{prefix}.physical_unit")
    row["effective_bandwidth_hz"] = _interval(
        row["effective_bandwidth_hz"], f"{prefix}.effective_bandwidth_hz"
    )
    row["required_bandwidth_hz"] = _interval(
        row["required_bandwidth_hz"], f"{prefix}.required_bandwidth_hz"
    )
    if row["qc_opportunity_state"] not in {"pass", "uncertain", "not_evaluable"}:
        raise ValueError(f"{prefix}.qc_opportunity_state is unsupported")
    if row["bandwidth_state"] not in {"pass", "uncertain", "not_evaluable"}:
        raise ValueError(f"{prefix}.bandwidth_state is unsupported")
    if row["measurement_opportunity_state"] not in {
        "sufficient",
        "uncertain",
        "not_evaluable",
    }:
        raise ValueError(f"{prefix}.measurement_opportunity_state is unsupported")
    for field in ("effect_threshold_state", "minimum_persistence_state"):
        if row[field] not in {"pass", "uncertain", "not_evaluable"}:
            raise ValueError(f"{prefix}.{field} is unsupported")
    if type(row["qc_opportunity_censor"]) is not bool:
        raise TypeError(f"{prefix}.qc_opportunity_censor must be boolean")
    if row["qc_opportunity_state"] == "pass" and row["qc_opportunity_censor"]:
        raise ValueError(f"{prefix} cannot pass QC while censored")
    if row["bandwidth_state"] == "pass" and not _contains(
        row["effective_bandwidth_hz"], row["required_bandwidth_hz"], _TOL
    ):
        raise ValueError(f"{prefix} passed bandwidth without covering its requirement")
    if row["query_transition_state"] not in _QUERY_TRANSITIONS:
        raise ValueError(f"{prefix}.query_transition_state is unsupported")
    row["operator_id"] = _identifier(row["operator_id"], f"{prefix}.operator_id")
    row["operator_version"] = _identifier(
        row["operator_version"], f"{prefix}.operator_version"
    )
    row["effect_size_and_unit"] = _validated_effect(
        row["effect_size_and_unit"], f"{prefix}.effect_size_and_unit"
    )
    if row["effect_size_and_unit"]["unit"] != row["physical_unit"]:
        raise ValueError(f"{prefix} physical unit differs from the effect unit")
    row["uncertainty"] = _validated_uncertainty(
        row["uncertainty"],
        effect=row["effect_size_and_unit"],
        context=f"{prefix}.uncertainty",
    )
    if row["permission_lane"] not in _PERMISSION_LANES:
        raise ValueError(f"{prefix}.permission_lane is unsupported")
    for field in (
        "native_remeasurement_verified",
        "future_sample_access",
        "late_course_feature_used",
        "whole_bipolar_lead_identity_preserved",
        "bipolar_endpoint_attribution_authorized",
    ):
        if type(row[field]) is not bool:
            raise TypeError(f"{prefix}.{field} must be boolean")
    if row["native_remeasurement_verified"] is not True:
        raise ValueError(f"{prefix} lacks deterministic native remeasurement")
    if row["whole_bipolar_lead_identity_preserved"] is not True:
        raise ValueError(f"{prefix} does not preserve whole bipolar leads")
    if row["bipolar_endpoint_attribution_authorized"] is not False:
        raise ValueError(f"{prefix} opened bipolar endpoint attribution")
    row["trigger_source_firewall"] = _validate_exact_bool_mapping(
        row["trigger_source_firewall"],
        _ATOM_SOURCE_FIREWALL,
        f"{prefix}.trigger_source_firewall",
    )
    receipt_fields = (
        "transform_receipt_sha256",
        "operator_parameter_receipt_sha256",
        "native_measurement_validation_receipt_sha256",
        "raw_dependency_receipt_sha256",
        "reference_transform_receipt_sha256",
        "qc_opportunity_receipt_sha256",
        "bandwidth_receipt_sha256",
        "effect_threshold_decision_receipt_sha256",
        "minimum_persistence_decision_receipt_sha256",
        "query_closure_receipt_sha256",
        "producer_receipt_sha256",
        "permission_receipt_sha256",
    )
    for field in receipt_fields:
        row[field] = _sha256(row[field], f"{prefix}.{field}")
        _require_trusted(row[field], trusted_receipts, f"{prefix}.{field}")
    if row["measurement_state"] not in _MEASUREMENT_STATES:
        raise ValueError(f"{prefix}.measurement_state is unsupported")
    if row["measurement_state"] != "not_evaluable" and (
        not row["raw_dependency_sha256s"]
        or not row["raw_dependency_interval_union_s"]
    ):
        raise ValueError(f"{prefix} evaluable state lacks raw dependencies")
    if row["measurement_state"] == "present" and (
        row["qc_opportunity_state"] != "pass"
        or row["bandwidth_state"] != "pass"
        or row["measurement_opportunity_state"] != "sufficient"
    ):
        raise ValueError(f"{prefix} present state contradicts opportunity gates")
    if row["measurement_state"] == "not_evaluable" and not (
        row["qc_opportunity_state"] == "not_evaluable"
        or row["bandwidth_state"] == "not_evaluable"
        or row["measurement_opportunity_state"] == "not_evaluable"
        or row["effect_threshold_state"] == "not_evaluable"
        or row["minimum_persistence_state"] == "not_evaluable"
        or row["query_transition_state"] == "invalidated"
    ):
        raise ValueError(f"{prefix} not-evaluable state lacks a typed gate")
    row["measurement_content_sha256"] = _sha256(
        row["measurement_content_sha256"], f"{prefix}.measurement_content_sha256"
    )
    if row["measurement_content_sha256"] != _self_hash(
        row, "measurement_content_sha256"
    ):
        raise ValueError(f"{prefix} content hash does not replay")
    _require_trusted(
        row["measurement_content_sha256"], trusted_content, f"{prefix} content"
    )
    return row


_REASON_ORDER = (
    "candidate_state_not_evaluable",
    "candidate_state_uncertain",
    "candidate_query_after_locked_prefix",
    "candidate_interval_outside_k3",
    "typed_unit_mismatch",
    "reference_family_mismatch",
    "reference_transform_receipt_mismatch",
    "atom_permission_lane_not_onset_causal",
    "atom_future_sample_access_forbidden",
    "atom_late_course_feature_forbidden",
    "atom_query_after_locked_prefix",
    "atom_measurement_interval_outside_locked_causal_prefix",
    "atom_change_interval_outside_locked_causal_prefix",
    "raw_dependency_outside_locked_causal_prefix",
    "change_interval_after_candidate_earliest_tolerance",
    "measurement_state_not_evaluable",
    "absent_with_opportunity_not_authorized_for_positive_trigger",
    "measurement_state_uncertain",
    "qc_opportunity_not_evaluable",
    "qc_opportunity_uncertain",
    "bandwidth_not_evaluable",
    "bandwidth_uncertain",
    "measurement_opportunity_not_evaluable",
    "measurement_opportunity_uncertain",
    "effect_threshold_not_evaluable",
    "effect_threshold_uncertain",
    "minimum_persistence_not_evaluable",
    "minimum_persistence_uncertain",
    "query_atom_invalidated",
    "query_atom_not_stabilized",
)
_REASON_INDEX = {reason: index for index, reason in enumerate(_REASON_ORDER)}


def _ordered_reasons(values: Collection[str]) -> list[str]:
    unknown = set(values) - set(_REASON_INDEX)
    if unknown:
        raise RuntimeError(f"unregistered onset-trigger reason codes: {sorted(unknown)}")
    return sorted(set(values), key=lambda value: _REASON_INDEX[value])


def _candidate_gate_reasons(
    candidate: Mapping[str, Any], context: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if candidate["candidate_state"] == "not_evaluable":
        reasons.append("candidate_state_not_evaluable")
    elif candidate["candidate_state"] == "uncertain":
        reasons.append("candidate_state_uncertain")
    if candidate["query_index"] > context["locked_prefix_query_index"]:
        reasons.append("candidate_query_after_locked_prefix")
    if not _contains(
        context["k3_interval_s"],
        candidate["recording_relative_earliest_interval_s"],
        context["temporal_tolerance_s"] + _TOL,
    ):
        reasons.append("candidate_interval_outside_k3")
    return _ordered_reasons(reasons)


def _atom_attribution(
    candidate: Mapping[str, Any],
    atom: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = _candidate_gate_reasons(candidate, context)
    if atom["typed_unit"] != candidate["typed_unit"]:
        reasons.append("typed_unit_mismatch")
    if atom["reference_family"] != candidate["reference_family"]:
        reasons.append("reference_family_mismatch")
    if (
        atom["reference_transform_receipt_sha256"]
        != candidate["reference_transform_receipt_sha256"]
    ):
        reasons.append("reference_transform_receipt_mismatch")
    if atom["permission_lane"] != "onset_causal":
        reasons.append("atom_permission_lane_not_onset_causal")
    if atom["future_sample_access"]:
        reasons.append("atom_future_sample_access_forbidden")
    if atom["late_course_feature_used"]:
        reasons.append("atom_late_course_feature_forbidden")
    if atom["query_index"] > context["locked_prefix_query_index"]:
        reasons.append("atom_query_after_locked_prefix")
    # Sample/raw/prefix containment is a digital-clock invariant.  It must
    # never inherit the (potentially seconds-wide) physiological overlap
    # tolerance used for comparing an estimated change with a candidate
    # earliest interval.  Half a sample admits only deterministic rounding at
    # physical-time/sample conversion boundaries and is <= one sample by
    # construction.
    sample_containment_tolerance = 0.5 / float(atom["sample_rate_hz"]) + _TOL
    candidate_overlap_tolerance = context["temporal_tolerance_s"] + _TOL
    if not _contains(
        context["locked_causal_prefix_interval_s"],
        atom["recording_relative_half_open_interval_s"],
        sample_containment_tolerance,
    ):
        reasons.append("atom_measurement_interval_outside_locked_causal_prefix")
    if not _contains(
        context["locked_causal_prefix_interval_s"],
        atom["change_interval_s"],
        sample_containment_tolerance,
    ):
        reasons.append("atom_change_interval_outside_locked_causal_prefix")
    if not _union_inside_interval(
        atom["raw_dependency_interval_union_s"],
        context["locked_causal_prefix_interval_s"],
        sample_containment_tolerance,
    ):
        reasons.append("raw_dependency_outside_locked_causal_prefix")
    if atom["raw_dependency_interval_union_s"] and not _interval_inside_union(
        atom["change_interval_s"],
        atom["raw_dependency_interval_union_s"],
        sample_containment_tolerance,
    ):
        reasons.append("raw_dependency_outside_locked_causal_prefix")
    if (
        atom["change_interval_s"][0]
        > candidate["recording_relative_earliest_interval_s"][1]
        + candidate_overlap_tolerance
    ):
        reasons.append("change_interval_after_candidate_earliest_tolerance")
    if atom["measurement_state"] == "not_evaluable":
        reasons.append("measurement_state_not_evaluable")
    elif atom["measurement_state"] == "absent_with_opportunity":
        reasons.append("absent_with_opportunity_not_authorized_for_positive_trigger")
    elif atom["measurement_state"] == "uncertain":
        reasons.append("measurement_state_uncertain")
    if atom["qc_opportunity_state"] == "not_evaluable":
        reasons.append("qc_opportunity_not_evaluable")
    elif atom["qc_opportunity_state"] == "uncertain":
        reasons.append("qc_opportunity_uncertain")
    if atom["bandwidth_state"] == "not_evaluable":
        reasons.append("bandwidth_not_evaluable")
    elif atom["bandwidth_state"] == "uncertain":
        reasons.append("bandwidth_uncertain")
    if atom["measurement_opportunity_state"] == "not_evaluable":
        reasons.append("measurement_opportunity_not_evaluable")
    elif atom["measurement_opportunity_state"] == "uncertain":
        reasons.append("measurement_opportunity_uncertain")
    if atom["effect_threshold_state"] == "not_evaluable":
        reasons.append("effect_threshold_not_evaluable")
    elif atom["effect_threshold_state"] == "uncertain":
        reasons.append("effect_threshold_uncertain")
    if atom["minimum_persistence_state"] == "not_evaluable":
        reasons.append("minimum_persistence_not_evaluable")
    elif atom["minimum_persistence_state"] == "uncertain":
        reasons.append("minimum_persistence_uncertain")
    if atom["query_transition_state"] == "invalidated":
        reasons.append("query_atom_invalidated")
    elif atom["query_transition_state"] in _UNCERTAIN_QUERY_TRANSITIONS:
        reasons.append("query_atom_not_stabilized")

    ordered = _ordered_reasons(reasons)
    hard = {
        "candidate_state_not_evaluable",
        "candidate_query_after_locked_prefix",
        "candidate_interval_outside_k3",
        "typed_unit_mismatch",
        "reference_family_mismatch",
        "reference_transform_receipt_mismatch",
        "atom_permission_lane_not_onset_causal",
        "atom_future_sample_access_forbidden",
        "atom_late_course_feature_forbidden",
        "atom_query_after_locked_prefix",
        "atom_measurement_interval_outside_locked_causal_prefix",
        "atom_change_interval_outside_locked_causal_prefix",
        "raw_dependency_outside_locked_causal_prefix",
        "change_interval_after_candidate_earliest_tolerance",
        "measurement_state_not_evaluable",
        "absent_with_opportunity_not_authorized_for_positive_trigger",
        "qc_opportunity_not_evaluable",
        "bandwidth_not_evaluable",
        "measurement_opportunity_not_evaluable",
        "effect_threshold_not_evaluable",
        "minimum_persistence_not_evaluable",
        "query_atom_invalidated",
    }
    if any(reason in hard for reason in ordered):
        state = "not_evaluable"
    elif ordered:
        state = "uncertain"
    else:
        state = "allowed"
    body: dict[str, Any] = {
        "measurement_atom_id": atom["measurement_atom_id"],
        "source_slot_id": atom["source_slot_id"],
        "namespace": atom["namespace"],
        "measurement_state": atom["measurement_state"],
        "permission_lane": atom["permission_lane"],
        "query_index": atom["query_index"],
        "query_transition_state": atom["query_transition_state"],
        "qc_opportunity_state": atom["qc_opportunity_state"],
        "bandwidth_state": atom["bandwidth_state"],
        "measurement_opportunity_state": atom["measurement_opportunity_state"],
        "effect_threshold_state": atom["effect_threshold_state"],
        "minimum_persistence_state": atom["minimum_persistence_state"],
        "future_sample_access": atom["future_sample_access"],
        "late_course_feature_used": atom["late_course_feature_used"],
        "sample_clock_containment_tolerance_s": sample_containment_tolerance,
        "candidate_physiologic_overlap_tolerance_s": candidate_overlap_tolerance,
        "binding_state": state,
        "structural_positive_trigger_binding_allowed": state == "allowed",
        "uncertainty_support_only": state == "uncertain",
        "positive_rank_contribution_authorized_by_this_module": False,
        "effect_size_and_unit": deepcopy(atom["effect_size_and_unit"]),
        "uncertainty": deepcopy(atom["uncertainty"]),
        "change_interval_s": deepcopy(atom["change_interval_s"]),
        "recording_relative_half_open_interval_s": deepcopy(
            atom["recording_relative_half_open_interval_s"]
        ),
        "typed_unit": deepcopy(atom["typed_unit"]),
        "reference_family": atom["reference_family"],
        "raw_dependency_sha256s": deepcopy(atom["raw_dependency_sha256s"]),
        "raw_dependency_interval_union_s": deepcopy(
            atom["raw_dependency_interval_union_s"]
        ),
        "effective_bandwidth_hz": deepcopy(atom["effective_bandwidth_hz"]),
        "required_bandwidth_hz": deepcopy(atom["required_bandwidth_hz"]),
        "receipt_binding": {
            "candidate_content_sha256": candidate["candidate_content_sha256"],
            "candidate_validation_receipt_sha256": candidate[
                "candidate_validation_receipt_sha256"
            ],
            "candidate_generator_receipt_sha256": context[
                "candidate_generator_receipt_sha256"
            ],
            "candidate_reference_transform_receipt_sha256": candidate[
                "reference_transform_receipt_sha256"
            ],
            "measurement_content_sha256": atom["measurement_content_sha256"],
            "native_measurement_validation_receipt_sha256": atom[
                "native_measurement_validation_receipt_sha256"
            ],
            "native_operator_registry_receipt_sha256": context[
                "native_operator_registry_receipt_sha256"
            ],
            "transform_receipt_sha256": atom["transform_receipt_sha256"],
            "operator_parameter_receipt_sha256": atom[
                "operator_parameter_receipt_sha256"
            ],
            "raw_dependency_receipt_sha256": atom["raw_dependency_receipt_sha256"],
            "reference_transform_receipt_sha256": atom[
                "reference_transform_receipt_sha256"
            ],
            "qc_opportunity_receipt_sha256": atom["qc_opportunity_receipt_sha256"],
            "bandwidth_receipt_sha256": atom["bandwidth_receipt_sha256"],
            "effect_threshold_decision_receipt_sha256": atom[
                "effect_threshold_decision_receipt_sha256"
            ],
            "minimum_persistence_decision_receipt_sha256": atom[
                "minimum_persistence_decision_receipt_sha256"
            ],
            "query_closure_receipt_sha256": atom["query_closure_receipt_sha256"],
            "locked_causal_prefix_receipt_sha256": context[
                "locked_causal_prefix_receipt_sha256"
            ],
            "final_left_closure_receipt_sha256": context[
                "final_left_closure_receipt_sha256"
            ],
            "final_left_support_union_sha256": context[
                "final_left_support_union_sha256"
            ],
            "k3_gate_receipt_sha256": context["k3_gate_receipt_sha256"],
            "reference_policy_receipt_sha256": context[
                "reference_policy_receipt_sha256"
            ],
            "temporal_tolerance_registry_receipt_sha256": context[
                "temporal_tolerance_registry_receipt_sha256"
            ],
            "onset_trigger_threshold_registry_receipt_sha256": context[
                "onset_trigger_threshold_registry_receipt_sha256"
            ],
            "producer_receipt_sha256": atom["producer_receipt_sha256"],
            "permission_receipt_sha256": atom["permission_receipt_sha256"],
        },
        "reason_codes": ordered,
    }
    body["binding_receipt_sha256"] = _canonical_sha256(body)
    return body


def _rank_counterfactual_interface(
    candidate: Mapping[str, Any], allowed_atom_ids: Sequence[str], ledger_receipt: str
) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    for atom_id in allowed_atom_ids:
        for kind in ("single_deletion", "single_insertion"):
            request = {
                "candidate_id": candidate["candidate_id"],
                "counterfactual_kind": kind,
                "atom_ids": [atom_id],
                "explicit_unknown_mask_required": True,
                "execution_status": "not_executed",
            }
            request["request_sha256"] = _canonical_sha256(request)
            requests.append(request)
    if allowed_atom_ids:
        request = {
            "candidate_id": candidate["candidate_id"],
            "counterfactual_kind": "complete_bound_set_group_deletion",
            "atom_ids": list(allowed_atom_ids),
            "explicit_unknown_mask_required": True,
            "execution_status": "not_executed",
        }
        request["request_sha256"] = _canonical_sha256(request)
        requests.append(request)
    return {
        "interface_version": "rank_counterfactual_request_interface_v1_5_1",
        "candidate_id": candidate["candidate_id"],
        "permission_locked_ledger_receipt_sha256": ledger_receipt,
        "requests": requests,
        "ranker_executed": False,
        "execution_state": "not_evaluable",
        "single_deletion_deltas": None,
        "single_insertion_deltas": None,
        "group_deletion_delta": None,
        "sufficiency_status": "not_proven",
        "minimality_status": "not_proven",
        "reason_codes": [
            "rank_counterfactual_interface_only",
            "no_frozen_real_ranker_or_threshold_registry_bound",
        ],
    }


def _candidate_attribution(
    candidate: Mapping[str, Any],
    atoms: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    ledger_receipt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_gate_reasons = _candidate_gate_reasons(candidate, context)
    audits = [_atom_attribution(candidate, atom, context) for atom in atoms]
    audits.sort(key=lambda row: row["measurement_atom_id"])
    allowed = [
        row["measurement_atom_id"]
        for row in audits
        if row["binding_state"] == "allowed"
    ]
    uncertain = [
        row["measurement_atom_id"]
        for row in audits
        if row["binding_state"] == "uncertain"
    ]
    excluded = [
        row["measurement_atom_id"]
        for row in audits
        if row["binding_state"] == "not_evaluable"
    ]
    compatible = [
        row
        for row in audits
        if "typed_unit_mismatch" not in row["reason_codes"]
        and "reference_family_mismatch" not in row["reason_codes"]
        and "reference_transform_receipt_mismatch" not in row["reason_codes"]
    ]
    hard_candidate_reasons = {
        "candidate_state_not_evaluable",
        "candidate_query_after_locked_prefix",
        "candidate_interval_outside_k3",
    }
    if any(reason in hard_candidate_reasons for reason in candidate_gate_reasons):
        state = "not_evaluable"
        decision_reasons = list(candidate_gate_reasons)
        if not atoms:
            decision_reasons.append("no_native_measurement_atoms_provided")
    elif allowed:
        state = "allowed"
        decision_reasons = ["eligible_present_native_trigger_atom_bound"]
    elif not atoms:
        state = "not_evaluable"
        decision_reasons = [
            *candidate_gate_reasons,
            "no_native_measurement_atoms_provided",
        ]
    elif not compatible:
        state = "uncertain"
        decision_reasons = [
            *candidate_gate_reasons,
            "no_compatible_native_trigger_atom",
        ]
    elif uncertain:
        state = "uncertain"
        decision_reasons = [
            *candidate_gate_reasons,
            "only_uncertain_native_trigger_support",
        ]
    elif any(
        "absent_with_opportunity_not_authorized_for_positive_trigger"
        in row["reason_codes"]
        for row in compatible
    ):
        state = "uncertain"
        decision_reasons = [
            *candidate_gate_reasons,
            "positive_native_trigger_not_established",
        ]
    else:
        state = "not_evaluable"
        decision_reasons = [
            *candidate_gate_reasons,
            "compatible_native_trigger_atoms_not_evaluable",
        ]
    body: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "source_slot_id": candidate["source_slot_id"],
        "candidate_content_sha256": candidate["candidate_content_sha256"],
        "candidate_state": candidate["candidate_state"],
        "recording_relative_earliest_interval_s": deepcopy(
            candidate["recording_relative_earliest_interval_s"]
        ),
        "typed_unit": deepcopy(candidate["typed_unit"]),
        "reference_family": candidate["reference_family"],
        "attribution_state": state,
        "structural_binding_allowed": state == "allowed",
        "positive_rank_contribution_authorized_by_this_module": False,
        "bound_present_trigger_atom_ids": allowed,
        "uncertainty_only_trigger_atom_ids": uncertain,
        "not_evaluable_or_incompatible_atom_ids": excluded,
        "atom_audit_roster": audits,
        "complete_atom_roster_retained": len(audits) == len(atoms),
        "decision_reason_codes": decision_reasons,
        "clinical_term_qualified": False,
        "report_promotion_authorized": False,
    }
    body["candidate_attribution_receipt_sha256"] = _canonical_sha256(body)
    rank_interface = _rank_counterfactual_interface(candidate, allowed, ledger_receipt)
    return body, rank_interface


def _used_receipts(
    context: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    atoms: Sequence[Mapping[str, Any]],
) -> list[str]:
    fields = (
        "candidate_generator_receipt_sha256",
        "native_operator_registry_receipt_sha256",
        "final_left_closure_receipt_sha256",
        "final_left_support_union_sha256",
        "locked_causal_prefix_receipt_sha256",
        "k3_gate_receipt_sha256",
        "reference_policy_receipt_sha256",
        "temporal_tolerance_registry_receipt_sha256",
        "onset_trigger_threshold_registry_receipt_sha256",
        "context_content_sha256",
    )
    result = {context[field] for field in fields}
    for candidate in candidates:
        result.add(candidate["candidate_generator_receipt_sha256"])
        result.add(candidate["reference_transform_receipt_sha256"])
        result.add(candidate["candidate_validation_receipt_sha256"])
    atom_fields = (
        "transform_receipt_sha256",
        "operator_parameter_receipt_sha256",
        "native_measurement_validation_receipt_sha256",
        "raw_dependency_receipt_sha256",
        "reference_transform_receipt_sha256",
        "qc_opportunity_receipt_sha256",
        "bandwidth_receipt_sha256",
        "effect_threshold_decision_receipt_sha256",
        "minimum_persistence_decision_receipt_sha256",
        "query_closure_receipt_sha256",
        "producer_receipt_sha256",
        "permission_receipt_sha256",
    )
    for atom in atoms:
        result.update(atom[field] for field in atom_fields)
    return sorted(result)


def _build_body(
    *,
    context: object,
    candidates: Sequence[object],
    measurement_atoms: Sequence[object],
    trusted_receipt_sha256s: Collection[str],
    trusted_candidate_content_sha256s: Collection[str],
    trusted_measurement_content_sha256s: Collection[str],
) -> dict[str, Any]:
    trusted_receipts = _validated_trust_set(
        trusted_receipt_sha256s, "trusted_receipt_sha256s"
    )
    trusted_candidates = _validated_trust_set(
        trusted_candidate_content_sha256s,
        "trusted_candidate_content_sha256s",
    )
    trusted_atoms = _validated_trust_set(
        trusted_measurement_content_sha256s,
        "trusted_measurement_content_sha256s",
        allow_empty=True,
    )
    validated_context = _validated_context(context, trusted_receipts)
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be an array")
    if not isinstance(measurement_atoms, Sequence) or isinstance(
        measurement_atoms, (str, bytes)
    ):
        raise TypeError("measurement_atoms must be an array")
    validated_candidates = [
        _validated_candidate(
            row,
            index=index,
            context=validated_context,
            trusted_receipts=trusted_receipts,
            trusted_content=trusted_candidates,
        )
        for index, row in enumerate(candidates)
    ]
    validated_atoms = [
        _validated_atom(
            row,
            index=index,
            context=validated_context,
            trusted_receipts=trusted_receipts,
            trusted_content=trusted_atoms,
        )
        for index, row in enumerate(measurement_atoms)
    ]
    candidate_ids = [row["candidate_id"] for row in validated_candidates]
    atom_ids = [row["measurement_atom_id"] for row in validated_atoms]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate roster must be non-empty with unique IDs")
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError("measurement atom IDs must be unique")
    validated_candidates.sort(key=lambda row: row["candidate_id"])
    validated_atoms.sort(key=lambda row: row["measurement_atom_id"])
    ledger_receipt = _canonical_sha256(
        {
            "recording_id": validated_context["recording_id"],
            "occurrence_id": validated_context["occurrence_id"],
            "context_content_sha256": validated_context["context_content_sha256"],
            "locked_causal_prefix_receipt_sha256": validated_context[
                "locked_causal_prefix_receipt_sha256"
            ],
            "candidate_content_sha256s": [
                row["candidate_content_sha256"] for row in validated_candidates
            ],
            "measurement_content_sha256s": [
                row["measurement_content_sha256"] for row in validated_atoms
            ],
        }
    )
    attributions: list[dict[str, Any]] = []
    rank_interfaces: list[dict[str, Any]] = []
    for candidate in validated_candidates:
        attribution, rank_interface = _candidate_attribution(
            candidate, validated_atoms, validated_context, ledger_receipt
        )
        attributions.append(attribution)
        rank_interfaces.append(rank_interface)
    used_receipts = _used_receipts(
        validated_context, validated_candidates, validated_atoms
    )
    trust_binding = {
        "used_trusted_receipt_sha256s": used_receipts,
        "used_trusted_candidate_content_sha256s": sorted(
            row["candidate_content_sha256"] for row in validated_candidates
        ),
        "used_trusted_measurement_content_sha256s": sorted(
            row["measurement_content_sha256"] for row in validated_atoms
        ),
    }
    trust_binding["trust_binding_receipt_sha256"] = _canonical_sha256(trust_binding)
    return {
        "schema_version": ONSET_TRIGGER_ATTRIBUTION_SCHEMA_VERSION,
        "method_id": ONSET_TRIGGER_ATTRIBUTION_METHOD_ID,
        "parent_findings_freeze_sha256": PARENT_FINDINGS_FREEZE_SHA256,
        "recording_id": validated_context["recording_id"],
        "occurrence_id": validated_context["occurrence_id"],
        "query_index": validated_context["query_index"],
        "permission_locked_ledger_receipt_sha256": ledger_receipt,
        "inputs": {
            "context": validated_context,
            "candidates": validated_candidates,
            "measurement_atoms": validated_atoms,
        },
        "input_trust_binding": trust_binding,
        "candidate_attributions": attributions,
        "rank_counterfactual_interfaces": rank_interfaces,
        "firewall": deepcopy(_CONTEXT_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
        "implementation_truth": {
            "content_binding_implemented": True,
            "synthetic_CPU_replay_only": True,
            "real_A1_native_atom_rollout_completed": False,
            "real_onset_trigger_threshold_registry_admitted": False,
            "rank_counterfactual_executed": False,
            "minimal_trigger_set_proven": False,
            "clinical_term_qualification_completed": False,
            "SOZ_performance_established": False,
        },
    }


def materialize_onset_trigger_attribution_v1_5_1(
    *,
    context: object,
    candidates: Sequence[object],
    measurement_atoms: Sequence[object],
    trusted_receipt_sha256s: Collection[str],
    trusted_candidate_content_sha256s: Collection[str],
    trusted_measurement_content_sha256s: Collection[str],
) -> dict[str, Any]:
    """Bind trusted native trigger atoms to trusted S07/S08 candidates.

    Trust collections are mandatory.  Syntactically valid hashes embedded in
    an input are not sufficient authority by themselves.
    """

    body = _build_body(
        context=context,
        candidates=candidates,
        measurement_atoms=measurement_atoms,
        trusted_receipt_sha256s=trusted_receipt_sha256s,
        trusted_candidate_content_sha256s=trusted_candidate_content_sha256s,
        trusted_measurement_content_sha256s=trusted_measurement_content_sha256s,
    )
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_onset_trigger_attribution_v1_5_1(
        body,
        trusted_receipt_sha256s=trusted_receipt_sha256s,
        trusted_candidate_content_sha256s=trusted_candidate_content_sha256s,
        trusted_measurement_content_sha256s=trusted_measurement_content_sha256s,
    )


def validate_onset_trigger_attribution_context_v1_5_1(
    context: object,
    *,
    trusted_receipt_sha256s: Collection[str],
) -> dict[str, Any]:
    """Validate one standalone attribution context against explicit trust.

    Producer-to-wire adapters need to validate the same locked-prefix and
    receipt authority as the complete attribution materializer before they
    can project a native measurement atom.  Exposing this narrow validator
    avoids copying (and eventually drifting from) the exact v1.5.1 context
    contract.  It grants no additional clinical, rank, or report permission.
    """

    trusted_receipts = _validated_trust_set(
        trusted_receipt_sha256s, "trusted_receipt_sha256s"
    )
    return _validated_context(context, trusted_receipts)


def validate_onset_trigger_measurement_atom_v1_5_1(
    measurement_atom: object,
    *,
    context: object,
    trusted_receipt_sha256s: Collection[str],
    trusted_measurement_content_sha256s: Collection[str],
) -> dict[str, Any]:
    """Validate one standalone S03--S06 atom with the frozen atom schema.

    This is an integration boundary, not an atom constructor.  Callers still
    must provide explicit trust for every referenced receipt and for the atom
    content hash.  A syntactically valid or self-invented hash is not accepted
    unless the caller has placed it in those trust collections.
    """

    trusted_receipts = _validated_trust_set(
        trusted_receipt_sha256s, "trusted_receipt_sha256s"
    )
    trusted_atoms = _validated_trust_set(
        trusted_measurement_content_sha256s,
        "trusted_measurement_content_sha256s",
    )
    validated_context = _validated_context(context, trusted_receipts)
    return _validated_atom(
        measurement_atom,
        index=0,
        context=validated_context,
        trusted_receipts=trusted_receipts,
        trusted_content=trusted_atoms,
    )


def validate_onset_trigger_attribution_v1_5_1(
    payload: object,
    *,
    trusted_receipt_sha256s: Collection[str],
    trusted_candidate_content_sha256s: Collection[str],
    trusted_measurement_content_sha256s: Collection[str],
) -> dict[str, Any]:
    """Fully replay one attribution receipt against explicit caller trust."""

    fields = {
        "schema_version",
        "method_id",
        "parent_findings_freeze_sha256",
        "recording_id",
        "occurrence_id",
        "query_index",
        "permission_locked_ledger_receipt_sha256",
        "inputs",
        "input_trust_binding",
        "candidate_attributions",
        "rank_counterfactual_interfaces",
        "firewall",
        "authorization",
        "implementation_truth",
        "receipt_sha256",
    }
    row = _exact_mapping(payload, fields, "attribution payload")
    receipt = _sha256(row["receipt_sha256"], "attribution receipt_sha256")
    if receipt != _self_hash(row, "receipt_sha256"):
        raise ValueError("attribution receipt hash does not replay")
    inputs = _exact_mapping(
        row["inputs"], {"context", "candidates", "measurement_atoms"}, "inputs"
    )
    expected = _build_body(
        context=inputs["context"],
        candidates=inputs["candidates"],
        measurement_atoms=inputs["measurement_atoms"],
        trusted_receipt_sha256s=trusted_receipt_sha256s,
        trusted_candidate_content_sha256s=trusted_candidate_content_sha256s,
        trusted_measurement_content_sha256s=trusted_measurement_content_sha256s,
    )
    observed_body = deepcopy(row)
    observed_body.pop("receipt_sha256")
    if _canonical_json(observed_body) != _canonical_json(expected):
        raise ValueError("attribution payload does not replay from its trusted inputs")
    return deepcopy(row)


__all__ = [
    "ONSET_TRIGGER_ATTRIBUTION_SCHEMA_VERSION",
    "ONSET_TRIGGER_ATTRIBUTION_CONTEXT_SCHEMA_VERSION",
    "ONSET_TRIGGER_ATTRIBUTION_METHOD_ID",
    "PARENT_FINDINGS_FREEZE_SHA256",
    "materialize_onset_trigger_attribution_v1_5_1",
    "validate_onset_trigger_attribution_context_v1_5_1",
    "validate_onset_trigger_measurement_atom_v1_5_1",
    "validate_onset_trigger_attribution_v1_5_1",
]
