"""Independent, EEG-only evaluation of atomic report claims.

This module evaluates a *structured realization* of a report against a frozen
EEG EvidenceGraph projection.  It deliberately does not read EDF files,
annotations, spreadsheets, physician labels, clinical text, or source-eval.
The caller must provide de-identified atomic claims and a closed stage ledger.

The evaluator addresses a different question from SOZ Top-1 accuracy and from
``validate_multievent_soz_report_payload``:

* Top-1 asks whether one ranked channel agrees with an external endpoint.
* the report validator proves that one claim graph is internally well formed;
* this evaluator asks whether every realized claim has one EEG-supported
  counterpart, whether every salient reference claim survived, and whether
  the derivation from supported premises used an allowed rule.

Matching is maximum-weight and one-to-one.  Duplicate predictions therefore
cannot repeatedly consume one reference fact.  Patient-macro aggregation and
patient bootstrap are implemented explicitly so records or seizures from one
patient never become independent statistical units.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import random
import re
from typing import Any, Iterable, Mapping, Sequence


CLAIM_FACTUALITY_CASE_SCHEMA_VERSION = "eeg_claim_factuality_case_v1"
CLAIM_FACTUALITY_EVALUATION_SCHEMA_VERSION = "eeg_claim_factuality_evaluation_v1"
CLAIM_FACTUALITY_POLICY_SCHEMA_VERSION = "eeg_claim_factuality_match_policy_v1"
EEG_CLAIM_GROUND_SCHEMA_VERSION = "eeg_claim_ground_v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_CLAIM_KINDS = {
    "observation",
    "event_inference",
    "mode_inference",
    "record_hypothesis",
    "evidence_relation",
}
_POLARITIES = {"affirmed", "negated"}
_NEGATION_SCOPES = {"none", "predicate", "object_or_value", "full_claim"}
_EPISTEMIC_STATUSES = {
    "measured",
    "model_candidate",
    "clinically_qualified",
    "not_evaluable",
    "research_ai_hypothesis",
    "risk_controlled_hypothesis",
    "technical_limited",
}
_ASSERTION_STATUSES = {
    "present",
    "absent_with_opportunity",
    "uncertain",
    "not_evaluable",
}
_LEGACY_EVIDENCE_ROLES = {
    "onset_support",
    "spread_support",
    "contradiction",
    "context_only",
}
_REPORT_GRAPH_V2_EVIDENCE_ROLES = {
    "ictal_pattern_qualification",
    "onset_time_support",
    "onset_topography_support",
    "course_or_spread_support",
    "counterevidence",
}
# The portable v1 case remains backward compatible with the original four
# roles, while report-graph-v2 callers retain the five source roles verbatim.
# In particular, callers must not collapse onset time and onset topography to
# the legacy ``onset_support`` label or relabel later course as onset evidence.
_EVIDENCE_ROLES = _LEGACY_EVIDENCE_ROLES | _REPORT_GRAPH_V2_EVIDENCE_ROLES
_INTRINSIC_EVIDENCE_ROLES = {
    "onset_eligible",
    "early_context",
    "later_involvement",
    "non_event_context",
    "limitation",
}
_VIEW_ROLES = {
    "canonical_physical_evidence",
    "onset_causal",
    "context_offline",
    "detector_navigation",
    "unknown",
}
_TIME_KINDS = {"none", "recording_interval", "delay_interval"}
_TIMEBASES = {
    "not_applicable",
    "recording_relative_seconds",
    "relative_delay_seconds",
}
_RELATION_PREDICATES = {
    "supports_claim",
    "contradicts_claim",
    "precedes_recruitment_of",
    "near_synchronous_with",
    "recruits_to",
}
_LOCAL_ENTITY_TYPES = {"finding", "evidence", "claim", "hypothesis"}
_EVIDENCE_FLOW_STAGES = (
    "detector_recovered",
    "adaptive_window_retained",
    "finding_emitted",
    "record_claim_retained",
    "rendered_claim_retained",
)

_COMPONENT_WEIGHTS = {
    "concept": 0.18,
    "subject": 0.08,
    "object": 0.18,
    "event_mode_attribution": 0.10,
    "time": 0.14,
    "polarity": 0.06,
    "negation_scope": 0.06,
    "assertion_status": 0.10,
    "epistemic_status": 0.10,
}

DEFAULT_CLAIM_FACTUALITY_POLICY: Mapping[str, Any] = {
    "schema_version": CLAIM_FACTUALITY_POLICY_SCHEMA_VERSION,
    "policy_id": "eeg_atomic_claim_hungarian_v1",
    "minimum_alignment_score": 0.45,
    "minimum_interval_iou": 0.50,
    "time_tolerance_seconds": 0.25,
    "measurement_absolute_tolerance": 1e-6,
    "measurement_relative_tolerance": 0.05,
    "component_weights": dict(_COMPONENT_WEIGHTS),
    "strict_support_requires_all_components": True,
    "local_entity_ids_are_not_semantic": True,
    "relations_require_strictly_supported_endpoints": True,
    "localizing_derivations_require_onset_support_leaves": True,
}

_CASE_KEYS = {
    "schema_version",
    "case_id",
    "patient_id",
    "record_id",
    "predicted_claims",
    "reference_claims",
    "derivations",
    "evidence_flow",
    "claim_boundary",
}
_EVALUATION_ARTIFACT_KEYS = {
    "schema_version",
    "status",
    "case_id",
    "patient_id",
    "record_id",
    "policy",
    "policy_sha256",
    "input_case_sha256",
    "claim_metrics",
    "eeg_claim_ground_metrics",
    "relation_metrics",
    "chain_metrics",
    "evidence_flow_metrics",
    "matches",
    "derivations",
    "unmatched_predicted_claim_ids",
    "unmatched_reference_claim_ids",
    "sufficient_statistics",
    "claim_boundary",
    "artifact_sha256",
}
_EVALUATION_SUFFICIENT_STATISTIC_KEYS = {
    "aligned_count",
    "assertion_status_exact_count",
    "chain_valid_weight",
    "chain_weight",
    "channel_entity_evaluable_count",
    "channel_entity_exact_count",
    "claim_ground_aligned_count",
    "claim_ground_channel_evaluable_count",
    "claim_ground_channel_jaccard_sum",
    "claim_ground_event_evaluable_count",
    "claim_ground_event_match_count",
    "claim_ground_mode_evaluable_count",
    "claim_ground_mode_match_count",
    "claim_ground_onset_authorized_count",
    "claim_ground_onset_claim_count",
    "claim_ground_onset_complete_binding_count",
    "claim_ground_region_evaluable_count",
    "claim_ground_region_jaccard_sum",
    "claim_ground_score_sum",
    "claim_ground_strict_count",
    "claim_ground_temporal_evaluable_count",
    "claim_ground_temporal_iou_sum",
    "critical_overstated_count",
    "critical_unsupported_count",
    "epistemic_exact_count",
    "evidence_binding_evaluable_count",
    "evidence_binding_exact_count",
    "evidence_role_evaluable_count",
    "evidence_role_exact_count",
    "flow_end_weight",
    "flow_total_weight",
    "overstatement_count",
    "predicted_count",
    "predicted_weight",
    "reference_count",
    "relation_predicted",
    "relation_reference",
    "relation_tp",
    "salient_supported_weight",
    "salient_weight",
    "strict_predicted_count",
    "strict_reference_count",
    "supported_weight",
    "time_evaluable_count",
    "time_exact_count",
}
_CLAIM_KEYS = {
    "claim_id",
    "claim_kind",
    "subject",
    "predicate",
    "object_or_value",
    "event_id",
    "mode_id",
    "time",
    "polarity",
    "negation_scope",
    "assertion_status",
    "epistemic_status",
    "evidence_ids",
    "evidence_roles",
    "relation_endpoints",
    "salient",
    "salience_weight",
    "severity_weight",
    "critical",
    # Optional in v1 cases.  When present, this closes every cited evidence ID
    # to the time-direction fields already emitted by event_eeg_findings_v2.
    # Legacy cases remain accepted and are reported as role-only grounding.
    "evidence_temporal_bindings",
}
_ENTITY_KEYS = {"type", "id"}
_OBJECT_KEYS = {"entities", "measurements", "code"}
_MEASUREMENT_KEYS = {"name", "value", "unit"}
_TIME_KEYS = {
    "kind",
    "timebase",
    "lower",
    "upper",
    "left_censored",
    "right_censored",
}
_RELATION_ENDPOINT_KEYS = {"source_claim_id", "target_claim_id"}
_EVIDENCE_TEMPORAL_BINDING_KEYS = {
    "evidence_id",
    "evidence_role",
    "intrinsic_evidence_role",
    "view_role",
    "future_sample_access",
    "onset_evidence_authorized",
    "onset_support_eligible",
}
_DERIVATION_KEYS = {
    "derivation_id",
    "conclusion_claim_id",
    "premise_claim_ids",
    "rule_id",
    "weight",
}
_EVIDENCE_FLOW_KEYS = {
    "evidence_id",
    "weight",
    *_EVIDENCE_FLOW_STAGES,
}
_CLAIM_BOUNDARY = {
    "eeg_signal_claims_only": True,
    "private_data_loaded_by_evaluator": False,
    "excel_loaded_by_evaluator": False,
    "edf_annotations_loaded_by_evaluator": False,
    "doctor_labels_loaded_by_evaluator": False,
    "clinical_text_loaded_by_evaluator": False,
    "source_eval_loaded_by_evaluator": False,
}

_ALLOWED_DERIVATION_RULES: Mapping[str, Mapping[str, Any]] = {
    "event_observation_to_event_hypothesis_v1": {
        "conclusion_kinds": {"event_inference"},
        "premise_kinds": {"observation", "evidence_relation"},
        "scope": "same_event",
        "minimum_premises": 1,
    },
    "event_hypothesis_to_mode_hypothesis_v1": {
        "conclusion_kinds": {"mode_inference"},
        "premise_kinds": {"event_inference"},
        "scope": "same_mode",
        "minimum_premises": 1,
    },
    # The record claim graph may preserve direct support edges from atomic
    # event observations to a mode summary.  This is not equivalent to
    # inventing an unpersisted event-level hypothesis during evaluation: the
    # source-bound materializer only emits this rule when those exact support
    # edges exist in the frozen graph and both endpoints were rendered.
    "event_observation_to_mode_hypothesis_v1": {
        "conclusion_kinds": {"mode_inference"},
        "premise_kinds": {"observation", "event_inference"},
        "scope": "same_mode",
        "minimum_premises": 1,
    },
    "mode_hypothesis_to_record_hypothesis_v1": {
        "conclusion_kinds": {"record_hypothesis"},
        "premise_kinds": {"mode_inference"},
        "scope": "record",
        "minimum_premises": 1,
    },
    # Likewise, some frozen graphs bind onset observations directly to a
    # focal record hypothesis.  The temporal-permission gate below still
    # requires every localizing leaf to be onset-causal and future-free.
    "onset_evidence_to_record_hypothesis_v1": {
        "conclusion_kinds": {"record_hypothesis"},
        "conclusion_predicates": {
            "record_primary_soz_hypothesis",
            "record_alternative_soz_hypothesis",
        },
        "premise_kinds": {"observation", "event_inference", "mode_inference"},
        "scope": "record",
        "minimum_premises": 1,
    },
    "bilateral_synchrony_to_generalized_record_v1": {
        "conclusion_kinds": {"record_hypothesis"},
        "conclusion_predicates": {"record_has_generalized_synchronous_onset"},
        "premise_kinds": {"observation", "event_inference", "mode_inference"},
        "required_leaf_predicates": {"bilateral_synchronous_evolution_observed"},
        "scope": "record",
        "minimum_premises": 1,
    },
    "limitation_to_nonlocalizable_record_v1": {
        "conclusion_kinds": {"record_hypothesis"},
        "conclusion_predicates": {
            "record_onset_nonlocalizable",
            "record_technical_limited",
        },
        "premise_kinds": {"observation", "event_inference", "mode_inference"},
        "required_leaf_predicates": {
            "no_stable_focal_lead_observed",
            "artifact_limits_interpretation",
            "record_signal_technically_limited",
        },
        "scope": "record",
        "minimum_premises": 1,
    },
    "multiple_modes_to_record_hypothesis_v1": {
        "conclusion_kinds": {"record_hypothesis"},
        "conclusion_predicates": {"record_has_multiple_onset_modes"},
        "premise_kinds": {"mode_inference"},
        "scope": "distinct_modes",
        "minimum_premises": 2,
    },
}

_LOCALIZING_PREDICATES = {
    "event_supports_soz_candidate",
    "mode_supports_soz_candidate",
    "record_primary_soz_hypothesis",
    "record_alternative_soz_hypothesis",
}

# Positive onset/SOZ statements require causal, future-free evidence.  A
# non-localizable limitation is deliberately absent: it may be supported by a
# limitation or contradiction rather than by a positive onset observation.
_POSITIVE_ONSET_OR_SOZ_PREDICATES = _LOCALIZING_PREDICATES | {
    "earliest_sustained_change_maximal_at",
    "event_has_onset_phenotype",
    "mode_repeats_onset_pattern",
    "record_has_generalized_synchronous_onset",
    "record_has_multiple_onset_modes",
}

_CHANNEL_ENTITY_TYPES = {"electrode", "channel", "bipolar_derivation"}

_EPISTEMIC_RANKS: Mapping[str, tuple[str, int]] = {
    "not_evaluable": ("observation", 0),
    "model_candidate": ("observation", 1),
    "measured": ("observation", 2),
    "clinically_qualified": ("observation", 3),
    "research_ai_hypothesis": ("inference", 1),
    "risk_controlled_hypothesis": ("inference", 2),
    # Technical limitation is a distinct state, not a low-confidence
    # physiological assertion; only exact agreement is considered compatible.
    "technical_limited": ("technical", 0),
}


def _strict_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    actual = set(value)
    missing = keys - actual
    extra = actual - keys
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{context} must be an opaque identifier")
    return value


def _sha256_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _optional_identifier(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, context)


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _finite_nonnegative(
    value: object, context: str, *, positive: bool = False
) -> float:
    result = _finite_number(value, context)
    if result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be finite and {qualifier}")
    return result


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _ratio(numerator: float, denominator: float) -> float | str:
    if denominator <= 0:
        return "not_available"
    return round(float(numerator) / float(denominator), 12)


def _f1(precision: float | str, recall: float | str) -> float | str:
    if isinstance(precision, str) or isinstance(recall, str):
        return "not_available"
    if precision + recall <= 0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 12)


def validate_claim_factuality_policy(value: object | None = None) -> dict[str, Any]:
    """Validate a closed matching policy and return a defensive copy."""

    raw = deepcopy(DEFAULT_CLAIM_FACTUALITY_POLICY if value is None else value)
    keys = set(DEFAULT_CLAIM_FACTUALITY_POLICY)
    data = _strict_object(raw, keys, "claim factuality policy")
    if data["schema_version"] != CLAIM_FACTUALITY_POLICY_SCHEMA_VERSION:
        raise ValueError("claim factuality policy schema_version mismatch")
    _identifier(data["policy_id"], "claim factuality policy.policy_id")
    for name in (
        "minimum_alignment_score",
        "minimum_interval_iou",
        "time_tolerance_seconds",
        "measurement_absolute_tolerance",
        "measurement_relative_tolerance",
    ):
        data[name] = _finite_nonnegative(data[name], f"claim factuality policy.{name}")
    if not 0 <= data["minimum_alignment_score"] <= 1:
        raise ValueError("minimum_alignment_score must be in [0, 1]")
    if not 0 <= data["minimum_interval_iou"] <= 1:
        raise ValueError("minimum_interval_iou must be in [0, 1]")
    weights = _strict_object(
        data["component_weights"],
        set(_COMPONENT_WEIGHTS),
        "claim factuality policy.component_weights",
    )
    data["component_weights"] = {
        name: _finite_nonnegative(weight, f"component_weights.{name}", positive=True)
        for name, weight in weights.items()
    }
    if not math.isclose(sum(data["component_weights"].values()), 1.0, abs_tol=1e-9):
        raise ValueError("claim factuality component weights must sum to 1")
    for name in (
        "strict_support_requires_all_components",
        "local_entity_ids_are_not_semantic",
        "relations_require_strictly_supported_endpoints",
        "localizing_derivations_require_onset_support_leaves",
    ):
        if type(data[name]) is not bool:
            raise TypeError(f"claim factuality policy.{name} must be boolean")
        if data[name] is not True:
            raise ValueError(
                f"claim factuality policy.{name} is a fixed fail-closed invariant"
            )
    return data


def _validate_entity(value: object, context: str) -> dict[str, str]:
    data = _strict_object(value, _ENTITY_KEYS, context)
    entity_type = _identifier(data["type"], f"{context}.type")
    entity_id = _identifier(data["id"], f"{context}.id")
    return {"type": entity_type, "id": entity_id}


def _validate_time(value: object, context: str) -> dict[str, Any]:
    data = _strict_object(value, _TIME_KEYS, context)
    kind = str(data["kind"])
    timebase = str(data["timebase"])
    if kind not in _TIME_KINDS or timebase not in _TIMEBASES:
        raise ValueError(f"{context} has invalid kind/timebase")
    if (
        type(data["left_censored"]) is not bool
        or type(data["right_censored"]) is not bool
    ):
        raise TypeError(f"{context} censoring flags must be boolean")
    lower = data["lower"]
    upper = data["upper"]
    if kind == "none":
        if (
            timebase != "not_applicable"
            or lower is not None
            or upper is not None
            or data["left_censored"]
            or data["right_censored"]
        ):
            raise ValueError(f"{context} kind=none cannot carry temporal values")
    else:
        expected_timebase = {
            "recording_interval": "recording_relative_seconds",
            "delay_interval": "relative_delay_seconds",
        }[kind]
        if timebase != expected_timebase:
            raise ValueError(f"{context} kind/timebase mismatch")
        if lower is None or upper is None:
            raise ValueError(f"{context} interval requires lower and upper")
        lower = (
            _finite_nonnegative(lower, f"{context}.lower")
            if kind == "recording_interval"
            else _finite_number(lower, f"{context}.lower")
        )
        upper = (
            _finite_nonnegative(upper, f"{context}.upper")
            if kind == "recording_interval"
            else _finite_number(upper, f"{context}.upper")
        )
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"{context} interval is invalid")
    data["lower"] = lower
    data["upper"] = upper
    return data


def _validate_claim(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    # ``evidence_temporal_bindings`` was added as an optional closure field to
    # the v1 evaluator.  Filling only this known key preserves exact rejection
    # of arbitrary extras while keeping all previously valid v1 payloads valid.
    raw_claim = {str(key): deepcopy(item) for key, item in value.items()}
    raw_claim.setdefault("evidence_temporal_bindings", None)
    data = _strict_object(raw_claim, _CLAIM_KEYS, context)
    data["claim_id"] = _identifier(data["claim_id"], f"{context}.claim_id")
    if data["claim_kind"] not in _CLAIM_KINDS:
        raise ValueError(f"{context}.claim_kind is invalid")
    data["subject"] = _validate_entity(data["subject"], f"{context}.subject")
    data["predicate"] = _identifier(data["predicate"], f"{context}.predicate")
    obj = _strict_object(
        data["object_or_value"], _OBJECT_KEYS, f"{context}.object_or_value"
    )
    if not isinstance(obj["entities"], list):
        raise TypeError(f"{context}.object_or_value.entities must be a list")
    entities = [
        _validate_entity(item, f"{context}.object_or_value.entities[{index}]")
        for index, item in enumerate(obj["entities"])
    ]
    entity_keys = [(item["type"], item["id"]) for item in entities]
    if len(entity_keys) != len(set(entity_keys)):
        raise ValueError(f"{context}.object_or_value.entities contains duplicates")
    if not isinstance(obj["measurements"], list):
        raise TypeError(f"{context}.object_or_value.measurements must be a list")
    measurements: list[dict[str, Any]] = []
    measurement_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(obj["measurements"]):
        row = _strict_object(
            item, _MEASUREMENT_KEYS, f"{context}.measurements[{index}]"
        )
        name = _identifier(row["name"], f"{context}.measurements[{index}].name")
        unit = row["unit"]
        if not isinstance(unit, str) or not unit or len(unit) > 64:
            raise ValueError(f"{context}.measurements[{index}].unit is invalid")
        value_float = _finite_number(
            row["value"], f"{context}.measurements[{index}].value"
        )
        key = (name, unit)
        if key in measurement_keys:
            raise ValueError(
                f"{context}.object_or_value.measurements contains duplicates"
            )
        measurement_keys.add(key)
        measurements.append({"name": name, "value": value_float, "unit": unit})
    code = obj["code"]
    if code is not None:
        code = _identifier(code, f"{context}.object_or_value.code")
    data["object_or_value"] = {
        "entities": entities,
        "measurements": measurements,
        "code": code,
    }
    data["event_id"] = _optional_identifier(data["event_id"], f"{context}.event_id")
    data["mode_id"] = _optional_identifier(data["mode_id"], f"{context}.mode_id")
    data["time"] = _validate_time(data["time"], f"{context}.time")
    if data["polarity"] not in _POLARITIES:
        raise ValueError(f"{context}.polarity is invalid")
    if data["negation_scope"] not in _NEGATION_SCOPES:
        raise ValueError(f"{context}.negation_scope is invalid")
    if (data["polarity"] == "affirmed") != (data["negation_scope"] == "none"):
        raise ValueError(f"{context} polarity and negation_scope conflict")
    if data["epistemic_status"] not in _EPISTEMIC_STATUSES:
        raise ValueError(f"{context}.epistemic_status is invalid")
    if data["assertion_status"] not in _ASSERTION_STATUSES:
        raise ValueError(f"{context}.assertion_status is invalid")
    if (data["assertion_status"] == "not_evaluable") != (
        data["epistemic_status"] == "not_evaluable"
    ):
        raise ValueError(
            f"{context} assertion_status and not_evaluable epistemic status conflict"
        )
    for key, allowed in (("evidence_ids", None), ("evidence_roles", _EVIDENCE_ROLES)):
        raw_values = data[key]
        if not isinstance(raw_values, list):
            raise TypeError(f"{context}.{key} must be a list")
        values = [
            _identifier(item, f"{context}.{key}[{index}]")
            for index, item in enumerate(raw_values)
        ]
        if len(values) != len(set(values)):
            raise ValueError(f"{context}.{key} contains duplicates")
        if allowed is not None and not set(values).issubset(allowed):
            raise ValueError(f"{context}.{key} contains an invalid role")
        data[key] = values
    temporal_bindings = data["evidence_temporal_bindings"]
    if temporal_bindings is not None:
        if not isinstance(temporal_bindings, list):
            raise TypeError(f"{context}.evidence_temporal_bindings must be a list")
        validated_bindings: list[dict[str, Any]] = []
        binding_ids: set[str] = set()
        binding_roles: set[str] = set()
        for index, item in enumerate(temporal_bindings):
            binding_context = f"{context}.evidence_temporal_bindings[{index}]"
            row = _strict_object(item, _EVIDENCE_TEMPORAL_BINDING_KEYS, binding_context)
            row["evidence_id"] = _identifier(
                row["evidence_id"], f"{binding_context}.evidence_id"
            )
            if row["evidence_id"] in binding_ids:
                raise ValueError(
                    f"{context}.evidence_temporal_bindings contains duplicate evidence IDs"
                )
            binding_ids.add(row["evidence_id"])
            if row["evidence_role"] not in _EVIDENCE_ROLES:
                raise ValueError(f"{binding_context}.evidence_role is invalid")
            binding_roles.add(str(row["evidence_role"]))
            if row["intrinsic_evidence_role"] not in _INTRINSIC_EVIDENCE_ROLES:
                raise ValueError(
                    f"{binding_context}.intrinsic_evidence_role is invalid"
                )
            if row["view_role"] not in _VIEW_ROLES:
                raise ValueError(f"{binding_context}.view_role is invalid")
            for name in (
                "future_sample_access",
                "onset_evidence_authorized",
                "onset_support_eligible",
            ):
                if type(row[name]) is not bool:
                    raise TypeError(f"{binding_context}.{name} must be boolean")
            validated_bindings.append(row)
        if binding_ids != set(data["evidence_ids"]):
            raise ValueError(
                f"{context}.evidence_temporal_bindings must close every evidence_id exactly once"
            )
        if binding_roles != set(data["evidence_roles"]):
            raise ValueError(
                f"{context}.evidence_temporal_bindings roles must equal evidence_roles"
            )
        data["evidence_temporal_bindings"] = sorted(
            validated_bindings, key=lambda row: str(row["evidence_id"])
        )
    relation = data["relation_endpoints"]
    is_relation = (
        data["claim_kind"] == "evidence_relation"
        or data["predicate"] in _RELATION_PREDICATES
    )
    if relation is None:
        if is_relation:
            raise ValueError(f"{context} relation claim requires relation_endpoints")
    else:
        relation = _strict_object(
            relation, _RELATION_ENDPOINT_KEYS, f"{context}.relation_endpoints"
        )
        relation = {
            key: _identifier(item, f"{context}.relation_endpoints.{key}")
            for key, item in relation.items()
        }
        if relation["source_claim_id"] == relation["target_claim_id"]:
            raise ValueError(f"{context} relation cannot be a self-edge")
        if not is_relation:
            raise ValueError(
                f"{context} non-relation claim cannot carry relation_endpoints"
            )
    data["relation_endpoints"] = relation
    if type(data["salient"]) is not bool or type(data["critical"]) is not bool:
        raise TypeError(f"{context} salient and critical must be boolean")
    data["salience_weight"] = _finite_nonnegative(
        data["salience_weight"], f"{context}.salience_weight"
    )
    data["severity_weight"] = _finite_nonnegative(
        data["severity_weight"], f"{context}.severity_weight", positive=True
    )
    if data["salient"] != (data["salience_weight"] > 0):
        raise ValueError(f"{context} salient must equal salience_weight>0")
    return data


def _validate_claim_set(value: object, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    rows = [
        _validate_claim(item, f"{context}[{index}]") for index, item in enumerate(value)
    ]
    ids = [str(row["claim_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context} contains duplicate claim IDs")
    known = set(ids)
    for row in rows:
        relation = row["relation_endpoints"]
        if relation is not None:
            endpoints = {relation["source_claim_id"], relation["target_claim_id"]}
            if not endpoints.issubset(known):
                raise ValueError(f"{context} relation references an unknown claim")
            by_id = {str(item["claim_id"]): item for item in rows}
            if any(by_id[item]["relation_endpoints"] is not None for item in endpoints):
                raise ValueError(
                    f"{context} relation endpoints must be non-relation claims"
                )
    return rows


def validate_claim_factuality_case(value: object) -> dict[str, Any]:
    """Validate a de-identified, no-I/O factuality evaluation case."""

    data = _strict_object(value, _CASE_KEYS, "claim factuality case")
    if data["schema_version"] != CLAIM_FACTUALITY_CASE_SCHEMA_VERSION:
        raise ValueError("claim factuality case schema_version mismatch")
    for name in ("case_id", "patient_id", "record_id"):
        data[name] = _identifier(data[name], f"claim factuality case.{name}")
    data["predicted_claims"] = _validate_claim_set(
        data["predicted_claims"], "claim factuality case.predicted_claims"
    )
    data["reference_claims"] = _validate_claim_set(
        data["reference_claims"], "claim factuality case.reference_claims"
    )
    predicted_ids = {str(row["claim_id"]) for row in data["predicted_claims"]}
    if not isinstance(data["derivations"], list):
        raise TypeError("claim factuality case.derivations must be a list")
    derivations: list[dict[str, Any]] = []
    derivation_ids: set[str] = set()
    conclusions: set[str] = set()
    for index, item in enumerate(data["derivations"]):
        context = f"claim factuality case.derivations[{index}]"
        row = _strict_object(item, _DERIVATION_KEYS, context)
        row["derivation_id"] = _identifier(
            row["derivation_id"], f"{context}.derivation_id"
        )
        if row["derivation_id"] in derivation_ids:
            raise ValueError("claim factuality case contains duplicate derivation IDs")
        derivation_ids.add(row["derivation_id"])
        row["conclusion_claim_id"] = _identifier(
            row["conclusion_claim_id"], f"{context}.conclusion_claim_id"
        )
        if row["conclusion_claim_id"] not in predicted_ids:
            raise ValueError(
                f"{context} conclusion references an unknown predicted claim"
            )
        if row["conclusion_claim_id"] in conclusions:
            raise ValueError(
                "one predicted conclusion cannot have multiple derivations"
            )
        conclusions.add(row["conclusion_claim_id"])
        if (
            not isinstance(row["premise_claim_ids"], list)
            or not row["premise_claim_ids"]
        ):
            raise ValueError(f"{context}.premise_claim_ids must be a non-empty list")
        premises = [
            _identifier(value, f"{context}.premise_claim_ids[{position}]")
            for position, value in enumerate(row["premise_claim_ids"])
        ]
        if len(premises) != len(set(premises)) or not set(premises).issubset(
            predicted_ids
        ):
            raise ValueError(f"{context}.premise_claim_ids are duplicate or unknown")
        if row["conclusion_claim_id"] in premises:
            raise ValueError(f"{context} conclusion cannot be its own premise")
        row["premise_claim_ids"] = premises
        row["rule_id"] = _identifier(row["rule_id"], f"{context}.rule_id")
        row["weight"] = _finite_nonnegative(
            row["weight"], f"{context}.weight", positive=True
        )
        derivations.append(row)
    # A derivation DAG is required even when all individual references close.
    edges = {
        str(row["conclusion_claim_id"]): set(
            str(item) for item in row["premise_claim_ids"]
        )
        for row in derivations
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("claim factuality derivation graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in edges.get(node, set()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in edges:
        visit(node)
    data["derivations"] = derivations

    if not isinstance(data["evidence_flow"], list):
        raise TypeError("claim factuality case.evidence_flow must be a list")
    flow: list[dict[str, Any]] = []
    flow_ids: set[str] = set()
    for index, item in enumerate(data["evidence_flow"]):
        context = f"claim factuality case.evidence_flow[{index}]"
        row = _strict_object(item, _EVIDENCE_FLOW_KEYS, context)
        row["evidence_id"] = _identifier(row["evidence_id"], f"{context}.evidence_id")
        if row["evidence_id"] in flow_ids:
            raise ValueError(
                "claim factuality case contains duplicate evidence-flow IDs"
            )
        flow_ids.add(row["evidence_id"])
        row["weight"] = _finite_nonnegative(
            row["weight"], f"{context}.weight", positive=True
        )
        stage_values: list[bool] = []
        for stage in _EVIDENCE_FLOW_STAGES:
            if type(row[stage]) is not bool:
                raise TypeError(f"{context}.{stage} must be boolean")
            stage_values.append(row[stage])
        if any(
            later and not earlier
            for earlier, later in zip(stage_values, stage_values[1:])
        ):
            raise ValueError(
                f"{context} evidence flow must be monotonically non-increasing"
            )
        flow.append(row)
    data["evidence_flow"] = flow
    boundary = _strict_object(
        data["claim_boundary"],
        set(_CLAIM_BOUNDARY),
        "claim factuality case.claim_boundary",
    )
    for key, expected in _CLAIM_BOUNDARY.items():
        if boundary[key] is not expected:
            raise ValueError(
                f"claim factuality case.claim_boundary.{key} must be {expected}"
            )
    data["claim_boundary"] = dict(_CLAIM_BOUNDARY)
    return data


def _semantic_entity(entity: Mapping[str, Any]) -> tuple[str, str]:
    entity_type = str(entity["type"])
    entity_id = "<local>" if entity_type in _LOCAL_ENTITY_TYPES else str(entity["id"])
    return entity_type, entity_id


def _set_f1(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left.intersection(right))
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _measurement_comparison(
    predicted: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[float, bool]:
    pred = {
        (str(row["name"]), str(row["unit"])): float(row["value"]) for row in predicted
    }
    ref = {
        (str(row["name"]), str(row["unit"])): float(row["value"]) for row in reference
    }
    if not pred and not ref:
        return 1.0, True
    keys = set(pred).union(ref)
    if not keys:
        return 1.0, True
    matched = 0
    for key in keys:
        if key not in pred or key not in ref:
            continue
        tolerance = max(
            float(policy["measurement_absolute_tolerance"]),
            float(policy["measurement_relative_tolerance"])
            * max(abs(ref[key]), abs(pred[key])),
        )
        if abs(pred[key] - ref[key]) <= tolerance:
            matched += 1
    score = matched / len(keys)
    return score, matched == len(keys)


def _interval_comparison(
    predicted: Mapping[str, Any],
    reference: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[float, bool, dict[str, float | str]]:
    if (
        predicted["kind"] != reference["kind"]
        or predicted["timebase"] != reference["timebase"]
    ):
        return (
            0.0,
            False,
            {
                "interval_iou": "not_available",
                "lower_error_seconds": "not_available",
                "upper_error_seconds": "not_available",
            },
        )
    if predicted["kind"] == "none":
        exact = (
            predicted["left_censored"] == reference["left_censored"]
            and predicted["right_censored"] == reference["right_censored"]
        )
        return (
            (1.0 if exact else 0.0),
            exact,
            {
                "interval_iou": "not_available",
                "lower_error_seconds": "not_available",
                "upper_error_seconds": "not_available",
            },
        )
    pred_lower = float(predicted["lower"])
    pred_upper = float(predicted["upper"])
    ref_lower = float(reference["lower"])
    ref_upper = float(reference["upper"])
    intersection = max(0.0, min(pred_upper, ref_upper) - max(pred_lower, ref_lower))
    union = max(pred_upper, ref_upper) - min(pred_lower, ref_lower)
    if union == 0:
        iou = 1.0
    elif (
        intersection == 0
        and pred_lower == pred_upper
        and ref_lower <= pred_lower <= ref_upper
    ):
        # A point estimate inside a reference uncertainty interval is a valid
        # temporal localization, but receives no inflated IoU credit.
        iou = 0.0
    else:
        iou = intersection / union
    lower_error = pred_lower - ref_lower
    upper_error = pred_upper - ref_upper
    tolerance = float(policy["time_tolerance_seconds"])
    endpoint_match = abs(lower_error) <= tolerance and abs(upper_error) <= tolerance
    point_inside_reference = (
        pred_lower == pred_upper and ref_lower <= pred_lower <= ref_upper
    )
    censor_match = (
        predicted["left_censored"] == reference["left_censored"]
        and predicted["right_censored"] == reference["right_censored"]
    )
    compatible = censor_match and (
        endpoint_match
        or point_inside_reference
        or iou >= float(policy["minimum_interval_iou"])
    )
    score = (
        iou if iou > 0 else (1.0 if endpoint_match or point_inside_reference else 0.0)
    )
    if not censor_match:
        score *= 0.5
    return (
        round(score, 12),
        compatible,
        {
            "interval_iou": round(iou, 12),
            "lower_error_seconds": round(lower_error, 12),
            "upper_error_seconds": round(upper_error, 12),
        },
    )


def _claim_entity_ids(claim: Mapping[str, Any], entity_types: set[str]) -> set[str]:
    return {
        str(item["id"])
        for item in [claim["subject"], *claim["object_or_value"]["entities"]]
        if str(item["type"]) in entity_types
    }


def _set_jaccard(left: set[str], right: set[str]) -> float | str:
    if not left and not right:
        return "not_available"
    union = left.union(right)
    return round(len(left.intersection(right)) / len(union), 12)


def _requires_positive_onset_or_soz_authorization(
    claim: Mapping[str, Any],
) -> bool:
    return (
        str(claim["predicate"]) in _POSITIVE_ONSET_OR_SOZ_PREDICATES
        and claim["polarity"] == "affirmed"
        and claim["assertion_status"] in {"present", "uncertain"}
    )


def _evidence_role_vocabulary(roles: set[str]) -> str:
    legacy = bool(roles.intersection(_LEGACY_EVIDENCE_ROLES))
    report_graph_v2 = bool(roles.intersection(_REPORT_GRAPH_V2_EVIDENCE_ROLES))
    if legacy and report_graph_v2:
        return "mixed"
    if report_graph_v2:
        return "report_graph_v2_five_role"
    return "legacy_four_role"


def _requires_onset_topography_authorization(
    claim: Mapping[str, Any],
) -> bool:
    return (
        _requires_positive_onset_or_soz_authorization(claim)
        and str(claim["predicate"]) in _LOCALIZING_PREDICATES
    )


def _temporal_evidence_authorization(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate whether one claim uses evidence in an authorized time role.

    Legacy v1 claims only contain claim-level ``evidence_roles``.  They remain
    executable, but the result explicitly records that future-sample and view
    provenance were not auditable.  When bindings are supplied, every cited
    evidence ID has already been closed by validation and positive onset/SOZ
    support must be onset-causal, future-free, authorized, and eligible.
    """

    requires_onset = _requires_positive_onset_or_soz_authorization(claim)
    requires_topography = _requires_onset_topography_authorization(claim)
    roles = set(str(item) for item in claim["evidence_roles"])
    vocabulary = _evidence_role_vocabulary(roles)
    reasons: list[str] = []
    if requires_onset:
        if vocabulary == "mixed":
            reasons.append("mixed_legacy_and_five_role_evidence_semantics")
        elif vocabulary == "report_graph_v2_five_role":
            if "onset_time_support" not in roles:
                reasons.append("positive_onset_or_soz_claim_lacks_onset_time_support")
            if requires_topography and "onset_topography_support" not in roles:
                reasons.append("localizing_claim_lacks_onset_topography_support")
            if "course_or_spread_support" in roles:
                reasons.append(
                    "later_course_or_spread_cannot_support_positive_onset_or_soz"
                )
            if "counterevidence" in roles:
                reasons.append("counterevidence_cannot_support_positive_onset_or_soz")
        else:
            if "onset_support" not in roles:
                reasons.append("positive_onset_or_soz_claim_lacks_onset_support")
            if "spread_support" in roles:
                reasons.append("later_spread_cannot_support_positive_onset_or_soz")
            if "context_only" in roles:
                reasons.append("context_only_cannot_support_positive_onset_or_soz")

    bindings = claim.get("evidence_temporal_bindings")
    if bindings is None:
        return {
            "requires_positive_onset_or_soz_authorization": requires_onset,
            "authorized": not reasons,
            "authorization_complete": False,
            "authorization_basis": "legacy_claim_evidence_role_only",
            "invalid_reason_codes": sorted(set(reasons)),
        }

    positive_support_roles = (
        {"onset_time_support", "onset_topography_support"}
        if vocabulary == "report_graph_v2_five_role"
        else {"onset_support"}
    )
    onset_support_rows = [
        row for row in bindings if row["evidence_role"] in positive_support_roles
    ]
    if requires_onset:
        for row in onset_support_rows:
            role = str(row["evidence_role"])
            reason_prefix = role
            if row["intrinsic_evidence_role"] != "onset_eligible":
                reasons.append(f"{reason_prefix}_not_intrinsically_onset_eligible")
            if row["view_role"] != "onset_causal":
                reasons.append(f"{reason_prefix}_not_from_onset_causal_view")
            if row["view_role"] == "context_offline":
                reasons.append("context_offline_cannot_support_positive_onset_or_soz")
            if row["future_sample_access"]:
                reasons.append("future_dependent_evidence_cannot_support_onset")
            if not row["onset_evidence_authorized"]:
                reasons.append("onset_evidence_not_authorized_by_source_view")
            if not row["onset_support_eligible"]:
                reasons.append("raw_dependency_not_onset_support_eligible")
    return {
        "requires_positive_onset_or_soz_authorization": requires_onset,
        "authorized": not reasons,
        "authorization_complete": True,
        "authorization_basis": "closed_per_evidence_temporal_bindings",
        "invalid_reason_codes": sorted(set(reasons)),
    }


def _strict_temporal_evidence_authorized(claim: Mapping[str, Any]) -> bool:
    authorization = _temporal_evidence_authorization(claim)
    return bool(authorization["authorized"]) and (
        not authorization["requires_positive_onset_or_soz_authorization"]
        or bool(authorization["authorization_complete"])
    )


def _positive_onset_leaf_has_required_roles(
    claim: Mapping[str, Any], *, require_topography: bool
) -> bool:
    roles = set(str(item) for item in claim["evidence_roles"])
    vocabulary = _evidence_role_vocabulary(roles)
    if vocabulary == "legacy_four_role":
        return "onset_support" in roles
    if vocabulary != "report_graph_v2_five_role":
        return False
    return "onset_time_support" in roles and (
        not require_topography or "onset_topography_support" in roles
    )


def _strict_positive_onset_leaf_authorized(
    claim: Mapping[str, Any], *, require_topography: bool = False
) -> bool:
    """Require causal, future-free provenance when a leaf supports onset.

    A descriptive leaf predicate may not itself be an onset claim.  Once that
    leaf is used to derive a positive onset/SOZ conclusion, however, every
    cited evidence item must satisfy the stronger onset-support contract.
    """

    roles = set(str(item) for item in claim["evidence_roles"])
    vocabulary = _evidence_role_vocabulary(roles)
    bindings = claim.get("evidence_temporal_bindings")
    if not bindings:
        return False
    if not _positive_onset_leaf_has_required_roles(
        claim, require_topography=require_topography
    ):
        return False
    if vocabulary == "legacy_four_role":
        if roles != {"onset_support"}:
            return False
        positive_roles = {"onset_support"}
    elif vocabulary == "report_graph_v2_five_role":
        allowed_roles = {
            "ictal_pattern_qualification",
            "onset_time_support",
            "onset_topography_support",
        }
        if (
            "onset_time_support" not in roles
            or (require_topography and "onset_topography_support" not in roles)
            or not roles.issubset(allowed_roles)
        ):
            return False
        positive_roles = {"onset_time_support", "onset_topography_support"}
    else:
        return False
    return all(
        (
            row["evidence_role"] == "ictal_pattern_qualification"
            or (
                row["evidence_role"] in positive_roles
                and row["intrinsic_evidence_role"] == "onset_eligible"
                and row["view_role"] == "onset_causal"
                and not row["future_sample_access"]
                and row["onset_evidence_authorized"]
                and row["onset_support_eligible"]
            )
        )
        for row in bindings
    )


def _eeg_claim_ground(
    predicted: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    time_compatible: bool,
    time_details: Mapping[str, float | str],
) -> dict[str, Any]:
    """Ground one aligned claim to event, time, space, and evidence role."""

    event_applicable = (
        predicted["event_id"] is not None or reference["event_id"] is not None
    )
    mode_applicable = (
        predicted["mode_id"] is not None or reference["mode_id"] is not None
    )
    event_match = predicted["event_id"] == reference["event_id"]
    mode_match = predicted["mode_id"] == reference["mode_id"]

    pred_channels = _claim_entity_ids(predicted, _CHANNEL_ENTITY_TYPES)
    ref_channels = _claim_entity_ids(reference, _CHANNEL_ENTITY_TYPES)
    pred_regions = _claim_entity_ids(predicted, {"region"})
    ref_regions = _claim_entity_ids(reference, {"region"})
    channel_jaccard = _set_jaccard(pred_channels, ref_channels)
    region_jaccard = _set_jaccard(pred_regions, ref_regions)
    spatial_values = [
        float(item)
        for item in (channel_jaccard, region_jaccard)
        if not isinstance(item, str)
    ]
    spatial_jaccard: float | str = (
        round(sum(spatial_values) / len(spatial_values), 12)
        if spatial_values
        else "not_available"
    )
    spatial_exact = all(math.isclose(item, 1.0) for item in spatial_values)

    both_recording_relative = (
        predicted["time"]["kind"] == "recording_interval"
        and reference["time"]["kind"] == "recording_interval"
        and predicted["time"]["timebase"] == "recording_relative_seconds"
        and reference["time"]["timebase"] == "recording_relative_seconds"
    )
    recording_temporal_iou: float | str = (
        time_details["interval_iou"] if both_recording_relative else "not_available"
    )
    if both_recording_relative:
        temporal_factor = float(recording_temporal_iou)
    else:
        # Non-temporal and relative-delay claims are still evaluated by the
        # existing typed time comparison, but are not mislabelled as a
        # recording-relative IoU.
        temporal_factor = float(time_compatible)

    predicted_authorization = _temporal_evidence_authorization(predicted)
    reference_authorization = _temporal_evidence_authorization(reference)
    role_compatible = bool(predicted_authorization["authorized"]) and bool(
        reference_authorization["authorized"]
    )
    predicted_complete_when_required = not predicted_authorization[
        "requires_positive_onset_or_soz_authorization"
    ] or bool(predicted_authorization["authorization_complete"])
    authorized = role_compatible and predicted_complete_when_required
    # Evidence IDs are canonical source-graph identities, not local claim IDs.
    # Temporal permission closure within each claim does not prove that the
    # predicted claim cites the *same* evidence as its aligned reference.  Keep
    # the set comparison order-invariant, but require exact equality for every
    # strict/required-facet grounding decision.  This closes the failure mode
    # where a well-typed but wrong evidence ID previously received full
    # EEG-ClaimGround credit.
    predicted_evidence_ids = set(str(item) for item in predicted["evidence_ids"])
    reference_evidence_ids = set(str(item) for item in reference["evidence_ids"])
    canonical_evidence_id_binding_exact = (
        bool(predicted_evidence_ids)
        and bool(reference_evidence_ids)
        and predicted_evidence_ids == reference_evidence_ids
    )
    strict_grounded = (
        event_match
        and mode_match
        and time_compatible
        and spatial_exact
        and authorized
        and canonical_evidence_id_binding_exact
    )
    score = (
        float(event_match)
        * float(mode_match)
        * temporal_factor
        * (float(spatial_jaccard) if not isinstance(spatial_jaccard, str) else 1.0)
        * float(authorized)
        * float(canonical_evidence_id_binding_exact)
    )
    return {
        "schema_version": EEG_CLAIM_GROUND_SCHEMA_VERSION,
        "event_applicable": event_applicable,
        "event_match": event_match,
        "mode_applicable": mode_applicable,
        "mode_match": mode_match,
        "recording_relative_temporal_iou": recording_temporal_iou,
        "recording_relative_temporal_compatible": (
            time_compatible if both_recording_relative else "not_available"
        ),
        "predicted_channel_ids": sorted(pred_channels),
        "reference_channel_ids": sorted(ref_channels),
        "channel_set_jaccard": channel_jaccard,
        "predicted_region_ids": sorted(pred_regions),
        "reference_region_ids": sorted(ref_regions),
        "region_set_jaccard": region_jaccard,
        "spatial_set_jaccard": spatial_jaccard,
        "predicted_canonical_evidence_ids": sorted(predicted_evidence_ids),
        "reference_canonical_evidence_ids": sorted(reference_evidence_ids),
        "canonical_evidence_id_binding_available": bool(predicted_evidence_ids)
        and bool(reference_evidence_ids),
        "canonical_evidence_id_binding_exact": canonical_evidence_id_binding_exact,
        "canonical_evidence_id_binding_required": True,
        "predicted_temporal_authorization": predicted_authorization,
        "reference_temporal_authorization": reference_authorization,
        "temporal_evidence_role_compatible": role_compatible,
        "authorized_temporal_evidence_role": authorized,
        "strict_grounded": strict_grounded,
        "grounding_score": round(score, 12),
        "score_semantics": (
            "event_x_mode_x_recording_time_iou_x_mean_available_spatial_jaccard"
            "_x_authorized_temporal_role_x_canonical_evidence_id_exact_binding"
        ),
    }


def _claim_similarity(
    predicted: Mapping[str, Any],
    reference: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    # Different predicates/kinds are not alternative phrasings of one atomic
    # claim.  They may still be related in the derivation graph, but cannot be
    # paired for factuality credit.
    concept = (
        predicted["claim_kind"] == reference["claim_kind"]
        and predicted["predicate"] == reference["predicate"]
    )
    if not concept:
        return {"alignable": False, "score": 0.0, "strict_supported": False}
    subject_pred = _semantic_entity(predicted["subject"])
    subject_ref = _semantic_entity(reference["subject"])
    subject_exact = subject_pred == subject_ref
    pred_entities = {
        _semantic_entity(item) for item in predicted["object_or_value"]["entities"]
    }
    ref_entities = {
        _semantic_entity(item) for item in reference["object_or_value"]["entities"]
    }
    entity_score = _set_f1(pred_entities, ref_entities)
    entity_exact = pred_entities == ref_entities
    measurement_score, measurement_exact = _measurement_comparison(
        predicted["object_or_value"]["measurements"],
        reference["object_or_value"]["measurements"],
        policy,
    )
    code_exact = (
        predicted["object_or_value"]["code"] == reference["object_or_value"]["code"]
    )
    evidence_ids_pred = set(str(item) for item in predicted["evidence_ids"])
    evidence_ids_ref = set(str(item) for item in reference["evidence_ids"])
    evidence_id_score = (
        len(evidence_ids_pred.intersection(evidence_ids_ref))
        / len(evidence_ids_pred.union(evidence_ids_ref))
        if evidence_ids_pred or evidence_ids_ref
        else 1.0
    )
    # Empty sets do not constitute a canonical provenance binding.  Portable
    # legacy cases may still be evaluated, but cannot receive strict support
    # merely because both sides omit the required source identity.
    evidence_ids_exact = (
        bool(evidence_ids_pred)
        and bool(evidence_ids_ref)
        and evidence_ids_pred == evidence_ids_ref
    )
    evidence_roles_pred = set(str(item) for item in predicted["evidence_roles"])
    evidence_roles_ref = set(str(item) for item in reference["evidence_roles"])
    evidence_role_score = (
        len(evidence_roles_pred.intersection(evidence_roles_ref))
        / len(evidence_roles_pred.union(evidence_roles_ref))
        if evidence_roles_pred or evidence_roles_ref
        else 1.0
    )
    evidence_roles_exact = evidence_roles_pred == evidence_roles_ref
    object_score = (
        entity_score
        + measurement_score
        + float(code_exact)
        + evidence_id_score
        + evidence_role_score
    ) / 5.0
    object_exact = (
        entity_exact
        and measurement_exact
        and code_exact
        and evidence_ids_exact
        and evidence_roles_exact
    )
    attribution_exact = (
        predicted["event_id"] == reference["event_id"]
        and predicted["mode_id"] == reference["mode_id"]
    )
    time_score, time_exact, time_details = _interval_comparison(
        predicted["time"], reference["time"], policy
    )
    eeg_claim_ground = _eeg_claim_ground(
        predicted,
        reference,
        time_compatible=time_exact,
        time_details=time_details,
    )
    polarity_exact = predicted["polarity"] == reference["polarity"]
    negation_exact = predicted["negation_scope"] == reference["negation_scope"]
    assertion_status_exact = (
        predicted["assertion_status"] == reference["assertion_status"]
    )
    epistemic_exact = predicted["epistemic_status"] == reference["epistemic_status"]
    components = {
        "concept": 1.0,
        "subject": float(subject_exact),
        "object": object_score,
        "event_mode_attribution": float(attribution_exact),
        "time": time_score,
        "polarity": float(polarity_exact),
        "negation_scope": float(negation_exact),
        "assertion_status": float(assertion_status_exact),
        "epistemic_status": float(epistemic_exact),
    }
    score = sum(
        float(policy["component_weights"][name]) * value
        for name, value in components.items()
    )
    strict_components = {
        "concept": True,
        "subject": subject_exact,
        "object": object_exact,
        "event_mode_attribution": attribution_exact,
        "time": time_exact,
        "polarity": polarity_exact,
        "negation_scope": negation_exact,
        "assertion_status": assertion_status_exact,
        "epistemic_status": epistemic_exact,
    }
    strict = all(strict_components.values()) and bool(
        eeg_claim_ground["authorized_temporal_evidence_role"]
    )
    return {
        "alignable": score >= float(policy["minimum_alignment_score"]),
        "score": round(score, 12),
        "strict_supported": strict,
        "components": components,
        "strict_components": strict_components,
        "fine_components": {
            "entities": entity_exact,
            "measurements": measurement_exact,
            "code": code_exact,
            "evidence_ids": evidence_ids_exact,
            "evidence_roles": evidence_roles_exact,
        },
        "time_details": time_details,
        "eeg_claim_ground": eeg_claim_ground,
    }


def _maximum_weight_pairs(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return a deterministic maximum-weight one-to-one assignment.

    This is the O(n^3) Hungarian algorithm on a square zero-padded matrix.
    Keeping it local avoids an optional SciPy dependency in report auditing.
    """

    row_count = len(weights)
    column_count = max((len(row) for row in weights), default=0)
    if row_count == 0 or column_count == 0:
        return []
    size = max(row_count, column_count)
    maximum = max((max(row) for row in weights if row), default=0.0)
    cost = [[maximum for _ in range(size)] for _ in range(size)]
    for i, row in enumerate(weights):
        for j, value in enumerate(row):
            cost[i][j] = maximum - float(value)

    # 1-indexed shortest augmenting path formulation for min-cost assignment.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [(p[j] - 1, j - 1) for j in range(1, size + 1) if p[j] > 0]
    return [
        (row, column)
        for row, column in assignment
        if row < row_count and column < column_count
    ]


def _align_nonrelation_claims(
    predicted: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str], set[str], dict[str, str]]:
    comparisons = [
        [_claim_similarity(pred, ref, policy) for ref in reference]
        for pred in predicted
    ]
    weights = [
        [float(item["score"]) if item["alignable"] else 0.0 for item in row]
        for row in comparisons
    ]
    rows: list[dict[str, Any]] = []
    strict_predicted: set[str] = set()
    strict_reference: set[str] = set()
    pred_to_ref: dict[str, str] = {}
    for pred_index, ref_index in _maximum_weight_pairs(weights):
        comparison = comparisons[pred_index][ref_index]
        if not comparison["alignable"] or comparison["score"] <= 0:
            continue
        pred_id = str(predicted[pred_index]["claim_id"])
        ref_id = str(reference[ref_index]["claim_id"])
        pred_to_ref[pred_id] = ref_id
        if comparison["strict_supported"]:
            strict_predicted.add(pred_id)
            strict_reference.add(ref_id)
        rows.append(
            {
                "predicted_claim_id": pred_id,
                "reference_claim_id": ref_id,
                **deepcopy(comparison),
            }
        )
    rows.sort(key=lambda item: (item["predicted_claim_id"], item["reference_claim_id"]))
    return rows, strict_predicted, strict_reference, pred_to_ref


def _align_relation_claims(
    predicted: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    base_pred_to_ref: Mapping[str, str],
    strict_base_predicted: set[str],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    comparisons: list[list[dict[str, Any]]] = []
    for pred in predicted:
        relation = pred["relation_endpoints"]
        mapped_source = base_pred_to_ref.get(str(relation["source_claim_id"]))
        mapped_target = base_pred_to_ref.get(str(relation["target_claim_id"]))
        row: list[dict[str, Any]] = []
        for ref in reference:
            base = _claim_similarity(pred, ref, policy)
            ref_relation = ref["relation_endpoints"]
            endpoints_exact = mapped_source == str(
                ref_relation["source_claim_id"]
            ) and mapped_target == str(ref_relation["target_claim_id"])
            base["relation_endpoints_exact"] = endpoints_exact
            endpoints_strict = (
                str(relation["source_claim_id"]) in strict_base_predicted
                and str(relation["target_claim_id"]) in strict_base_predicted
            )
            if not endpoints_exact:
                base["alignable"] = False
                base["score"] = 0.0
                base["strict_supported"] = False
            elif (
                policy["relations_require_strictly_supported_endpoints"]
                and not endpoints_strict
            ):
                base["strict_supported"] = False
            row.append(base)
        comparisons.append(row)
    weights = [
        [float(item["score"]) if item["alignable"] else 0.0 for item in row]
        for row in comparisons
    ]
    rows: list[dict[str, Any]] = []
    strict_predicted: set[str] = set()
    strict_reference: set[str] = set()
    for pred_index, ref_index in _maximum_weight_pairs(weights):
        comparison = comparisons[pred_index][ref_index]
        if not comparison["alignable"] or comparison["score"] <= 0:
            continue
        pred_id = str(predicted[pred_index]["claim_id"])
        ref_id = str(reference[ref_index]["claim_id"])
        if comparison["strict_supported"]:
            strict_predicted.add(pred_id)
            strict_reference.add(ref_id)
        rows.append(
            {
                "predicted_claim_id": pred_id,
                "reference_claim_id": ref_id,
                **deepcopy(comparison),
            }
        )
    rows.sort(key=lambda item: (item["predicted_claim_id"], item["reference_claim_id"]))
    return rows, strict_predicted, strict_reference


def _epistemic_overstatement(predicted: str, reference: str) -> bool:
    if predicted == reference:
        return False
    pred_family, pred_rank = _EPISTEMIC_RANKS[predicted]
    ref_family, ref_rank = _EPISTEMIC_RANKS[reference]
    if pred_family != ref_family:
        return pred_family != "technical"
    if pred_family == "technical":
        return False
    return pred_rank > ref_rank


def _claim_metrics(
    predicted: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    matches: Sequence[Mapping[str, Any]],
    strict_predicted: set[str],
    strict_reference: set[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    pred_by_id = {str(row["claim_id"]): row for row in predicted}
    ref_by_id = {str(row["claim_id"]): row for row in reference}
    pred_weight = sum(float(row["severity_weight"]) for row in predicted)
    supported_weight = sum(
        float(pred_by_id[item]["severity_weight"]) for item in strict_predicted
    )
    salient_weight = sum(
        float(row["salience_weight"]) for row in reference if row["salient"]
    )
    salient_supported = sum(
        float(ref_by_id[item]["salience_weight"])
        for item in strict_reference
        if ref_by_id[item]["salient"]
    )
    all_reference_weight = sum(
        float(row["salience_weight"] or 1.0) for row in reference
    )
    all_reference_supported = sum(
        float(ref_by_id[item]["salience_weight"] or 1.0) for item in strict_reference
    )
    supported_precision = _ratio(supported_weight, pred_weight)
    salient_recall = _ratio(salient_supported, salient_weight)
    atomic_recall = _ratio(all_reference_supported, all_reference_weight)
    strict_count_precision = _ratio(len(strict_predicted), len(predicted))
    strict_count_recall = _ratio(len(strict_reference), len(reference))
    by_pred = {str(row["predicted_claim_id"]): row for row in matches}
    component_numerator: dict[str, float] = defaultdict(float)
    fine_numerator: dict[str, float] = defaultdict(float)
    fine_denominator: dict[str, float] = defaultdict(float)
    temporal_iou: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    epistemic_overstatements = 0
    critical_unsupported = 0
    critical_overstated = 0
    time_evaluable_count = 0
    time_exact_count = 0
    for pred_id, pred in pred_by_id.items():
        match = by_pred.get(pred_id)
        if pred["critical"] and pred_id not in strict_predicted:
            critical_unsupported += 1
        if match is None:
            continue
        reference_claim = ref_by_id[str(match["reference_claim_id"])]
        for name, exact in match["strict_components"].items():
            component_numerator[name] += float(exact)
        for name, exact in match["fine_components"].items():
            if name != "measurements" or (
                pred["object_or_value"]["measurements"]
                or reference_claim["object_or_value"]["measurements"]
            ):
                fine_denominator[name] += 1.0
                fine_numerator[name] += float(exact)
        for entity_type, metric_name in (
            ("electrode", "channel_entities"),
            ("region", "region_entities"),
            ("laterality", "laterality_entities"),
        ):
            pred_values = {
                str(item["id"])
                for item in [pred["subject"], *pred["object_or_value"]["entities"]]
                if item["type"] == entity_type
            }
            ref_values = {
                str(item["id"])
                for item in [
                    reference_claim["subject"],
                    *reference_claim["object_or_value"]["entities"],
                ]
                if item["type"] == entity_type
            }
            if pred_values or ref_values:
                fine_denominator[metric_name] += 1.0
                fine_numerator[metric_name] += float(pred_values == ref_values)
        if pred["time"]["kind"] != "none" or reference_claim["time"]["kind"] != "none":
            time_evaluable_count += 1
            time_exact_count += int(bool(match["strict_components"]["time"]))
        details = match["time_details"]
        if not isinstance(details["interval_iou"], str):
            temporal_iou.append(float(details["interval_iou"]))
            lower_errors.append(float(details["lower_error_seconds"]))
            upper_errors.append(float(details["upper_error_seconds"]))
        ref = reference_claim
        if _epistemic_overstatement(
            str(pred["epistemic_status"]), str(ref["epistemic_status"])
        ):
            epistemic_overstatements += 1
            if pred["critical"]:
                critical_overstated += 1
    aligned_count = len(matches)
    component_accuracy = {
        name: _ratio(value, aligned_count)
        for name, value in sorted(component_numerator.items())
    }
    # Keep absent components explicit rather than silently changing denominators.
    for name in _COMPONENT_WEIGHTS:
        component_accuracy.setdefault(name, "not_available")
    grounding_accuracy = {
        name: _ratio(fine_numerator[name], fine_denominator[name])
        for name in (
            "channel_entities",
            "region_entities",
            "laterality_entities",
            "measurements",
            "evidence_ids",
            "evidence_roles",
            "entities",
            "code",
        )
    }
    output = {
        "predicted_claim_count": len(predicted),
        "reference_claim_count": len(reference),
        "aligned_claim_count": aligned_count,
        "strict_supported_claim_count": len(strict_predicted),
        "strict_recalled_claim_count": len(strict_reference),
        "supported_claim_precision": supported_precision,
        "salient_claim_recall": salient_recall,
        "atomic_claim_recall": atomic_recall,
        "strict_count_precision": strict_count_precision,
        "strict_count_recall": strict_count_recall,
        "strict_count_f1": _f1(strict_count_precision, strict_count_recall),
        "hallucination_rate": (
            round(1.0 - supported_precision, 12)
            if not isinstance(supported_precision, str)
            else "not_available"
        ),
        "salient_omission_rate": (
            round(1.0 - salient_recall, 12)
            if not isinstance(salient_recall, str)
            else "not_available"
        ),
        "component_accuracy_among_aligned": component_accuracy,
        "grounding_accuracy_among_aligned": grounding_accuracy,
        "mean_temporal_iou": (
            round(sum(temporal_iou) / len(temporal_iou), 12)
            if temporal_iou
            else "not_available"
        ),
        "mean_lower_endpoint_error_seconds": (
            round(sum(lower_errors) / len(lower_errors), 12)
            if lower_errors
            else "not_available"
        ),
        "mean_upper_endpoint_error_seconds": (
            round(sum(upper_errors) / len(upper_errors), 12)
            if upper_errors
            else "not_available"
        ),
        "epistemic_overstatement_count": epistemic_overstatements,
        "epistemic_overstatement_rate_among_aligned": _ratio(
            epistemic_overstatements, aligned_count
        ),
        "unsupported_critical_claim_count": critical_unsupported,
        "overstated_critical_claim_count": critical_overstated,
    }
    stats = {
        "supported_weight": supported_weight,
        "predicted_weight": pred_weight,
        "salient_supported_weight": salient_supported,
        "salient_weight": salient_weight,
        "strict_predicted_count": float(len(strict_predicted)),
        "predicted_count": float(len(predicted)),
        "strict_reference_count": float(len(strict_reference)),
        "reference_count": float(len(reference)),
        "aligned_count": float(aligned_count),
        "epistemic_exact_count": float(
            component_numerator.get("epistemic_status", 0.0)
        ),
        "assertion_status_exact_count": float(
            component_numerator.get("assertion_status", 0.0)
        ),
        "channel_entity_exact_count": float(fine_numerator["channel_entities"]),
        "channel_entity_evaluable_count": float(fine_denominator["channel_entities"]),
        "evidence_binding_exact_count": float(fine_numerator["evidence_ids"]),
        "evidence_binding_evaluable_count": float(fine_denominator["evidence_ids"]),
        "evidence_role_exact_count": float(fine_numerator["evidence_roles"]),
        "evidence_role_evaluable_count": float(fine_denominator["evidence_roles"]),
        "time_exact_count": float(time_exact_count),
        "time_evaluable_count": float(time_evaluable_count),
        "overstatement_count": float(epistemic_overstatements),
        "critical_unsupported_count": float(critical_unsupported),
        "critical_overstated_count": float(critical_overstated),
    }
    return output, stats


def _eeg_claim_ground_metrics(
    matches: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    grounds = [row["eeg_claim_ground"] for row in matches]
    event_rows = [row for row in grounds if row["event_applicable"]]
    mode_rows = [row for row in grounds if row["mode_applicable"]]
    temporal_rows = [
        row
        for row in grounds
        if not isinstance(row["recording_relative_temporal_iou"], str)
    ]
    channel_rows = [
        row for row in grounds if not isinstance(row["channel_set_jaccard"], str)
    ]
    region_rows = [
        row for row in grounds if not isinstance(row["region_set_jaccard"], str)
    ]
    onset_rows = [
        row
        for row in grounds
        if row["predicted_temporal_authorization"][
            "requires_positive_onset_or_soz_authorization"
        ]
    ]
    complete_onset_rows = [
        row
        for row in onset_rows
        if row["predicted_temporal_authorization"]["authorization_complete"]
    ]

    temporal_iou_sum = sum(
        float(row["recording_relative_temporal_iou"]) for row in temporal_rows
    )
    channel_jaccard_sum = sum(float(row["channel_set_jaccard"]) for row in channel_rows)
    region_jaccard_sum = sum(float(row["region_set_jaccard"]) for row in region_rows)
    grounding_score_sum = sum(float(row["grounding_score"]) for row in grounds)
    exact_evidence_binding_count = sum(
        bool(row["canonical_evidence_id_binding_exact"]) for row in grounds
    )
    output = {
        "schema_version": EEG_CLAIM_GROUND_SCHEMA_VERSION,
        "aligned_claim_count": len(grounds),
        "strict_grounded_claim_count": sum(
            bool(row["strict_grounded"]) for row in grounds
        ),
        "strict_grounding_rate": _ratio(
            sum(bool(row["strict_grounded"]) for row in grounds), len(grounds)
        ),
        "canonical_evidence_id_binding_exact_count": exact_evidence_binding_count,
        "canonical_evidence_id_binding_accuracy": _ratio(
            exact_evidence_binding_count, len(grounds)
        ),
        "canonical_evidence_id_binding_required_for_strict_grounding": True,
        "mean_grounding_score": _ratio(grounding_score_sum, len(grounds)),
        "event_match_rate": _ratio(
            sum(bool(row["event_match"]) for row in event_rows), len(event_rows)
        ),
        "mode_match_rate": _ratio(
            sum(bool(row["mode_match"]) for row in mode_rows), len(mode_rows)
        ),
        "mean_recording_relative_temporal_iou": _ratio(
            temporal_iou_sum, len(temporal_rows)
        ),
        "mean_channel_set_jaccard": _ratio(channel_jaccard_sum, len(channel_rows)),
        "mean_region_set_jaccard": _ratio(region_jaccard_sum, len(region_rows)),
        "positive_onset_or_soz_claim_count": len(onset_rows),
        "authorized_positive_onset_or_soz_claim_count": sum(
            bool(row["authorized_temporal_evidence_role"]) for row in onset_rows
        ),
        "positive_onset_or_soz_temporal_authorization_rate": _ratio(
            sum(bool(row["authorized_temporal_evidence_role"]) for row in onset_rows),
            len(onset_rows),
        ),
        "positive_onset_or_soz_complete_binding_rate": _ratio(
            len(complete_onset_rows), len(onset_rows)
        ),
        "legacy_role_only_positive_onset_or_soz_claim_count": (
            len(onset_rows) - len(complete_onset_rows)
        ),
        "interpretation": (
            "event_mode_time_space_temporal_role_and_canonical_evidence_binding_"
            "are_reported_separately"
        ),
    }
    stats = {
        "claim_ground_strict_count": float(
            sum(bool(row["strict_grounded"]) for row in grounds)
        ),
        "claim_ground_aligned_count": float(len(grounds)),
        "claim_ground_score_sum": grounding_score_sum,
        "claim_ground_event_match_count": float(
            sum(bool(row["event_match"]) for row in event_rows)
        ),
        "claim_ground_event_evaluable_count": float(len(event_rows)),
        "claim_ground_mode_match_count": float(
            sum(bool(row["mode_match"]) for row in mode_rows)
        ),
        "claim_ground_mode_evaluable_count": float(len(mode_rows)),
        "claim_ground_temporal_iou_sum": temporal_iou_sum,
        "claim_ground_temporal_evaluable_count": float(len(temporal_rows)),
        "claim_ground_channel_jaccard_sum": channel_jaccard_sum,
        "claim_ground_channel_evaluable_count": float(len(channel_rows)),
        "claim_ground_region_jaccard_sum": region_jaccard_sum,
        "claim_ground_region_evaluable_count": float(len(region_rows)),
        "claim_ground_onset_authorized_count": float(
            sum(bool(row["authorized_temporal_evidence_role"]) for row in onset_rows)
        ),
        "claim_ground_onset_claim_count": float(len(onset_rows)),
        "claim_ground_onset_complete_binding_count": float(len(complete_onset_rows)),
    }
    return output, stats


_LEFT_SCALP_ELECTRODES = {
    "FP1",
    "F3",
    "F7",
    "C3",
    "T3",
    "T5",
    "T7",
    "P3",
    "P7",
    "O1",
    "A1",
    "M1",
}
_RIGHT_SCALP_ELECTRODES = {
    "FP2",
    "F4",
    "F8",
    "C4",
    "T4",
    "T6",
    "T8",
    "P4",
    "P8",
    "O2",
    "A2",
    "M2",
}
_MIDLINE_SCALP_ELECTRODES = {"FZ", "CZ", "PZ", "OZ"}
_SPATIAL_CHANNEL_ENTITY_TYPES = {
    "electrode",
    "channel",
    "bipolar_derivation",
}

# Conservative scalp-topology implications used only for resolution backoff.
# They let an explicitly cited electrode entail its broader region/laterality,
# while never allowing a region or a bipolar derivation to manufacture one of
# its endpoint electrodes.  The aliases mirror the standard 10--20 names that
# the evaluator already recognizes for laterality.  Auricular/mastoid A1/A2
# and M1/M2 remain explicit electrode identities but intentionally do not
# entail a temporal cerebral region.
_SCALP_ELECTRODE_TO_REGION = {
    "FP1": "frontal",
    "FP2": "frontal",
    "F3": "frontal",
    "F4": "frontal",
    "F7": "frontal",
    "F8": "frontal",
    "FZ": "frontal",
    "C3": "central",
    "C4": "central",
    "CZ": "central",
    "T3": "temporal",
    "T4": "temporal",
    "T5": "temporal",
    "T6": "temporal",
    "T7": "temporal",
    "T8": "temporal",
    "P7": "temporal",
    "P8": "temporal",
    "P3": "parietal",
    "P4": "parietal",
    "PZ": "parietal",
    "O1": "occipital",
    "O2": "occipital",
    "OZ": "occipital",
}


def _entity_lateralities(entity: Mapping[str, Any]) -> set[str]:
    """Return only conservative, explicit scalp laterality semantics."""

    entity_type = str(entity["type"])
    entity_id = str(entity["id"])
    normalized = entity_id.lower().replace("-", "_")
    if entity_type == "laterality":
        if normalized == "bilateral":
            return {"left", "right"}
        if normalized in {"left", "right", "midline"}:
            return {normalized}
        return set()
    if entity_type == "region":
        for laterality in ("left", "right", "midline"):
            if normalized == laterality or normalized.startswith(f"{laterality}_"):
                return {laterality}
        if normalized == "bilateral" or normalized.startswith("bilateral_"):
            return {"left", "right"}
        return set()
    if entity_type not in _SPATIAL_CHANNEL_ENTITY_TYPES:
        return set()
    tokens = {token for token in re.split(r"[^A-Za-z0-9]+", entity_id.upper()) if token}
    token_lateralities: dict[str, str] = {}
    for token in tokens:
        if token in _LEFT_SCALP_ELECTRODES:
            token_lateralities[token] = "left"
        elif token in _RIGHT_SCALP_ELECTRODES:
            token_lateralities[token] = "right"
        elif token in _MIDLINE_SCALP_ELECTRODES:
            token_lateralities[token] = "midline"
    if entity_type == "electrode":
        return set(token_lateralities.values())
    if entity_type == "bipolar_derivation" and len(token_lateralities) != 2:
        return set()
    # A lead supports laterality only when every recognized endpoint agrees.
    # A cross-side edge is not constructive evidence for a bilateral onset.
    lateralities = set(token_lateralities.values())
    return lateralities if len(lateralities) == 1 else set()


def _base_region_id(entity_id: str) -> str:
    normalized = entity_id.lower().replace("-", "_")
    for prefix in ("left_", "right_", "bilateral_", "midline_"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _scalp_electrode_tokens(entity_id: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^A-Za-z0-9]+", entity_id.upper())
        if token in _SCALP_ELECTRODE_TO_REGION
    }


def _entity_regions(entity: Mapping[str, Any]) -> set[str]:
    """Return conservative region semantics explicitly entailed by an entity."""

    entity_type = str(entity["type"])
    entity_id = str(entity["id"])
    if entity_type == "region":
        return {_base_region_id(entity_id)}
    if entity_type != "electrode":
        return set()
    return {
        _SCALP_ELECTRODE_TO_REGION[token]
        for token in _scalp_electrode_tokens(entity_id)
    }


def _spatial_signature(claims: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    signature = {
        "lateralities": set(),
        "regions": set(),
        "electrodes": set(),
        "leads": set(),
    }
    for claim in claims:
        entities = [claim["subject"], *claim["object_or_value"]["entities"]]
        for entity in entities:
            signature["lateralities"].update(_entity_lateralities(entity))
            signature["regions"].update(_entity_regions(entity))
            entity_type = str(entity["type"])
            entity_id = str(entity["id"])
            if entity_type == "electrode":
                tokens = _scalp_electrode_tokens(entity_id)
                signature["electrodes"].update(tokens or {entity_id.upper()})
            elif entity_type in {"channel", "bipolar_derivation"}:
                # Lead identity is atomic.  In particular, T7-P7 cannot be
                # split into positive T7 and P7 electrode support.
                signature["leads"].add(entity_id.upper())
    return signature


def _spatial_entailment_reasons(
    conclusion: Mapping[str, Any],
    supporting_claims: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reject an explicit spatial conclusion contradicted by its support.

    Missing spatial resolution remains unresolved for v1 compatibility.  If
    both sides expose a comparable facet, their semantics must overlap.
    """

    if not supporting_claims:
        return []
    conclusion_signature = _spatial_signature([conclusion])
    support_signature = _spatial_signature(supporting_claims)
    reasons: list[str] = []
    for facet in ("lateralities", "regions", "electrodes", "leads"):
        asserted = conclusion_signature[facet]
        supported = support_signature[facet]
        if asserted and supported and asserted.isdisjoint(supported):
            reasons.append(f"conclusion_{facet}_contradicted_by_support")
    return reasons


def _constructive_onset_spatial_entailment_reasons(
    conclusion: Mapping[str, Any],
    authorized_onset_leaves: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Require every asserted spatial facet to exist in causal onset leaves.

    This stronger check is intentionally limited to positive onset/SOZ
    derivations.  The legacy conflict checker above remains compatible with
    portable v1 cases, while a localizing chain can no longer treat a missing
    facet as evidence.  Hierarchical backoff is conservative: an electrode
    may entail its region/laterality, but neither a region nor a lead entails
    an endpoint electrode, and endpoint electrodes do not synthesize a lead.
    """

    conclusion_signature = _spatial_signature([conclusion])
    support_signature = _spatial_signature(authorized_onset_leaves)
    reasons: list[str] = []
    for facet in ("lateralities", "regions", "electrodes", "leads"):
        asserted = conclusion_signature[facet]
        if asserted and not asserted.issubset(support_signature[facet]):
            reasons.append(
                f"conclusion_{facet}_not_constructively_entailed_by_"
                "authorized_onset_leaves"
            )
    if reasons:
        reasons.append("unsupported_spatial_resolution")
    return reasons


def _conclusion_time_supported_by_chain(
    conclusion: Mapping[str, Any],
    premises: Sequence[Mapping[str, Any]],
    leaves: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> bool:
    """Check that an asserted conclusion time is inherited from the chain."""

    conclusion_time = conclusion["time"]
    if conclusion_time["kind"] == "none":
        return True
    # Prefer physical-time leaves.  A derived premise may repeat an invented
    # time, so it is used only when no leaf carries a temporal assertion.
    leaf_times = [item["time"] for item in leaves if item["time"]["kind"] != "none"]
    candidate_times = leaf_times or [
        item["time"] for item in premises if item["time"]["kind"] != "none"
    ]
    return any(
        _interval_comparison(conclusion_time, candidate, policy)[1]
        for candidate in candidate_times
    )


def _validate_derivations(
    derivations: Sequence[Mapping[str, Any]],
    predicted_claims: Sequence[Mapping[str, Any]],
    strict_supported_predicted: set[str],
    strict_reference_by_predicted: Mapping[str, str],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    by_id = {str(row["claim_id"]): row for row in predicted_claims}
    derivation_by_conclusion = {
        str(row["conclusion_claim_id"]): row for row in derivations
    }

    def leaf_claim_ids(claim_id: str, seen: set[str] | None = None) -> set[str]:
        active = set() if seen is None else set(seen)
        if claim_id in active:
            return set()
        active.add(claim_id)
        derivation = derivation_by_conclusion.get(claim_id)
        if derivation is None:
            return {claim_id}
        result: set[str] = set()
        for premise_id in derivation["premise_claim_ids"]:
            result.update(leaf_claim_ids(str(premise_id), active))
        return result

    rows: list[dict[str, Any]] = []
    numerator = 0.0
    denominator = 0.0
    for derivation in derivations:
        conclusion_id = str(derivation["conclusion_claim_id"])
        premise_ids = [str(item) for item in derivation["premise_claim_ids"]]
        conclusion = by_id[conclusion_id]
        premises = [by_id[item] for item in premise_ids]
        reasons: list[str] = []
        if conclusion_id not in strict_supported_predicted:
            reasons.append("conclusion_not_strictly_supported")
        rule = _ALLOWED_DERIVATION_RULES.get(str(derivation["rule_id"]))
        if rule is None:
            reasons.append("rule_not_in_closed_registry")
        else:
            if conclusion["claim_kind"] not in rule["conclusion_kinds"]:
                reasons.append("conclusion_kind_not_allowed_by_rule")
            allowed_predicates = rule.get("conclusion_predicates")
            if (
                allowed_predicates is not None
                and conclusion["predicate"] not in allowed_predicates
            ):
                reasons.append("conclusion_predicate_not_allowed_by_rule")
            if len(premises) < int(rule["minimum_premises"]):
                reasons.append("insufficient_premise_count")
            if any(
                item["claim_kind"] not in rule["premise_kinds"] for item in premises
            ):
                reasons.append("premise_kind_not_allowed_by_rule")
            scope = str(rule["scope"])
            if scope == "same_event" and any(
                item["event_id"] != conclusion["event_id"] for item in premises
            ):
                reasons.append("cross_event_mixing")
            if scope == "same_mode" and any(
                item["mode_id"] != conclusion["mode_id"] for item in premises
            ):
                reasons.append("cross_mode_mixing")
            if scope == "distinct_modes":
                modes = {
                    item["mode_id"] for item in premises if item["mode_id"] is not None
                }
                if len(modes) < 2:
                    reasons.append("multiple_mode_rule_lacks_distinct_modes")
            required_leaf_predicates = set(rule.get("required_leaf_predicates", set()))
            if required_leaf_predicates:
                leaves = [by_id[item] for item in leaf_claim_ids(conclusion_id)]
                if not any(
                    item["predicate"] in required_leaf_predicates for item in leaves
                ):
                    reasons.append("required_leaf_predicate_missing")
        unsupported_premises = sorted(
            item for item in premise_ids if item not in strict_supported_predicted
        )
        if unsupported_premises:
            reasons.append("premises_not_strictly_supported")
        leaf_ids = sorted(leaf_claim_ids(conclusion_id))
        leaves = [by_id[item] for item in leaf_ids]
        reasons.extend(_spatial_entailment_reasons(conclusion, premises))
        reasons.extend(_spatial_entailment_reasons(conclusion, leaves))
        if not _conclusion_time_supported_by_chain(
            conclusion, premises, leaves, policy
        ):
            reasons.append("conclusion_time_not_supported_by_chain")
        positive_onset_chain = _requires_positive_onset_or_soz_authorization(conclusion)
        if positive_onset_chain:
            requires_topography = str(conclusion["predicate"]) in _LOCALIZING_PREDICATES
            authorized_onset_leaves = [
                item
                for item in leaves
                if _strict_positive_onset_leaf_authorized(
                    item, require_topography=requires_topography
                )
            ]
            reasons.extend(
                _constructive_onset_spatial_entailment_reasons(
                    conclusion, authorized_onset_leaves
                )
            )
            disallowed_leaves = [
                item
                for item in leaf_ids
                if not _positive_onset_leaf_has_required_roles(
                    by_id[item], require_topography=requires_topography
                )
            ]
            if disallowed_leaves:
                reasons.append(
                    "localizing_chain_uses_non_onset_leaf"
                    if conclusion["predicate"] in _LOCALIZING_PREDICATES
                    else "positive_onset_chain_uses_non_onset_leaf"
                )
            unauthorized_temporal_leaves = [
                item
                for item in leaf_ids
                if not _strict_positive_onset_leaf_authorized(
                    by_id[item], require_topography=requires_topography
                )
            ]
            if unauthorized_temporal_leaves:
                reasons.append(
                    "localizing_chain_uses_unauthorized_temporal_evidence"
                    if conclusion["predicate"] in _LOCALIZING_PREDICATES
                    else "positive_onset_chain_uses_unauthorized_temporal_evidence"
                )
        valid = not reasons
        weight = float(derivation["weight"])
        denominator += weight
        if valid:
            numerator += weight
        rows.append(
            {
                "derivation_id": str(derivation["derivation_id"]),
                "conclusion_claim_id": conclusion_id,
                "conclusion_reference_claim_id": strict_reference_by_predicted.get(
                    conclusion_id
                ),
                "rule_id": str(derivation["rule_id"]),
                "premise_claim_ids": premise_ids,
                "leaf_claim_ids": leaf_ids,
                "unsupported_premise_claim_ids": unsupported_premises,
                "valid": valid,
                "invalid_reason_codes": sorted(set(reasons)),
                "weight": weight,
            }
        )
    metric = _ratio(numerator, denominator)
    summary = {
        "derivation_count": len(derivations),
        "valid_derivation_count": sum(item["valid"] for item in rows),
        "chain_validity": metric,
        "allowed_rule_registry": sorted(_ALLOWED_DERIVATION_RULES),
        "interpretation": "premises_supported_and_rule_allowed",
    }
    return (
        rows,
        summary,
        {
            "chain_valid_weight": numerator,
            "chain_weight": denominator,
        },
    )


def _evidence_flow_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    total = sum(float(row["weight"]) for row in rows)
    stage_weight = {
        stage: sum(float(row["weight"]) for row in rows if row[stage])
        for stage in _EVIDENCE_FLOW_STAGES
    }
    coverage = {stage: _ratio(value, total) for stage, value in stage_weight.items()}
    transition_loss: dict[str, float | str] = {}
    for earlier, later in zip(_EVIDENCE_FLOW_STAGES, _EVIDENCE_FLOW_STAGES[1:]):
        retained = stage_weight[earlier]
        transition_loss[f"{earlier}_to_{later}"] = (
            _ratio(retained - stage_weight[later], retained)
            if retained > 0
            else "not_available"
        )
    end_retained = stage_weight[_EVIDENCE_FLOW_STAGES[-1]]
    end_to_end_recall = _ratio(end_retained, total)
    end_to_end_loss = (
        round(1.0 - end_to_end_recall, 12)
        if not isinstance(end_to_end_recall, str)
        else "not_available"
    )
    return {
        "salient_evidence_count": len(rows),
        "weighted_stage_recall": coverage,
        "weighted_transition_loss": transition_loss,
        "end_to_end_salient_evidence_recall": end_to_end_recall,
        "evidence_flow_loss_rate": end_to_end_loss,
        "stage_ledger_semantics": "frozen_external_stage_presence_not_inferred_from_prose",
    }, {
        "flow_end_weight": end_retained,
        "flow_total_weight": total,
    }


def evaluate_claim_factuality_case(
    case: object,
    *,
    policy: object | None = None,
) -> dict[str, Any]:
    """Evaluate one record without opening any external data source."""

    data = validate_claim_factuality_case(case)
    match_policy = validate_claim_factuality_policy(policy)
    predicted_nonrelation = [
        row for row in data["predicted_claims"] if row["relation_endpoints"] is None
    ]
    reference_nonrelation = [
        row for row in data["reference_claims"] if row["relation_endpoints"] is None
    ]
    (
        base_matches,
        strict_predicted,
        strict_reference,
        pred_to_ref,
    ) = _align_nonrelation_claims(
        predicted_nonrelation, reference_nonrelation, match_policy
    )
    predicted_relations = [
        row for row in data["predicted_claims"] if row["relation_endpoints"] is not None
    ]
    reference_relations = [
        row for row in data["reference_claims"] if row["relation_endpoints"] is not None
    ]
    (
        relation_matches,
        strict_predicted_relations,
        strict_reference_relations,
    ) = _align_relation_claims(
        predicted_relations,
        reference_relations,
        base_pred_to_ref=pred_to_ref,
        strict_base_predicted=strict_predicted,
        policy=match_policy,
    )
    all_matches = base_matches + relation_matches
    all_strict_predicted = strict_predicted | strict_predicted_relations
    all_strict_reference = strict_reference | strict_reference_relations
    metrics, stats = _claim_metrics(
        data["predicted_claims"],
        data["reference_claims"],
        all_matches,
        all_strict_predicted,
        all_strict_reference,
    )
    eeg_claim_ground_metrics, eeg_claim_ground_stats = _eeg_claim_ground_metrics(
        all_matches
    )
    stats.update(eeg_claim_ground_stats)
    relation_precision = _ratio(
        len(strict_predicted_relations), len(predicted_relations)
    )
    relation_recall = _ratio(len(strict_reference_relations), len(reference_relations))
    relation_metrics = {
        "predicted_relation_count": len(predicted_relations),
        "reference_relation_count": len(reference_relations),
        "strict_relation_match_count": len(strict_predicted_relations),
        "relation_precision": relation_precision,
        "relation_recall": relation_recall,
        "relation_f1": _f1(relation_precision, relation_recall),
        "direction_and_endpoint_mapping_required": True,
    }
    stats.update(
        {
            "relation_tp": float(len(strict_predicted_relations)),
            "relation_predicted": float(len(predicted_relations)),
            "relation_reference": float(len(reference_relations)),
        }
    )
    strict_ref_by_pred = {
        str(row["predicted_claim_id"]): str(row["reference_claim_id"])
        for row in all_matches
        if row["strict_supported"]
    }
    derivation_rows, chain_metrics, chain_stats = _validate_derivations(
        data["derivations"],
        data["predicted_claims"],
        all_strict_predicted,
        strict_ref_by_pred,
        match_policy,
    )
    stats.update(chain_stats)
    evidence_flow, flow_stats = _evidence_flow_metrics(data["evidence_flow"])
    stats.update(flow_stats)
    unmatched_predicted = sorted(
        set(str(row["claim_id"]) for row in data["predicted_claims"])
        - set(str(row["predicted_claim_id"]) for row in all_matches)
    )
    unmatched_reference = sorted(
        set(str(row["claim_id"]) for row in data["reference_claims"])
        - set(str(row["reference_claim_id"]) for row in all_matches)
    )
    artifact = {
        "schema_version": CLAIM_FACTUALITY_EVALUATION_SCHEMA_VERSION,
        "status": "completed_eeg_claim_factuality_evaluation",
        "case_id": data["case_id"],
        "patient_id": data["patient_id"],
        "record_id": data["record_id"],
        "policy": match_policy,
        "policy_sha256": _canonical_sha256(match_policy),
        "input_case_sha256": _canonical_sha256(data),
        "claim_metrics": metrics,
        "eeg_claim_ground_metrics": eeg_claim_ground_metrics,
        "relation_metrics": relation_metrics,
        "chain_metrics": chain_metrics,
        "evidence_flow_metrics": evidence_flow,
        "matches": sorted(
            all_matches,
            key=lambda item: (item["predicted_claim_id"], item["reference_claim_id"]),
        ),
        "derivations": derivation_rows,
        "unmatched_predicted_claim_ids": unmatched_predicted,
        "unmatched_reference_claim_ids": unmatched_reference,
        "sufficient_statistics": stats,
        "claim_boundary": {
            **dict(_CLAIM_BOUNDARY),
            "free_text_parsed_by_evaluator": False,
            "clinical_correctness_claimed": False,
            "top1_accuracy_substituted_for_factuality": False,
            "structural_validation_substituted_for_factuality": False,
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def validate_claim_factuality_evaluation_artifact(
    value: object,
) -> dict[str, Any]:
    """Validate one sealed case-evaluation artifact before aggregation.

    This is an integrity and schema check, not a replacement for replaying the
    evaluation from its source case.  In particular, it prevents a caller from
    changing sufficient statistics while retaining the evaluator-issued
    content hash and then publishing altered patient-level metrics.
    """

    data = _strict_object(
        value,
        _EVALUATION_ARTIFACT_KEYS,
        "claim factuality evaluation artifact",
    )
    if data["schema_version"] != CLAIM_FACTUALITY_EVALUATION_SCHEMA_VERSION:
        raise ValueError("claim factuality evaluation schema_version mismatch")
    if data["status"] != "completed_eeg_claim_factuality_evaluation":
        raise ValueError("claim factuality evaluation status mismatch")
    for name in ("case_id", "patient_id", "record_id"):
        data[name] = _identifier(data[name], f"claim factuality evaluation.{name}")
    policy = validate_claim_factuality_policy(data["policy"])
    if data["policy"] != policy:
        raise ValueError("claim factuality evaluation policy is not canonical")
    policy_sha256 = _sha256_identifier(
        data["policy_sha256"], "claim factuality evaluation.policy_sha256"
    )
    if policy_sha256 != _canonical_sha256(policy):
        raise ValueError("claim factuality evaluation policy hash mismatch")
    _sha256_identifier(
        data["input_case_sha256"],
        "claim factuality evaluation.input_case_sha256",
    )
    artifact_sha256 = _sha256_identifier(
        data["artifact_sha256"],
        "claim factuality evaluation.artifact_sha256",
    )
    statistics = _strict_object(
        data["sufficient_statistics"],
        _EVALUATION_SUFFICIENT_STATISTIC_KEYS,
        "claim factuality evaluation.sufficient_statistics",
    )
    for name, statistic in statistics.items():
        _finite_nonnegative(
            statistic,
            f"claim factuality evaluation.sufficient_statistics.{name}",
        )
    data["sufficient_statistics"] = statistics
    unhashed = deepcopy(data)
    unhashed.pop("artifact_sha256")
    if artifact_sha256 != _canonical_sha256(unhashed):
        raise ValueError("claim factuality evaluation artifact content hash mismatch")
    return data


_AGGREGATE_METRICS = (
    "supported_claim_precision",
    "salient_claim_recall",
    "strict_count_f1",
    "relation_f1",
    "temporal_compatibility",
    "assertion_status_accuracy",
    "epistemic_status_accuracy",
    "epistemic_overstatement_rate",
    "channel_entity_accuracy",
    "evidence_binding_accuracy",
    "evidence_role_accuracy",
    "chain_validity",
    "end_to_end_salient_evidence_recall",
    "evidence_flow_loss_rate",
    "claim_ground_strict_rate",
    "claim_ground_mean_score",
    "claim_ground_event_match_rate",
    "claim_ground_mode_match_rate",
    "claim_ground_mean_temporal_iou",
    "claim_ground_mean_channel_jaccard",
    "claim_ground_mean_region_jaccard",
    "claim_ground_onset_authorization_rate",
    "claim_ground_onset_complete_binding_rate",
)


def _patient_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, float | str]:
    stats: dict[str, float] = defaultdict(float)
    for result in results:
        for key, value in result["sufficient_statistics"].items():
            stats[key] += float(value)
    precision = _ratio(stats["supported_weight"], stats["predicted_weight"])
    salient_recall = _ratio(stats["salient_supported_weight"], stats["salient_weight"])
    count_precision = _ratio(stats["strict_predicted_count"], stats["predicted_count"])
    count_recall = _ratio(stats["strict_reference_count"], stats["reference_count"])
    relation_precision = _ratio(stats["relation_tp"], stats["relation_predicted"])
    relation_recall = _ratio(stats["relation_tp"], stats["relation_reference"])
    flow_recall = _ratio(stats["flow_end_weight"], stats["flow_total_weight"])
    return {
        "supported_claim_precision": precision,
        "salient_claim_recall": salient_recall,
        "strict_count_f1": _f1(count_precision, count_recall),
        "relation_f1": _f1(relation_precision, relation_recall),
        "temporal_compatibility": _ratio(
            stats["time_exact_count"], stats["time_evaluable_count"]
        ),
        "assertion_status_accuracy": _ratio(
            stats["assertion_status_exact_count"], stats["aligned_count"]
        ),
        "epistemic_status_accuracy": _ratio(
            stats["epistemic_exact_count"], stats["aligned_count"]
        ),
        "epistemic_overstatement_rate": _ratio(
            stats["overstatement_count"], stats["aligned_count"]
        ),
        "channel_entity_accuracy": _ratio(
            stats["channel_entity_exact_count"],
            stats["channel_entity_evaluable_count"],
        ),
        "evidence_binding_accuracy": _ratio(
            stats["evidence_binding_exact_count"],
            stats["evidence_binding_evaluable_count"],
        ),
        "evidence_role_accuracy": _ratio(
            stats["evidence_role_exact_count"],
            stats["evidence_role_evaluable_count"],
        ),
        "chain_validity": _ratio(stats["chain_valid_weight"], stats["chain_weight"]),
        "end_to_end_salient_evidence_recall": flow_recall,
        "evidence_flow_loss_rate": (
            round(1.0 - flow_recall, 12)
            if not isinstance(flow_recall, str)
            else "not_available"
        ),
        "claim_ground_strict_rate": _ratio(
            stats["claim_ground_strict_count"],
            stats["claim_ground_aligned_count"],
        ),
        "claim_ground_mean_score": _ratio(
            stats["claim_ground_score_sum"],
            stats["claim_ground_aligned_count"],
        ),
        "claim_ground_event_match_rate": _ratio(
            stats["claim_ground_event_match_count"],
            stats["claim_ground_event_evaluable_count"],
        ),
        "claim_ground_mode_match_rate": _ratio(
            stats["claim_ground_mode_match_count"],
            stats["claim_ground_mode_evaluable_count"],
        ),
        "claim_ground_mean_temporal_iou": _ratio(
            stats["claim_ground_temporal_iou_sum"],
            stats["claim_ground_temporal_evaluable_count"],
        ),
        "claim_ground_mean_channel_jaccard": _ratio(
            stats["claim_ground_channel_jaccard_sum"],
            stats["claim_ground_channel_evaluable_count"],
        ),
        "claim_ground_mean_region_jaccard": _ratio(
            stats["claim_ground_region_jaccard_sum"],
            stats["claim_ground_region_evaluable_count"],
        ),
        "claim_ground_onset_authorization_rate": _ratio(
            stats["claim_ground_onset_authorized_count"],
            stats["claim_ground_onset_claim_count"],
        ),
        "claim_ground_onset_complete_binding_rate": _ratio(
            stats["claim_ground_onset_complete_binding_count"],
            stats["claim_ground_onset_claim_count"],
        ),
        "unsupported_critical_claim_count": round(
            stats["critical_unsupported_count"], 12
        ),
        "overstated_critical_claim_count": round(
            stats["critical_overstated_count"], 12
        ),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_patient_claim_factuality(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260822,
) -> dict[str, Any]:
    """Aggregate record evaluations with patients as the sampling unit."""

    if not evaluations:
        raise ValueError(
            "patient factuality aggregation requires at least one evaluation"
        )
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 0
    ):
        raise ValueError("bootstrap_replicates must be a non-negative integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    validated_evaluations = [
        validate_claim_factuality_evaluation_artifact(result) for result in evaluations
    ]
    case_ids: set[str] = set()
    record_keys: set[tuple[str, str]] = set()
    policy_hashes: set[str] = set()
    by_patient: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, result in enumerate(validated_evaluations):
        case_id = _identifier(result["case_id"], f"evaluation[{index}].case_id")
        patient_id = _identifier(
            result["patient_id"], f"evaluation[{index}].patient_id"
        )
        record_id = _identifier(result["record_id"], f"evaluation[{index}].record_id")
        if case_id in case_ids or (patient_id, record_id) in record_keys:
            raise ValueError("factuality aggregation contains duplicate cases/records")
        case_ids.add(case_id)
        record_keys.add((patient_id, record_id))
        policy_hashes.add(str(result["policy_sha256"]))
        by_patient[patient_id].append(result)
    if len(policy_hashes) != 1:
        raise ValueError("patient factuality aggregation cannot mix match policies")
    patient_rows = []
    for patient_id in sorted(by_patient):
        metrics = _patient_metrics(by_patient[patient_id])
        patient_rows.append(
            {
                "patient_id": patient_id,
                "record_count": len(by_patient[patient_id]),
                "case_ids": sorted(
                    str(item["case_id"]) for item in by_patient[patient_id]
                ),
                "metrics": metrics,
            }
        )
    macro: dict[str, dict[str, Any]] = {}
    for metric in _AGGREGATE_METRICS:
        values = [
            float(row["metrics"][metric])
            for row in patient_rows
            if not isinstance(row["metrics"][metric], str)
        ]
        macro[metric] = {
            "available_patient_count": len(values),
            "not_available_patient_count": len(patient_rows) - len(values),
            "patient_macro_average": (
                round(sum(values) / len(values), 12) if values else "not_available"
            ),
            "patient_bootstrap_percentile_95_ci": "not_available",
            "valid_bootstrap_replicates": 0,
        }
    if bootstrap_replicates:
        rng = random.Random(bootstrap_seed)
        patient_ids = sorted(by_patient)
        replicate_values: dict[str, list[float]] = defaultdict(list)
        for _ in range(bootstrap_replicates):
            sampled_ids = [rng.choice(patient_ids) for _ in patient_ids]
            sampled_rows = [_patient_metrics(by_patient[item]) for item in sampled_ids]
            for metric in _AGGREGATE_METRICS:
                values = [
                    float(row[metric])
                    for row in sampled_rows
                    if not isinstance(row[metric], str)
                ]
                if values:
                    replicate_values[metric].append(sum(values) / len(values))
        for metric in _AGGREGATE_METRICS:
            values = replicate_values[metric]
            if values:
                macro[metric]["patient_bootstrap_percentile_95_ci"] = [
                    round(_percentile(values, 0.025), 12),
                    round(_percentile(values, 0.975), 12),
                ]
                macro[metric]["valid_bootstrap_replicates"] = len(values)
    return {
        "patient_count": len(patient_rows),
        "record_count": len(validated_evaluations),
        "policy_sha256": next(iter(policy_hashes)),
        "patient_rows": patient_rows,
        "patient_macro_summary": macro,
        "bootstrap": {
            "sampling_unit": "patient",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "records_or_events_sampled_independently": False,
        },
        "aggregation_contract": {
            "primary_estimand": "equal_weight_patient_macro",
            "records_combined_within_patient_before_macro": True,
            "events_never_treated_as_independent_samples": True,
            "missing_metric_not_scored_as_zero": True,
            "micro_pooling_primary": False,
        },
    }


def evaluate_claim_factuality_dataset(
    cases: Sequence[object],
    *,
    policy: object | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260822,
) -> dict[str, Any]:
    """Evaluate cases and return a patient-level factuality dashboard."""

    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise ValueError("claim factuality dataset requires at least one case")
    match_policy = validate_claim_factuality_policy(policy)
    evaluations = [
        evaluate_claim_factuality_case(case, policy=match_policy) for case in cases
    ]
    patient = aggregate_patient_claim_factuality(
        evaluations,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    artifact = {
        "schema_version": "eeg_claim_factuality_dataset_evaluation_v1",
        "status": "completed_patient_level_eeg_claim_factuality_evaluation",
        "case_count": len(evaluations),
        "policy": match_policy,
        "policy_sha256": _canonical_sha256(match_policy),
        "record_evaluations": evaluations,
        "patient_aggregation": patient,
        "dashboard_policy": {
            "supported_precision_and_salient_recall_separate": True,
            "relation_time_epistemic_reported_separately": True,
            "eeg_claim_ground_event_time_space_role_reported_separately": True,
            "chain_validity_not_folded_into_top1": True,
            "evidence_flow_loss_not_conditioned_on_detected_events": True,
            "single_composite_score_reported": False,
        },
        "claim_boundary": {
            **dict(_CLAIM_BOUNDARY),
            "external_clinical_reference_used": False,
            "clinical_validation_claimed": False,
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


__all__ = [
    "CLAIM_FACTUALITY_CASE_SCHEMA_VERSION",
    "CLAIM_FACTUALITY_EVALUATION_SCHEMA_VERSION",
    "CLAIM_FACTUALITY_POLICY_SCHEMA_VERSION",
    "EEG_CLAIM_GROUND_SCHEMA_VERSION",
    "DEFAULT_CLAIM_FACTUALITY_POLICY",
    "aggregate_patient_claim_factuality",
    "evaluate_claim_factuality_case",
    "evaluate_claim_factuality_dataset",
    "validate_claim_factuality_case",
    "validate_claim_factuality_evaluation_artifact",
    "validate_claim_factuality_policy",
]
