"""Source-bound materialization of EEG report factuality cases.

The legacy :mod:`claim_factuality_evaluation` API intentionally accepts a
portable, no-I/O case.  That is useful for evaluation, but it also means a
caller can construct claim weights and stage booleans by hand.  This module is
the stricter bridge used before that evaluator:

* a frozen EEG EvidenceGraph projection supplies event/finding presence and
  per-evidence temporal permissions;
* a host-validated record hypothesis graph supplies typed claims and support
  edges;
* a persisted deterministic render supplies the sentence ledger;
* a frozen complete event roster supplies the denominator, including events
  that disappeared before Findings or report construction.

Predicted/reference claims, derivations, evidence-flow stages, severity,
salience and criticality are derived here.  None of those fields is accepted
from the caller.  Validation requires the four sources again and replays the
deterministic renderer byte-for-byte, so changing text, a ledger, a graph or a
source binding and merely recomputing a self hash fails closed.

This is an EEG-only, no-I/O evaluation utility.  It does not read EDF files,
private data, annotations, spreadsheets, physician labels or clinical text.
It proves serialization and provenance closure; it does not prove that an
upstream signal Finding is clinically correct.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .claim_factuality_evaluation import (
    CLAIM_FACTUALITY_CASE_SCHEMA_VERSION,
    evaluate_claim_factuality_case,
    validate_claim_factuality_case,
)
from .multievent_report_render import (
    render_multievent_soz_report_zh,
    validate_multievent_report_render,
)
from .multievent_soz_claim_validation import (
    validate_multievent_soz_report_payload,
)


SOURCE_BOUND_FACTUALITY_MATERIALIZATION_SCHEMA_VERSION = (
    "eeg_source_bound_factuality_case_materialization_v1"
)
SOURCE_BOUND_FACTUALITY_MATERIALIZER_ID = (
    "source_bound_eeg_factuality_case_materializer_v1"
)
FROZEN_EVIDENCE_GRAPH_PROJECTION_SCHEMA_VERSION = (
    "eeg_factuality_evidence_graph_projection_v1"
)
COMPLETE_EVENT_ROSTER_SCHEMA_VERSION = "eeg_factuality_complete_event_roster_v1"
FACTUALITY_WEIGHT_POLICY_ID = "eeg_source_bound_claim_weight_policy_v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

_FIREWALL = {
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "ecg_emg_eog_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
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

_EVIDENCE_ROLES = {
    "onset_support",
    "spread_support",
    "contradiction",
    "context_only",
}
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
}
_EVIDENCE_STATUSES = {
    "present",
    "absent_with_opportunity",
    "uncertain",
    "not_evaluable",
}
_ASSERTION_LEVELS = {"measured", "model_candidate", "clinically_qualified"}
_OBSERVATION_EPISTEMIC_STATUSES = {
    "measured",
    "model_candidate",
    "clinically_qualified",
    "not_evaluable",
}
_OBSERVATION_PREDICATES = {
    "event_detected",
    "earliest_sustained_change_maximal_at",
    "rhythm_or_morphology_observed",
    "evolves_in_frequency",
    "evolves_in_amplitude",
    "precedes_recruitment_of",
    "near_synchronous_with",
    "recruits_to",
    "terminates_at",
    "recovers_after",
    "artifact_limits_interpretation",
    "bilateral_synchronous_evolution_observed",
    "no_stable_focal_lead_observed",
    "record_signal_technically_limited",
}

# These predicates are the positive onset/SOZ set used by the factuality
# evaluator.  The materializer applies a stronger source gate: every cited
# evidence item, not merely one of them, must be onset-causal and future-free.
_POSITIVE_ONSET_OR_SOZ_PREDICATES = {
    "earliest_sustained_change_maximal_at",
    "event_has_onset_phenotype",
    "event_supports_soz_candidate",
    "mode_repeats_onset_pattern",
    "mode_supports_soz_candidate",
    "record_primary_soz_hypothesis",
    "record_alternative_soz_hypothesis",
    "record_has_generalized_synchronous_onset",
    "record_has_multiple_onset_modes",
}
_TEMPORAL_RELATION_PREDICATES = {
    "precedes_recruitment_of",
    "near_synchronous_with",
    "recruits_to",
}

# Salience, severity and criticality are an engineering policy, not an expert
# gold standard.  Keeping the full registry local and content-addressed makes
# it impossible for one evaluation case to self-award favorable weights.
_CLAIM_WEIGHT_POLICY: Mapping[str, Mapping[str, object]] = {
    "event_detected": {"severity": 1.0, "salience": 1.0, "critical": False},
    "earliest_sustained_change_maximal_at": {
        "severity": 3.0,
        "salience": 3.0,
        "critical": True,
    },
    "rhythm_or_morphology_observed": {
        "severity": 2.0,
        "salience": 2.0,
        "critical": False,
    },
    "evolves_in_frequency": {"severity": 2.0, "salience": 2.0, "critical": False},
    "evolves_in_amplitude": {"severity": 2.0, "salience": 2.0, "critical": False},
    "precedes_recruitment_of": {
        "severity": 3.0,
        "salience": 2.0,
        "critical": False,
    },
    "near_synchronous_with": {"severity": 3.0, "salience": 2.0, "critical": False},
    "recruits_to": {"severity": 2.0, "salience": 2.0, "critical": False},
    "terminates_at": {"severity": 1.0, "salience": 1.0, "critical": False},
    "recovers_after": {"severity": 1.0, "salience": 1.0, "critical": False},
    "artifact_limits_interpretation": {
        "severity": 3.0,
        "salience": 3.0,
        "critical": False,
    },
    "bilateral_synchronous_evolution_observed": {
        "severity": 3.0,
        "salience": 3.0,
        "critical": True,
    },
    "no_stable_focal_lead_observed": {
        "severity": 3.0,
        "salience": 3.0,
        "critical": True,
    },
    "record_signal_technically_limited": {
        "severity": 3.0,
        "salience": 3.0,
        "critical": True,
    },
    # Event-level hypotheses are useful derivation nodes but are not forced
    # into a concise record-level report.  Atomic recall still exposes their
    # omission; salient recall is reserved for source observations and the
    # final multi-event interpretation.
    "event_has_onset_phenotype": {
        "severity": 2.0,
        "salience": 0.0,
        "critical": False,
    },
    "event_supports_soz_candidate": {
        "severity": 3.0,
        "salience": 0.0,
        "critical": False,
    },
    "mode_repeats_onset_pattern": {
        "severity": 3.0,
        "salience": 3.0,
        "critical": False,
    },
    "mode_supports_soz_candidate": {
        "severity": 3.0,
        "salience": 3.0,
        "critical": False,
    },
    "record_primary_soz_hypothesis": {
        "severity": 4.0,
        "salience": 4.0,
        "critical": True,
    },
    "record_alternative_soz_hypothesis": {
        "severity": 3.0,
        "salience": 2.0,
        "critical": False,
    },
    "record_has_multiple_onset_modes": {
        "severity": 4.0,
        "salience": 4.0,
        "critical": True,
    },
    "record_has_generalized_synchronous_onset": {
        "severity": 4.0,
        "salience": 4.0,
        "critical": True,
    },
    "record_onset_nonlocalizable": {
        "severity": 4.0,
        "salience": 4.0,
        "critical": True,
    },
    "record_technical_limited": {
        "severity": 4.0,
        "salience": 4.0,
        "critical": True,
    },
    "supports_claim": {"severity": 1.0, "salience": 0.0, "critical": False},
    "contradicts_claim": {"severity": 2.0, "salience": 1.0, "critical": False},
}
_EVIDENCE_FLOW_WEIGHTS: Mapping[str, float] = {
    "onset_support": 3.0,
    "contradiction": 2.0,
    "spread_support": 1.5,
    "context_only": 1.0,
    "event_without_finding": 1.0,
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


FACTUALITY_WEIGHT_POLICY_SHA256 = _canonical_sha256(
    {
        "policy_id": FACTUALITY_WEIGHT_POLICY_ID,
        "claim_weights": _CLAIM_WEIGHT_POLICY,
        "evidence_flow_weights": _EVIDENCE_FLOW_WEIGHTS,
    }
)


def _strict_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    actual = set(value)
    missing = keys.difference(actual)
    extra = actual.difference(keys)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{context} must be an opaque identifier")
    return value


def _optional_identifier(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, context)


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite_nonnegative(value: object, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be finite and {qualifier}")
    return result


def _unique_identifiers(value: object, context: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{context} must be a{' non-empty' if not allow_empty else ''} list")
    rows = [_identifier(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if len(rows) != len(set(rows)):
        raise ValueError(f"{context} contains duplicate identifiers")
    return rows


def _validate_firewall(value: object, context: str) -> dict[str, bool]:
    data = _strict_object(value, set(_FIREWALL), context)
    for key, expected in _FIREWALL.items():
        if data[key] is not expected:
            raise ValueError(f"{context}.{key} must be {expected}")
    return dict(_FIREWALL)


def _validate_interval(value: object, context: str) -> dict[str, float]:
    data = _strict_object(value, {"lower", "upper", "resolution_seconds"}, context)
    lower = _finite_nonnegative(data["lower"], f"{context}.lower")
    upper = _finite_nonnegative(data["upper"], f"{context}.upper")
    resolution = _finite_nonnegative(
        data["resolution_seconds"], f"{context}.resolution_seconds", positive=True
    )
    if lower > upper:
        raise ValueError(f"{context} lower exceeds upper")
    return {"lower": lower, "upper": upper, "resolution_seconds": resolution}


def _validate_source_evidence(value: object, context: str) -> dict[str, Any]:
    keys = {
        "evidence_id",
        "finding_id",
        "family",
        "term",
        "evidence_role",
        "status",
        "assertion_level",
        "waveform_evidence_ids",
        "intrinsic_evidence_role",
        "view_role",
        "future_sample_access",
        "onset_evidence_authorized",
        "onset_support_eligible",
    }
    row = _strict_object(value, keys, context)
    for key in ("evidence_id", "finding_id", "family", "term"):
        row[key] = _identifier(row[key], f"{context}.{key}")
    if row["evidence_role"] not in _EVIDENCE_ROLES:
        raise ValueError(f"{context}.evidence_role is invalid")
    if row["status"] not in _EVIDENCE_STATUSES:
        raise ValueError(f"{context}.status is invalid")
    if row["assertion_level"] not in _ASSERTION_LEVELS:
        raise ValueError(f"{context}.assertion_level is invalid")
    if row["intrinsic_evidence_role"] not in _INTRINSIC_EVIDENCE_ROLES:
        raise ValueError(f"{context}.intrinsic_evidence_role is invalid")
    if row["view_role"] not in _VIEW_ROLES:
        raise ValueError(f"{context}.view_role is invalid")
    row["waveform_evidence_ids"] = _unique_identifiers(
        row["waveform_evidence_ids"], f"{context}.waveform_evidence_ids"
    )
    for key in (
        "future_sample_access",
        "onset_evidence_authorized",
        "onset_support_eligible",
    ):
        if type(row[key]) is not bool:
            raise TypeError(f"{context}.{key} must be boolean")
    if row["status"] == "not_evaluable" and row["waveform_evidence_ids"]:
        raise ValueError(f"{context} not_evaluable evidence cannot cite waveform evidence")
    strict_causal = (
        row["intrinsic_evidence_role"] == "onset_eligible"
        and row["view_role"] == "onset_causal"
        and not row["future_sample_access"]
        and row["onset_evidence_authorized"]
        and row["onset_support_eligible"]
    )
    if row["evidence_role"] == "onset_support" and not strict_causal:
        raise ValueError(
            f"{context} onset_support must be onset-causal, future-free and authorized"
        )
    if row["intrinsic_evidence_role"] == "onset_eligible" and not strict_causal:
        raise ValueError(f"{context} onset_eligible temporal permission is incomplete")
    if row["future_sample_access"] and (
        row["onset_evidence_authorized"] or row["onset_support_eligible"]
    ):
        raise ValueError(f"{context} future-dependent evidence cannot authorize onset")
    if row["view_role"] == "context_offline" and (
        row["onset_evidence_authorized"] or row["onset_support_eligible"]
    ):
        raise ValueError(f"{context} offline context cannot authorize onset")
    return row


def _validate_entity(value: object, context: str) -> dict[str, str]:
    row = _strict_object(value, {"type", "id"}, context)
    return {
        "type": _identifier(row["type"], f"{context}.type"),
        "id": _identifier(row["id"], f"{context}.id"),
    }


def _validate_claim_object(value: object, context: str) -> dict[str, Any]:
    row = _strict_object(value, {"entities", "measurements", "code"}, context)
    if not isinstance(row["entities"], list):
        raise TypeError(f"{context}.entities must be a list")
    entities = [
        _validate_entity(item, f"{context}.entities[{index}]")
        for index, item in enumerate(row["entities"])
    ]
    entity_keys = [(item["type"], item["id"]) for item in entities]
    if len(entity_keys) != len(set(entity_keys)):
        raise ValueError(f"{context}.entities contains duplicates")
    if not isinstance(row["measurements"], list):
        raise TypeError(f"{context}.measurements must be a list")
    measurements: list[dict[str, Any]] = []
    measurement_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(row["measurements"]):
        measurement_context = f"{context}.measurements[{index}]"
        measurement = _strict_object(
            item, {"name", "value", "unit"}, measurement_context
        )
        name = _identifier(measurement["name"], f"{measurement_context}.name")
        unit = measurement["unit"]
        if not isinstance(unit, str) or not unit or len(unit) > 64:
            raise ValueError(f"{measurement_context}.unit is invalid")
        value_number = measurement["value"]
        if isinstance(value_number, bool) or not isinstance(value_number, (int, float)):
            raise TypeError(f"{measurement_context}.value must be a finite number")
        value_float = float(value_number)
        if not math.isfinite(value_float):
            raise ValueError(f"{measurement_context}.value must be finite")
        key = (name, unit)
        if key in measurement_keys:
            raise ValueError(f"{context}.measurements contains duplicates")
        measurement_keys.add(key)
        measurements.append({"name": name, "value": value_float, "unit": unit})
    code = row["code"]
    if code is not None:
        code = _identifier(code, f"{context}.code")
    return {"entities": entities, "measurements": measurements, "code": code}


def _validate_claim_time(value: object, context: str) -> dict[str, Any]:
    row = _strict_object(
        value,
        {"kind", "timebase", "lower", "upper", "left_censored", "right_censored"},
        context,
    )
    kind = row["kind"]
    expected_timebase = {
        "none": "not_applicable",
        "recording_interval": "recording_relative_seconds",
        "delay_interval": "relative_delay_seconds",
    }.get(str(kind))
    if expected_timebase is None or row["timebase"] != expected_timebase:
        raise ValueError(f"{context} kind/timebase is invalid")
    for key in ("left_censored", "right_censored"):
        if type(row[key]) is not bool:
            raise TypeError(f"{context}.{key} must be boolean")
    if kind == "none":
        if (
            row["lower"] is not None
            or row["upper"] is not None
            or row["left_censored"]
            or row["right_censored"]
        ):
            raise ValueError(f"{context} kind=none cannot carry temporal values")
    else:
        for key in ("lower", "upper"):
            value_number = row[key]
            if isinstance(value_number, bool) or not isinstance(value_number, (int, float)):
                raise TypeError(f"{context}.{key} must be a finite number")
            value_float = float(value_number)
            if not math.isfinite(value_float) or (
                kind == "recording_interval" and value_float < 0
            ):
                raise ValueError(f"{context}.{key} is invalid")
            row[key] = value_float
        if float(row["lower"]) > float(row["upper"]):
            raise ValueError(f"{context} lower exceeds upper")
    return row


def _validate_source_observation(value: object, context: str) -> dict[str, Any]:
    keys = {
        "source_observation_id",
        "subject",
        "predicate",
        "object_or_value",
        "time",
        "polarity",
        "negation_scope",
        "epistemic_status",
        "evidence_ids",
    }
    row = _strict_object(value, keys, context)
    row["source_observation_id"] = _identifier(
        row["source_observation_id"], f"{context}.source_observation_id"
    )
    row["subject"] = _validate_entity(row["subject"], f"{context}.subject")
    if row["subject"]["type"] != "finding":
        raise ValueError(f"{context}.subject must be a finding")
    row["predicate"] = _identifier(row["predicate"], f"{context}.predicate")
    if row["predicate"] not in _OBSERVATION_PREDICATES:
        raise ValueError(f"{context}.predicate is not an EEG observation predicate")
    row["object_or_value"] = _validate_claim_object(
        row["object_or_value"], f"{context}.object_or_value"
    )
    row["time"] = _validate_claim_time(row["time"], f"{context}.time")
    if row["polarity"] not in {"affirmed", "negated"}:
        raise ValueError(f"{context}.polarity is invalid")
    if row["negation_scope"] not in {
        "none",
        "predicate",
        "object_or_value",
        "full_claim",
    }:
        raise ValueError(f"{context}.negation_scope is invalid")
    if (row["polarity"] == "affirmed") != (row["negation_scope"] == "none"):
        raise ValueError(f"{context} polarity/negation scope conflict")
    if row["epistemic_status"] not in _OBSERVATION_EPISTEMIC_STATUSES:
        raise ValueError(f"{context}.epistemic_status is invalid")
    row["evidence_ids"] = _unique_identifiers(
        row["evidence_ids"], f"{context}.evidence_ids", allow_empty=False
    )
    return row


def validate_frozen_evidence_graph_projection(value: object) -> dict[str, Any]:
    """Validate the no-I/O EvidenceGraph projection consumed by this bridge."""

    data = _strict_object(
        value,
        {"schema_version", "record_id", "signal_sha256", "source_firewall", "events"},
        "frozen evidence graph projection",
    )
    if data["schema_version"] != FROZEN_EVIDENCE_GRAPH_PROJECTION_SCHEMA_VERSION:
        raise ValueError("frozen evidence graph projection schema_version mismatch")
    data["record_id"] = _identifier(data["record_id"], "frozen evidence graph.record_id")
    data["signal_sha256"] = _sha256(
        data["signal_sha256"], "frozen evidence graph.signal_sha256"
    )
    data["source_firewall"] = _validate_firewall(
        data["source_firewall"], "frozen evidence graph.source_firewall"
    )
    if not isinstance(data["events"], list) or not data["events"]:
        raise ValueError("frozen evidence graph.events must be a non-empty list")
    event_keys = {
        "event_id",
        "detector_candidate_id",
        "adaptive_window_id",
        "term_decision_source_binding_sha256",
        "analysis_interval",
        "onset_interval",
        "evidence",
        "observation_claims",
    }
    events: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    evidence_ids: set[str] = set()
    finding_ids: set[str] = set()
    source_observation_ids: set[str] = set()
    for index, value_event in enumerate(data["events"]):
        context = f"frozen evidence graph.events[{index}]"
        event = _strict_object(value_event, event_keys, context)
        event["event_id"] = _identifier(event["event_id"], f"{context}.event_id")
        if event["event_id"] in event_ids:
            raise ValueError("frozen evidence graph contains duplicate event IDs")
        event_ids.add(event["event_id"])
        event["detector_candidate_id"] = _optional_identifier(
            event["detector_candidate_id"], f"{context}.detector_candidate_id"
        )
        event["adaptive_window_id"] = _optional_identifier(
            event["adaptive_window_id"], f"{context}.adaptive_window_id"
        )
        event["term_decision_source_binding_sha256"] = _sha256(
            event["term_decision_source_binding_sha256"],
            f"{context}.term_decision_source_binding_sha256",
        )
        for interval_name in ("analysis_interval", "onset_interval"):
            interval = event[interval_name]
            event[interval_name] = (
                None
                if interval is None
                else _validate_interval(interval, f"{context}.{interval_name}")
            )
        if not isinstance(event["evidence"], list):
            raise TypeError(f"{context}.evidence must be a list")
        event["evidence"] = [
            _validate_source_evidence(item, f"{context}.evidence[{position}]")
            for position, item in enumerate(event["evidence"])
        ]
        for evidence in event["evidence"]:
            evidence_id = str(evidence["evidence_id"])
            finding_id = str(evidence["finding_id"])
            if evidence_id in evidence_ids or finding_id in finding_ids:
                raise ValueError(
                    "frozen evidence graph evidence_id/finding_id must be globally unique"
                )
            evidence_ids.add(evidence_id)
            finding_ids.add(finding_id)
        if not isinstance(event["observation_claims"], list):
            raise TypeError(f"{context}.observation_claims must be a list")
        event["observation_claims"] = [
            _validate_source_observation(
                item, f"{context}.observation_claims[{position}]"
            )
            for position, item in enumerate(event["observation_claims"])
        ]
        local_evidence_ids = {
            str(item["evidence_id"]) for item in event["evidence"]
        }
        for observation in event["observation_claims"]:
            observation_id = str(observation["source_observation_id"])
            if observation_id in source_observation_ids:
                raise ValueError(
                    "frozen evidence graph contains duplicate source observation IDs"
                )
            source_observation_ids.add(observation_id)
            if not set(observation["evidence_ids"]).issubset(local_evidence_ids):
                raise ValueError(
                    f"{context} source observation cites evidence outside its event"
                )
            finding_set = {
                str(
                    next(
                        item
                        for item in event["evidence"]
                        if item["evidence_id"] == evidence_id
                    )["finding_id"]
                )
                for evidence_id in observation["evidence_ids"]
            }
            if finding_set != {str(observation["subject"]["id"])}:
                raise ValueError(
                    f"{context} source observation subject does not close its Findings"
                )
        if event["detector_candidate_id"] is None and event["adaptive_window_id"] is not None:
            raise ValueError(f"{context} cannot retain a window before detector recovery")
        if event["adaptive_window_id"] is None:
            if (
                event["analysis_interval"] is not None
                or event["onset_interval"] is not None
                or event["evidence"]
                or event["observation_claims"]
            ):
                raise ValueError(
                    f"{context} without an adaptive window cannot emit intervals or Findings"
                )
        else:
            if event["analysis_interval"] is None or event["onset_interval"] is None:
                raise ValueError(f"{context} retained window requires analysis/onset intervals")
            analysis = event["analysis_interval"]
            onset = event["onset_interval"]
            if (
                float(onset["lower"]) < float(analysis["lower"])
                or float(onset["upper"]) > float(analysis["upper"])
            ):
                raise ValueError(f"{context}.onset_interval lies outside analysis_interval")
        events.append(event)
    data["events"] = events
    return data


def validate_complete_event_roster(value: object) -> dict[str, Any]:
    """Validate a frozen, source-only record roster with no stage booleans."""

    data = _strict_object(
        value,
        {
            "schema_version",
            "patient_id",
            "record_id",
            "signal_sha256",
            "event_ids",
            "source_firewall",
        },
        "complete event roster",
    )
    if data["schema_version"] != COMPLETE_EVENT_ROSTER_SCHEMA_VERSION:
        raise ValueError("complete event roster schema_version mismatch")
    for key in ("patient_id", "record_id"):
        data[key] = _identifier(data[key], f"complete event roster.{key}")
    data["signal_sha256"] = _sha256(
        data["signal_sha256"], "complete event roster.signal_sha256"
    )
    data["event_ids"] = _unique_identifiers(
        data["event_ids"], "complete event roster.event_ids", allow_empty=False
    )
    data["source_firewall"] = _validate_firewall(
        data["source_firewall"], "complete event roster.source_firewall"
    )
    return data


def frozen_evidence_event_sha256(value: object) -> str:
    """Return the canonical digest used by ``event_bundle_sha256`` bindings."""

    wrapper = {
        "schema_version": FROZEN_EVIDENCE_GRAPH_PROJECTION_SCHEMA_VERSION,
        "record_id": "DIGEST-ONLY",
        "signal_sha256": "0" * 64,
        "source_firewall": dict(_FIREWALL),
        "events": [deepcopy(value)],
    }
    validated = validate_frozen_evidence_graph_projection(wrapper)
    return _canonical_sha256(validated["events"][0])


def _source_temporal_binding(
    evidence: Mapping[str, Any], *, claim_evidence_role: str
) -> dict[str, Any]:
    return {
        "evidence_id": str(evidence["evidence_id"]),
        "evidence_role": claim_evidence_role,
        "intrinsic_evidence_role": str(evidence["intrinsic_evidence_role"]),
        "view_role": str(evidence["view_role"]),
        "future_sample_access": bool(evidence["future_sample_access"]),
        "onset_evidence_authorized": bool(evidence["onset_evidence_authorized"]),
        "onset_support_eligible": bool(evidence["onset_support_eligible"]),
    }


def _aggregate_assertion_status(
    claim: Mapping[str, Any], source_evidence: Sequence[Mapping[str, Any]]
) -> str:
    if claim["claim_kind"] != "observation":
        return "present"
    if not source_evidence:
        return "not_evaluable" if claim["epistemic_status"] == "not_evaluable" else "present"
    states = {str(item["status"]) for item in source_evidence}
    if len(states) == 1:
        return next(iter(states))
    # Mixed opportunity/assertion states do not authorize a stronger union.
    return "uncertain"


def _relation_endpoints(
    claim: Mapping[str, Any],
    *,
    graph_claims: Sequence[Mapping[str, Any]],
) -> dict[str, str] | None:
    predicate = str(claim["predicate"])
    if predicate in {"supports_claim", "contradicts_claim"}:
        return {
            "source_claim_id": str(claim["subject"]["id"]),
            "target_claim_id": str(claim["object_or_value"]["entities"][0]["id"]),
        }
    if predicate not in _TEMPORAL_RELATION_PREDICATES:
        return None

    source_finding = str(claim["subject"]["id"])
    target_finding = str(claim["object_or_value"]["entities"][0]["id"])

    def owner(finding_id: str) -> str:
        candidates = [
            str(row["claim_id"])
            for row in graph_claims
            if row["claim_kind"] == "observation"
            and row["predicate"] not in _TEMPORAL_RELATION_PREDICATES
            and row["subject"]["type"] == "finding"
            and str(row["subject"]["id"]) == finding_id
        ]
        if len(candidates) != 1:
            raise ValueError(
                "temporal relation endpoint must resolve to exactly one atomic "
                f"observation claim; finding={finding_id!r}, candidates={candidates}"
            )
        return candidates[0]

    return {
        "source_claim_id": owner(source_finding),
        "target_claim_id": owner(target_finding),
    }


def _materialize_claim(
    claim: Mapping[str, Any],
    *,
    graph_claims: Sequence[Mapping[str, Any]],
    source_evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predicate = str(claim["predicate"])
    if predicate not in _CLAIM_WEIGHT_POLICY:
        raise ValueError(f"claim predicate {predicate!r} lacks a frozen weight policy")
    evidence_ids = [str(item) for item in claim["evidence_ids"]]
    missing = set(evidence_ids).difference(source_evidence_by_id)
    if missing:
        raise ValueError(
            f"claim {claim['claim_id']!r} cites evidence absent from the frozen "
            f"EvidenceGraph: {sorted(missing)}"
        )
    source_evidence = [source_evidence_by_id[item] for item in evidence_ids]
    assertion_status = _aggregate_assertion_status(claim, source_evidence)
    is_positive = (
        predicate in _POSITIVE_ONSET_OR_SOZ_PREDICATES
        and claim["polarity"] == "affirmed"
        and assertion_status in {"present", "uncertain"}
    )
    # ``evidence_catalog.evidence_role`` is relative to the selected
    # hypothesis.  A competing but causal onset field is therefore a
    # ``contradiction`` to one hypothesis while still being onset support for
    # its own atomic ``earliest_sustained_change`` observation.  Claim roles
    # are derived per edge instead of copying that hypothesis-relative label.
    def claim_role(item: Mapping[str, Any]) -> str:
        if predicate == "contradicts_claim":
            return "contradiction"
        if is_positive:
            return "onset_support"
        return str(item["evidence_role"])

    bindings = sorted(
        (
            _source_temporal_binding(
                item, claim_evidence_role=claim_role(item)
            )
            for item in source_evidence
        ),
        key=lambda row: row["evidence_id"],
    )
    roles = sorted({str(item["evidence_role"]) for item in bindings})
    if is_positive:
        if not bindings:
            raise ValueError(
                f"positive onset/SOZ claim {claim['claim_id']!r} lacks source evidence"
            )
        invalid = [
            row["evidence_id"]
            for row in bindings
            if not (
                row["evidence_role"] == "onset_support"
                and row["intrinsic_evidence_role"] == "onset_eligible"
                and row["view_role"] == "onset_causal"
                and not row["future_sample_access"]
                and row["onset_evidence_authorized"]
                and row["onset_support_eligible"]
            )
        ]
        if invalid:
            raise ValueError(
                f"positive onset/SOZ claim {claim['claim_id']!r} has incomplete "
                f"causal bindings: {sorted(invalid)}"
            )
    weights = _CLAIM_WEIGHT_POLICY[predicate]
    salience_weight = float(weights["salience"])
    return {
        "claim_id": str(claim["claim_id"]),
        "claim_kind": str(claim["claim_kind"]),
        "subject": deepcopy(claim["subject"]),
        "predicate": predicate,
        "object_or_value": deepcopy(claim["object_or_value"]),
        "event_id": claim["event_id"],
        "mode_id": claim["mode_id"],
        "time": deepcopy(claim["time"]),
        "polarity": str(claim["polarity"]),
        "negation_scope": str(claim["negation_scope"]),
        "assertion_status": assertion_status,
        "epistemic_status": str(claim["epistemic_status"]),
        "evidence_ids": evidence_ids,
        "evidence_roles": roles,
        "relation_endpoints": _relation_endpoints(claim, graph_claims=graph_claims),
        "salient": salience_weight > 0,
        "salience_weight": salience_weight,
        "severity_weight": float(weights["severity"]),
        "critical": bool(weights["critical"]),
        "evidence_temporal_bindings": bindings,
    }


def _direct_support_sources(
    claim: Mapping[str, Any], claim_by_id: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    sources: list[str] = []
    for relation_id in claim["supporting_relation_claim_ids"]:
        relation = claim_by_id[str(relation_id)]
        if relation["predicate"] != "supports_claim":
            raise ValueError("supporting_relation_claim_ids contains a non-support edge")
        source_id = str(relation["subject"]["id"])
        target_id = str(relation["object_or_value"]["entities"][0]["id"])
        if target_id != claim["claim_id"]:
            raise ValueError("support relation target conflicts with its conclusion")
        sources.append(source_id)
    if len(sources) != len(set(sources)):
        raise ValueError(f"claim {claim['claim_id']!r} has duplicate support premises")
    return sources


def _materialize_derivations(
    graph_claims: Sequence[Mapping[str, Any]], rendered_claim_ids: set[str]
) -> list[dict[str, Any]]:
    claim_by_id = {str(row["claim_id"]): row for row in graph_claims}
    rows: list[dict[str, Any]] = []
    for claim in graph_claims:
        claim_id = str(claim["claim_id"])
        kind = str(claim["claim_kind"])
        if claim_id not in rendered_claim_ids or kind not in {
            "event_inference",
            "mode_inference",
            "record_hypothesis",
        }:
            continue
        premise_ids = _direct_support_sources(claim, claim_by_id)
        if not premise_ids:
            raise ValueError(f"rendered inference {claim_id!r} lacks support premises")
        missing = set(premise_ids).difference(rendered_claim_ids)
        if missing:
            raise ValueError(
                f"rendered inference {claim_id!r} has premises absent from the "
                f"sentence ledger: {sorted(missing)}"
            )
        premise_kinds = {str(claim_by_id[item]["claim_kind"]) for item in premise_ids}
        predicate = str(claim["predicate"])
        if kind == "event_inference":
            rule_id = "event_observation_to_event_hypothesis_v1"
        elif kind == "mode_inference":
            rule_id = (
                "event_hypothesis_to_mode_hypothesis_v1"
                if premise_kinds == {"event_inference"}
                else "event_observation_to_mode_hypothesis_v1"
            )
        elif predicate == "record_has_generalized_synchronous_onset":
            rule_id = "bilateral_synchrony_to_generalized_record_v1"
        elif predicate in {"record_onset_nonlocalizable", "record_technical_limited"}:
            rule_id = "limitation_to_nonlocalizable_record_v1"
        elif predicate == "record_has_multiple_onset_modes":
            rule_id = "multiple_modes_to_record_hypothesis_v1"
        else:
            rule_id = (
                "mode_hypothesis_to_record_hypothesis_v1"
                if premise_kinds == {"mode_inference"}
                else "onset_evidence_to_record_hypothesis_v1"
            )
        rows.append(
            {
                "derivation_id": f"DERIVE:{claim_id}",
                "conclusion_claim_id": claim_id,
                "premise_claim_ids": premise_ids,
                "rule_id": rule_id,
                "weight": float(_CLAIM_WEIGHT_POLICY[predicate]["severity"]),
            }
        )
    return rows


def _atomic_clause_rows(
    render: Mapping[str, Any], graph_claim_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ledger_row in render["sentence_ledger"]:
        sentence_id = str(ledger_row["sentence_id"])
        text = str(ledger_row["text_zh"])
        claim_ids = [str(item) for item in ledger_row["claim_ids"]]
        template = str(ledger_row["template_id"])
        clause_specs: list[tuple[str, list[str]]]
        if template == "event_competing_onset_fields_v1":
            fragments = text[:-1].split("；") if text.endswith("。") else text.split("；")
            if len(fragments) != len(claim_ids):
                raise ValueError(
                    "competing-onset sentence cannot be atomically replayed to its claims"
                )
            clause_specs = [(fragment, [claim_id]) for fragment, claim_id in zip(fragments, claim_ids)]
        else:
            contradiction_ids = [
                claim_id
                for claim_id in claim_ids
                if graph_claim_by_id[claim_id]["predicate"] == "contradicts_claim"
            ]
            marker = "；同时存在 "
            if contradiction_ids:
                if marker not in text:
                    raise ValueError(
                        "counterevidence claims lack a distinct deterministic surface clause"
                    )
                base, suffix = text.split(marker, 1)
                base_ids = [item for item in claim_ids if item not in contradiction_ids]
                if not base_ids:
                    raise ValueError("counterevidence sentence lacks a non-counterevidence clause")
                clause_specs = [(base, base_ids), ("同时存在 " + suffix, contradiction_ids)]
            else:
                clause_specs = [(text, claim_ids)]
        for ordinal, (fragment, owned_claim_ids) in enumerate(clause_specs, start=1):
            if not fragment or not owned_claim_ids:
                raise ValueError("every non-format atomic clause requires claim ownership")
            rows.append(
                {
                    "clause_id": f"CLAUSE:{sentence_id}:{ordinal}",
                    "sentence_id": sentence_id,
                    "ordinal": ordinal,
                    "surface_fragment_zh": fragment,
                    "surface_fragment_sha256": _canonical_sha256(fragment),
                    "claim_ids": owned_claim_ids,
                }
            )
    ownership = Counter(
        claim_id for row in rows for claim_id in row["claim_ids"]
    )
    rendered_ids = {
        str(claim_id)
        for row in render["sentence_ledger"]
        for claim_id in row["claim_ids"]
    }
    if set(ownership) != rendered_ids or any(count != 1 for count in ownership.values()):
        raise ValueError(
            "every rendered claim must own exactly one non-format atomic clause"
        )
    return rows


def _bounded_flow_id(event_id: str) -> str:
    raw = f"FLOW-EVENT:{event_id}"
    if len(raw) <= 191:
        return raw
    return f"FLOW-EVENT:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _materialize_evidence_flow(
    *,
    evidence_graph: Mapping[str, Any],
    graph_claims: Sequence[Mapping[str, Any]],
    predicted_claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    graph_claim_evidence = {
        str(item) for claim in graph_claims for item in claim["evidence_ids"]
    }
    rendered_claim_evidence = {
        str(item) for claim in predicted_claims for item in claim["evidence_ids"]
    }
    rows: list[dict[str, Any]] = []
    for event in evidence_graph["events"]:
        detector_recovered = event["detector_candidate_id"] is not None
        window_retained = event["adaptive_window_id"] is not None
        if event["evidence"]:
            for evidence in event["evidence"]:
                evidence_id = str(evidence["evidence_id"])
                rows.append(
                    {
                        "evidence_id": evidence_id,
                        "weight": float(
                            _EVIDENCE_FLOW_WEIGHTS[str(evidence["evidence_role"])]
                        ),
                        "detector_recovered": detector_recovered,
                        "adaptive_window_retained": window_retained,
                        "finding_emitted": True,
                        "record_claim_retained": evidence_id in graph_claim_evidence,
                        "rendered_claim_retained": evidence_id in rendered_claim_evidence,
                    }
                )
        else:
            rows.append(
                {
                    "evidence_id": _bounded_flow_id(str(event["event_id"])),
                    "weight": float(_EVIDENCE_FLOW_WEIGHTS["event_without_finding"]),
                    "detector_recovered": detector_recovered,
                    "adaptive_window_retained": window_retained,
                    "finding_emitted": False,
                    "record_claim_retained": False,
                    "rendered_claim_retained": False,
                }
            )
    return rows


def _validate_source_alignment(
    *,
    evidence_graph: Mapping[str, Any],
    record_graph: Mapping[str, Any],
    event_roster: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    record_id = str(record_graph["record_id"])
    signal_sha256 = str(record_graph["provenance"]["signal_sha256"])
    for source_name, source in (
        ("frozen EvidenceGraph", evidence_graph),
        ("complete event roster", event_roster),
    ):
        if source["record_id"] != record_id or source["signal_sha256"] != signal_sha256:
            raise ValueError(f"{source_name} record/signal identity differs from report graph")
        if source["source_firewall"] != record_graph["provenance"]["inference_exclusions"]:
            raise ValueError(f"{source_name} firewall differs from report graph")
    source_event_ids = [str(row["event_id"]) for row in evidence_graph["events"]]
    if source_event_ids != list(event_roster["event_ids"]):
        raise ValueError(
            "complete event roster must exactly equal the ordered frozen EvidenceGraph roster"
        )
    source_event_by_id = {str(row["event_id"]): row for row in evidence_graph["events"]}
    record_event_ids = [str(row["event_id"]) for row in record_graph["events"]]
    if not set(record_event_ids).issubset(source_event_by_id):
        raise ValueError("record graph contains events outside the complete source roster")

    source_evidence_by_id = {
        str(row["evidence_id"]): row
        for event in evidence_graph["events"]
        for row in event["evidence"]
    }
    record_evidence_by_id = {
        str(row["evidence_id"]): row for row in record_graph["evidence_catalog"]
    }
    if not set(record_evidence_by_id).issubset(source_evidence_by_id):
        missing = set(record_evidence_by_id).difference(source_evidence_by_id)
        raise ValueError(
            f"record graph cites evidence absent from frozen EvidenceGraph: {sorted(missing)}"
        )

    for record_event in record_graph["events"]:
        event_id = str(record_event["event_id"])
        source_event = source_event_by_id[event_id]
        if source_event["adaptive_window_id"] is None:
            raise ValueError(f"record graph retained event {event_id!r} without a source window")
        if record_event["event_bundle_sha256"] != _canonical_sha256(source_event):
            raise ValueError(f"record event {event_id!r} source EvidenceGraph hash drifted")
        if (
            record_event["term_decision_source_binding_sha256"]
            != source_event["term_decision_source_binding_sha256"]
        ):
            raise ValueError(f"record event {event_id!r} term-decision binding drifted")
        for key in ("analysis_interval", "onset_interval"):
            if _canonical_json(record_event[key]) != _canonical_json(source_event[key]):
                raise ValueError(f"record event {event_id!r} {key} differs from source")

    projection_keys = (
        "evidence_id",
        "finding_id",
        "family",
        "term",
        "evidence_role",
        "status",
        "assertion_level",
        "waveform_evidence_ids",
    )
    for evidence_id, record_evidence in record_evidence_by_id.items():
        source_evidence = source_evidence_by_id[evidence_id]
        for key in projection_keys:
            if _canonical_json(record_evidence[key]) != _canonical_json(source_evidence[key]):
                raise ValueError(
                    f"record evidence {evidence_id!r}.{key} differs from frozen EvidenceGraph"
                )

    # Direct observation semantics are independently frozen by the
    # EvidenceGraph.  Evidence IDs therefore cannot be retained while the
    # record graph silently changes a predicate, spatial entity, measurement
    # or physical-time interval.
    source_observations_by_event = {
        str(event["event_id"]): list(event["observation_claims"])
        for event in evidence_graph["events"]
    }
    used_source_observation_ids: set[str] = set()
    semantic_keys = (
        "subject",
        "predicate",
        "object_or_value",
        "time",
        "polarity",
        "negation_scope",
        "epistemic_status",
        "evidence_ids",
    )
    for claim in record_graph["claims"]:
        if claim["claim_kind"] != "observation" or not claim["evidence_ids"]:
            continue
        event_id = str(claim["event_id"])
        candidate_projection_row = _validate_source_observation(
            {
                "source_observation_id": "COMPARE-ONLY",
                **{key: deepcopy(claim[key]) for key in semantic_keys},
            },
            f"record observation {claim['claim_id']!r} comparison projection",
        )
        candidate_projection = {
            key: candidate_projection_row[key] for key in semantic_keys
        }
        matches = [
            row
            for row in source_observations_by_event.get(event_id, [])
            if _canonical_json({key: row[key] for key in semantic_keys})
            == _canonical_json(candidate_projection)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"record observation {claim['claim_id']!r} differs from its frozen "
                f"EvidenceGraph atomic projection (matches={len(matches)})"
            )
        source_observation_id = str(matches[0]["source_observation_id"])
        if source_observation_id in used_source_observation_ids:
            raise ValueError(
                "multiple record observations consume one frozen EvidenceGraph atom"
            )
        used_source_observation_ids.add(source_observation_id)
    return source_evidence_by_id


def _trusted_kwargs(
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_capability_qualification_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    return {
        "trusted_producer_receipts": trusted_producer_receipts,
        "trusted_calibration_receipts": trusted_calibration_receipts,
        "trusted_capability_qualification_receipts": (
            trusted_capability_qualification_receipts
        ),
        "trusted_term_decision_receipts": trusted_term_decision_receipts,
    }


def _artifact_shape(value: object) -> dict[str, Any]:
    keys = {
        "schema_version",
        "materializer_id",
        "weight_policy_id",
        "weight_policy_sha256",
        "source_bindings",
        "coverage_receipt",
        "case",
        "materialization_sha256",
    }
    data = _strict_object(value, keys, "source-bound factuality materialization")
    if data["schema_version"] != SOURCE_BOUND_FACTUALITY_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("source-bound factuality materialization schema_version mismatch")
    if data["materializer_id"] != SOURCE_BOUND_FACTUALITY_MATERIALIZER_ID:
        raise ValueError("source-bound factuality materializer_id mismatch")
    if data["weight_policy_id"] != FACTUALITY_WEIGHT_POLICY_ID:
        raise ValueError("source-bound factuality weight_policy_id mismatch")
    if data["weight_policy_sha256"] != FACTUALITY_WEIGHT_POLICY_SHA256:
        raise ValueError("source-bound factuality weight policy hash drifted")
    source = _strict_object(
        data["source_bindings"],
        {
            "frozen_evidence_graph_sha256",
            "record_hypothesis_graph_sha256",
            "report_render_sha256",
            "report_self_sha256",
            "sentence_ledger_sha256",
            "complete_event_roster_sha256",
            "source_bundle_sha256",
            "event_graph_sha256s",
        },
        "source-bound factuality source_bindings",
    )
    for key in (
        "frozen_evidence_graph_sha256",
        "record_hypothesis_graph_sha256",
        "report_render_sha256",
        "report_self_sha256",
        "sentence_ledger_sha256",
        "complete_event_roster_sha256",
        "source_bundle_sha256",
    ):
        source[key] = _sha256(source[key], f"source_bindings.{key}")
    if not isinstance(source["event_graph_sha256s"], list):
        raise TypeError("source_bindings.event_graph_sha256s must be a list")
    event_rows: list[dict[str, str]] = []
    event_ids: set[str] = set()
    for index, item in enumerate(source["event_graph_sha256s"]):
        row = _strict_object(
            item, {"event_id", "sha256"}, f"source_bindings.event_graph_sha256s[{index}]"
        )
        row["event_id"] = _identifier(row["event_id"], f"event_graph_sha256s[{index}].event_id")
        row["sha256"] = _sha256(row["sha256"], f"event_graph_sha256s[{index}].sha256")
        if row["event_id"] in event_ids:
            raise ValueError("source_bindings contains duplicate event hashes")
        event_ids.add(row["event_id"])
        event_rows.append(row)
    source["event_graph_sha256s"] = event_rows
    data["source_bindings"] = source
    data["case"] = validate_claim_factuality_case(data["case"])

    coverage = _strict_object(
        data["coverage_receipt"],
        {
            "sentence_count",
            "atomic_clause_count",
            "source_claim_count",
            "rendered_claim_count",
            "mandatory_source_claim_count",
            "sentence_ledger_sha256",
            "atomic_clause_ledger_sha256",
            "atomic_clauses",
            "source_render_replayed_exactly",
            "all_nonformat_sentences_claim_owned",
            "rendered_claim_ownership_unique",
        },
        "source-bound factuality coverage_receipt",
    )
    for key in (
        "sentence_count",
        "atomic_clause_count",
        "source_claim_count",
        "rendered_claim_count",
        "mandatory_source_claim_count",
    ):
        value_number = _finite_nonnegative(coverage[key], f"coverage_receipt.{key}")
        if not value_number.is_integer():
            raise ValueError(f"coverage_receipt.{key} must be an integer")
        coverage[key] = int(value_number)
    for key in (
        "source_render_replayed_exactly",
        "all_nonformat_sentences_claim_owned",
        "rendered_claim_ownership_unique",
    ):
        if coverage[key] is not True:
            raise ValueError(f"coverage_receipt.{key} is a fail-closed invariant")
    for key in ("sentence_ledger_sha256", "atomic_clause_ledger_sha256"):
        coverage[key] = _sha256(coverage[key], f"coverage_receipt.{key}")
    if not isinstance(coverage["atomic_clauses"], list) or not coverage["atomic_clauses"]:
        raise ValueError("coverage_receipt.atomic_clauses must be non-empty")
    clause_keys = {
        "clause_id",
        "sentence_id",
        "ordinal",
        "surface_fragment_zh",
        "surface_fragment_sha256",
        "claim_ids",
    }
    clauses: list[dict[str, Any]] = []
    clause_ids: set[str] = set()
    for index, item in enumerate(coverage["atomic_clauses"]):
        row = _strict_object(item, clause_keys, f"coverage_receipt.atomic_clauses[{index}]")
        row["clause_id"] = _identifier(row["clause_id"], f"atomic_clauses[{index}].clause_id")
        row["sentence_id"] = _identifier(row["sentence_id"], f"atomic_clauses[{index}].sentence_id")
        if row["clause_id"] in clause_ids:
            raise ValueError("coverage receipt contains duplicate clause IDs")
        clause_ids.add(row["clause_id"])
        ordinal = _finite_nonnegative(row["ordinal"], f"atomic_clauses[{index}].ordinal", positive=True)
        if not ordinal.is_integer():
            raise ValueError("atomic clause ordinal must be an integer")
        row["ordinal"] = int(ordinal)
        if not isinstance(row["surface_fragment_zh"], str) or not row["surface_fragment_zh"]:
            raise ValueError("atomic clause surface fragment must be non-empty")
        row["surface_fragment_sha256"] = _sha256(
            row["surface_fragment_sha256"], f"atomic_clauses[{index}].surface_fragment_sha256"
        )
        if row["surface_fragment_sha256"] != _canonical_sha256(row["surface_fragment_zh"]):
            raise ValueError("atomic clause surface hash drifted")
        row["claim_ids"] = _unique_identifiers(
            row["claim_ids"], f"atomic_clauses[{index}].claim_ids", allow_empty=False
        )
        clauses.append(row)
    coverage["atomic_clauses"] = clauses
    if coverage["atomic_clause_count"] != len(clauses):
        raise ValueError("coverage_receipt.atomic_clause_count mismatch")
    if coverage["atomic_clause_ledger_sha256"] != _canonical_sha256(clauses):
        raise ValueError("coverage receipt atomic-clause ledger hash drifted")
    if coverage["sentence_ledger_sha256"] != source["sentence_ledger_sha256"]:
        raise ValueError("coverage/source sentence-ledger hashes differ")
    ownership = Counter(claim_id for row in clauses for claim_id in row["claim_ids"])
    predicted_ids = {str(row["claim_id"]) for row in data["case"]["predicted_claims"]}
    if set(ownership) != predicted_ids or any(count != 1 for count in ownership.values()):
        raise ValueError("coverage receipt does not uniquely own every predicted claim")
    if coverage["rendered_claim_count"] != len(predicted_ids):
        raise ValueError("coverage_receipt.rendered_claim_count mismatch")
    if coverage["source_claim_count"] != len(data["case"]["reference_claims"]):
        raise ValueError("coverage_receipt.source_claim_count mismatch")
    data["coverage_receipt"] = coverage
    data["materialization_sha256"] = _sha256(
        data["materialization_sha256"], "source-bound factuality materialization_sha256"
    )
    digest_source = deepcopy(data)
    digest_source["materialization_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["materialization_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("source-bound factuality materialization hash drifted")
    return data


def materialize_source_bound_factuality_case(
    *,
    frozen_evidence_graph: object,
    record_hypothesis_graph: object,
    report_render: object,
    complete_event_roster: object,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Materialize a factuality case exclusively from four frozen sources."""

    evidence_graph = validate_frozen_evidence_graph_projection(frozen_evidence_graph)
    roster = validate_complete_event_roster(complete_event_roster)
    trusted = _trusted_kwargs(
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_term_decision_receipts=trusted_term_decision_receipts,
    )
    graph = validate_multievent_soz_report_payload(record_hypothesis_graph, **trusted)
    persisted_render = validate_multievent_report_render(report_render)
    expected_render = render_multievent_soz_report_zh(graph, **trusted)
    if _canonical_json(persisted_render) != _canonical_json(expected_render):
        raise ValueError(
            "persisted report/sentence ledger does not replay exactly from the frozen source graph"
        )

    source_evidence_by_id = _validate_source_alignment(
        evidence_graph=evidence_graph,
        record_graph=graph,
        event_roster=roster,
    )
    graph_claims = list(graph["claims"])
    graph_claim_by_id = {str(row["claim_id"]): row for row in graph_claims}
    reference_claims = [
        _materialize_claim(
            row,
            graph_claims=graph_claims,
            source_evidence_by_id=source_evidence_by_id,
        )
        for row in graph_claims
    ]
    rendered_claim_ids_order = [
        str(claim_id)
        for ledger_row in persisted_render["sentence_ledger"]
        for claim_id in ledger_row["claim_ids"]
    ]
    if len(rendered_claim_ids_order) != len(set(rendered_claim_ids_order)):
        raise ValueError("sentence ledger duplicates a rendered claim")
    rendered_claim_ids = set(rendered_claim_ids_order)
    predicted_claims = [
        deepcopy(next(row for row in reference_claims if row["claim_id"] == claim_id))
        for claim_id in rendered_claim_ids_order
    ]
    # The portable evaluator requires relation endpoint closure inside each
    # claim set.  A sentence may not surface a relation while omitting either
    # endpoint claim.
    for row in predicted_claims:
        relation = row["relation_endpoints"]
        if relation is not None and not {
            relation["source_claim_id"],
            relation["target_claim_id"],
        }.issubset(rendered_claim_ids):
            raise ValueError(
                f"rendered relation {row['claim_id']!r} omits a typed endpoint claim"
            )

    derivations = _materialize_derivations(graph_claims, rendered_claim_ids)
    evidence_flow = _materialize_evidence_flow(
        evidence_graph=evidence_graph,
        graph_claims=graph_claims,
        predicted_claims=predicted_claims,
    )
    clauses = _atomic_clause_rows(persisted_render, graph_claim_by_id)

    base_source_hashes = {
        "frozen_evidence_graph_sha256": _canonical_sha256(evidence_graph),
        "record_hypothesis_graph_sha256": _canonical_sha256(graph),
        "report_render_sha256": _canonical_sha256(persisted_render),
        "report_self_sha256": str(persisted_render["report_sha256"]),
        "sentence_ledger_sha256": _canonical_sha256(persisted_render["sentence_ledger"]),
        "complete_event_roster_sha256": _canonical_sha256(roster),
    }
    source_bundle_sha256 = _canonical_sha256(
        {"domain": "eeg-source-bound-factuality-sources-v1", **base_source_hashes}
    )
    source_bindings = {
        **base_source_hashes,
        "source_bundle_sha256": source_bundle_sha256,
        "event_graph_sha256s": [
            {"event_id": str(row["event_id"]), "sha256": _canonical_sha256(row)}
            for row in evidence_graph["events"]
        ],
    }
    case_id = f"CASE:{source_bundle_sha256[:32]}"
    case = validate_claim_factuality_case(
        {
            "schema_version": CLAIM_FACTUALITY_CASE_SCHEMA_VERSION,
            "case_id": case_id,
            "patient_id": str(roster["patient_id"]),
            "record_id": str(roster["record_id"]),
            "predicted_claims": predicted_claims,
            "reference_claims": reference_claims,
            "derivations": derivations,
            "evidence_flow": evidence_flow,
            "claim_boundary": dict(_CLAIM_BOUNDARY),
        }
    )
    coverage = {
        "sentence_count": len(persisted_render["sentence_ledger"]),
        "atomic_clause_count": len(clauses),
        "source_claim_count": len(reference_claims),
        "rendered_claim_count": len(predicted_claims),
        "mandatory_source_claim_count": sum(
            bool(row["mandatory_for_report"]) for row in graph_claims
        ),
        "sentence_ledger_sha256": source_bindings["sentence_ledger_sha256"],
        "atomic_clause_ledger_sha256": _canonical_sha256(clauses),
        "atomic_clauses": clauses,
        "source_render_replayed_exactly": True,
        "all_nonformat_sentences_claim_owned": True,
        "rendered_claim_ownership_unique": True,
    }
    body: dict[str, Any] = {
        "schema_version": SOURCE_BOUND_FACTUALITY_MATERIALIZATION_SCHEMA_VERSION,
        "materializer_id": SOURCE_BOUND_FACTUALITY_MATERIALIZER_ID,
        "weight_policy_id": FACTUALITY_WEIGHT_POLICY_ID,
        "weight_policy_sha256": FACTUALITY_WEIGHT_POLICY_SHA256,
        "source_bindings": source_bindings,
        "coverage_receipt": coverage,
        "case": case,
        "materialization_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["materialization_sha256"] = _canonical_sha256(body)
    return _artifact_shape(body)


def validate_source_bound_factuality_case_materialization(
    value: object,
    *,
    frozen_evidence_graph: object,
    record_hypothesis_graph: object,
    report_render: object,
    complete_event_roster: object,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate an artifact by replaying every frozen source out of band."""

    data = _artifact_shape(value)
    expected = materialize_source_bound_factuality_case(
        frozen_evidence_graph=frozen_evidence_graph,
        record_hypothesis_graph=record_hypothesis_graph,
        report_render=report_render,
        complete_event_roster=complete_event_roster,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_term_decision_receipts=trusted_term_decision_receipts,
    )
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError(
            "source-bound factuality materialization differs from frozen-source replay"
        )
    return data


def evaluate_source_bound_factuality_case_materialization(
    value: object,
    *,
    frozen_evidence_graph: object,
    record_hypothesis_graph: object,
    report_render: object,
    complete_event_roster: object,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    policy: object | None = None,
) -> dict[str, Any]:
    """Replay a source-bound artifact, then run the independent evaluator."""

    validated = validate_source_bound_factuality_case_materialization(
        value,
        frozen_evidence_graph=frozen_evidence_graph,
        record_hypothesis_graph=record_hypothesis_graph,
        report_render=report_render,
        complete_event_roster=complete_event_roster,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_term_decision_receipts=trusted_term_decision_receipts,
    )
    return evaluate_claim_factuality_case(validated["case"], policy=policy)


__all__ = [
    "COMPLETE_EVENT_ROSTER_SCHEMA_VERSION",
    "FACTUALITY_WEIGHT_POLICY_ID",
    "FACTUALITY_WEIGHT_POLICY_SHA256",
    "FROZEN_EVIDENCE_GRAPH_PROJECTION_SCHEMA_VERSION",
    "SOURCE_BOUND_FACTUALITY_MATERIALIZATION_SCHEMA_VERSION",
    "SOURCE_BOUND_FACTUALITY_MATERIALIZER_ID",
    "evaluate_source_bound_factuality_case_materialization",
    "frozen_evidence_event_sha256",
    "materialize_source_bound_factuality_case",
    "validate_complete_event_roster",
    "validate_frozen_evidence_graph_projection",
    "validate_source_bound_factuality_case_materialization",
]
