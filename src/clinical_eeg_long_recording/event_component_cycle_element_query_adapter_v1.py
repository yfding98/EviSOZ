"""Replayable adapter from the native S06 ledgers to three term queries.

The adapter replaces exactly the S06 queries in the legacy ten-query
Findings-v1 closure.  It does not change S04/S05/S09/S12, onset ranking, or
the report allowlist.  Every trajectory and research-candidate instance is
rebuilt from an embedded, validated component/cycle/element ledger receipt.

The engineering rhythmic-run rule is deliberately not a clinical criterion.
Failure to satisfy it yields ``uncertain`` (when cycle opportunity exists) or
``not_evaluable``; it never yields absence.  All outputs remain ineligible for
onset/SOZ support and report promotion.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

from .event_component_cycle_element_ledger_v1 import (
    validate_event_component_cycle_element_ledger_v1,
)


EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_event_component_cycle_element_query_adapter_v1"
)
EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_METHOD_ID: Final[str] = (
    "EVENT-COMPONENT-CYCLE-ELEMENT-QUERY-ADAPTER-V1"
)
S06_REQUIRED_QUERY_IDS: Final[tuple[str, str, str]] = (
    "TQ-EVENT-RHYTHMICITY-COURSE",
    "TQ-PERIODIC-ELEMENT-INSTANCE",
    "TQ-RHYTHMIC-RUN-INSTANCE",
)
_QUERY_SPECS: Final[dict[str, tuple[str, str, str]]] = {
    "TQ-EVENT-RHYTHMICITY-COURSE": (
        "event_rhythmicity_course_profile",
        "measurement",
        "rhythm",
    ),
    "TQ-PERIODIC-ELEMENT-INSTANCE": (
        "periodic_element_candidate",
        "instance",
        "rhythm",
    ),
    "TQ-RHYTHMIC-RUN-INSTANCE": (
        "rhythmic_run_candidate",
        "instance",
        "rhythm",
    ),
}
_RUN_POLICY: Final[dict[str, object]] = {
    "policy_id": "S06-LEDGER-RHYTHMIC-RUN-PROJECTION-POLICY-V1",
    "minimum_rhythmic_run_elements": 4,
    "maximum_adjacent_cycle_ratio": 1.35,
    "maximum_run_robust_cv": 0.30,
    "threshold_semantics": "engineering_research_candidate_only",
    "clinical_thresholds_defined": False,
    "spectral_peak_authorized": False,
    "autocorrelation_peak_authorized": False,
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL: Final[float] = 1e-9

_FIREWALL: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "allowlisted_acquisition_metadata_used": True,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_or_report_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}
_AUTHORIZATION: Final[dict[str, bool | str | list[str]]] = {
    "event_card_slot_id": "S06_RHYTHMICITY_PERIODICITY",
    "projection_scope": "native_s06_measurement_and_research_candidate_only",
    "whole_bipolar_lead_identity_required": True,
    "bipolar_endpoint_fact_projection_authorized": False,
    "clinical_term_qualification_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
    "report_eligible_term_allowlist": [],
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _identifier(value: object, context: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _sha(value: object, context: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _reject_nonfinite(value: object, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{context}[{index}]")


def _sorted_reasons(values: Sequence[object]) -> list[str]:
    result = sorted({str(item) for item in values})
    if any(not item or item != item.strip() for item in result):
        raise ValueError("reason codes must be non-empty and trimmed")
    return result


def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("median/MAD requires at least one value")
    ordered = sorted(float(item) for item in values)

    def median(items: Sequence[float]) -> float:
        middle = len(items) // 2
        if len(items) % 2:
            return float(items[middle])
        return float((items[middle - 1] + items[middle]) / 2.0)

    center = median(ordered)
    return center, median(sorted(abs(item - center) for item in ordered))


def _raw_support_by_element(
    source: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for element in source["element_instance_ledger"]["instances"]:
        element_id = str(element["element_instance_id"])
        dependency = deepcopy(element["raw_sample_dependency"])
        if dependency["dependency_sha256"] != element[
            "raw_sample_dependency_sha256"
        ]:
            raise ValueError("S06 element raw dependency binding drifted")
        result[element_id] = dependency
    return result


def _query_opportunity(
    ledger: Mapping[str, Any],
    *,
    has_output: bool,
    uncertain_with_evidence: bool = False,
) -> tuple[str, str, list[str]]:
    opportunity = ledger["opportunity"]
    reasons = list(opportunity["reason_codes"])
    if has_output:
        return (
            "candidate_only" if uncertain_with_evidence else "measured",
            str(opportunity["status"]),
            _sorted_reasons(reasons),
        )
    if uncertain_with_evidence:
        return (
            "uncertain",
            "limited",
            _sorted_reasons(
                [
                    *reasons,
                    "engineering_rhythmic_run_rule_not_met_without_sensitivity_receipt",
                ]
            ),
        )
    return (
        "not_evaluable",
        "not_evaluable",
        _sorted_reasons(reasons or ["native_s06_ledger_not_evaluable"]),
    )


def _rhythmicity_trajectories(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for cycle in source["cycle_instance_ledger"]["instances"]:
        by_candidate.setdefault(str(cycle["source_candidate_id"]), []).append(cycle)
    result: list[dict[str, Any]] = []
    for candidate_id in sorted(by_candidate):
        cycles = sorted(
            by_candidate[candidate_id],
            key=lambda row: (
                int(row["ordinal_within_candidate"]),
                str(row["cycle_instance_id"]),
            ),
        )
        points = [
            {
                "ordinal": index,
                "recording_interval_seconds": deepcopy(
                    cycle["recording_interval_seconds"]
                ),
                "recording_time_seconds": sum(
                    float(item) for item in cycle["recording_interval_seconds"]
                )
                / 2.0,
                "cycle_interval_seconds": float(
                    cycle["onset_to_onset_seconds"]
                ),
                "cycle_rate_hz": float(cycle["cycle_rate_hz"]),
                "peak_to_peak_interval_seconds": float(
                    cycle["peak_to_peak_seconds"]
                ),
                "source_cycle_instance_id": str(cycle["cycle_instance_id"]),
                "raw_support_references": deepcopy(
                    cycle["raw_support_references"]
                ),
            }
            for index, cycle in enumerate(cycles, start=1)
        ]
        transitions = []
        for previous, current in zip(points, points[1:]):
            delta_time = (
                current["recording_time_seconds"]
                - previous["recording_time_seconds"]
            )
            if delta_time <= _TOL:
                raise ValueError("S06 cycle trajectory is not time ordered")
            transitions.append(
                {
                    "ordinal": len(transitions) + 1,
                    "recording_interval_seconds": [
                        previous["recording_time_seconds"],
                        current["recording_time_seconds"],
                    ],
                    "delta_cycle_rate_hz": (
                        current["cycle_rate_hz"] - previous["cycle_rate_hz"]
                    ),
                    "cycle_rate_slope_hz_per_second": (
                        current["cycle_rate_hz"] - previous["cycle_rate_hz"]
                    )
                    / delta_time,
                    "delta_cycle_interval_seconds": (
                        current["cycle_interval_seconds"]
                        - previous["cycle_interval_seconds"]
                    ),
                    "from_point_ordinal": previous["ordinal"],
                    "to_point_ordinal": current["ordinal"],
                }
            )
        body = {
            "source_candidate_id": candidate_id,
            "analysis_unit": deepcopy(cycles[0]["analysis_unit"]),
            "coordinate_system": "recording_relative_seconds",
            "trajectory_source": "native_successive_cycle_instance_ledger",
            "points": points,
            "transition_intervals": transitions,
            "spectral_peak_used": False,
            "autocorrelation_used": False,
            "clinical_rhythmicity_qualification_authorized": False,
        }
        result.append(
            {
                "trajectory_id": "S06COURSE-" + _canonical_sha256(body)[:24],
                **body,
            }
        )
    return result


def _periodic_element_instances(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    component_by_element = {
        str(row["source_element_instance_id"]): row
        for row in source["component_instance_ledger"]["instances"]
    }
    result = []
    for element in source["element_instance_ledger"]["instances"]:
        element_id = str(element["element_instance_id"])
        if element_id not in component_by_element:
            raise ValueError("S06 element lacks its numerical component instance")
        body = {
            "element_instance": deepcopy(element),
            "component_instance": deepcopy(component_by_element[element_id]),
            "candidate_semantics": (
                "bounded_element_in_repeated_sequence_research_candidate_only"
            ),
            "spectral_peak_used": False,
            "autocorrelation_used": False,
            "clinical_periodic_term_authorized": False,
        }
        result.append(
            {
                "instance_id": "S06PEREL-" + _canonical_sha256(body)[:24],
                **body,
            }
        )
    return result


def _run_spans(cycles: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    if not cycles:
        return []
    maximum_ratio = float(_RUN_POLICY["maximum_adjacent_cycle_ratio"])
    durations = [float(row["onset_to_onset_seconds"]) for row in cycles]
    start = 0
    result = []
    for index in range(1, len(durations)):
        ratio = max(durations[index - 1], durations[index]) / min(
            durations[index - 1], durations[index]
        )
        if ratio > maximum_ratio:
            result.append((start, index - 1))
            start = index
    result.append((start, len(cycles) - 1))
    return result


def _rhythmic_run_instances(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    element_by_id = {
        str(row["element_instance_id"]): row
        for row in source["element_instance_ledger"]["instances"]
    }
    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for cycle in source["cycle_instance_ledger"]["instances"]:
        by_candidate.setdefault(str(cycle["source_candidate_id"]), []).append(cycle)
    result = []
    minimum_cycles = int(_RUN_POLICY["minimum_rhythmic_run_elements"]) - 1
    for candidate_id in sorted(by_candidate):
        cycles = sorted(
            by_candidate[candidate_id],
            key=lambda row: int(row["ordinal_within_candidate"]),
        )
        for start, stop in _run_spans(cycles):
            selected = cycles[start : stop + 1]
            if len(selected) < minimum_cycles:
                continue
            durations = [float(row["onset_to_onset_seconds"]) for row in selected]
            median, mad = _median_mad(durations)
            robust_cv = 1.4826 * mad / median
            if robust_cv > float(_RUN_POLICY["maximum_run_robust_cv"]):
                continue
            first_element_id = str(selected[0]["from_element_instance_id"])
            last_element_id = str(selected[-1]["to_element_instance_id"])
            first = element_by_id[first_element_id]
            last = element_by_id[last_element_id]
            source_element_ids = [
                first_element_id,
                *[str(row["to_element_instance_id"]) for row in selected],
            ]
            body = {
                "source_candidate_id": candidate_id,
                "analysis_unit": deepcopy(selected[0]["analysis_unit"]),
                "recording_interval_seconds": [
                    float(first["recording_interval_seconds"][0]),
                    float(last["recording_interval_seconds"][1]),
                ],
                "element_count": len(source_element_ids),
                "cycle_interval_count": len(selected),
                "cycle_interval_median_seconds": median,
                "cycle_interval_mad_seconds": mad,
                "cycle_interval_robust_cv": robust_cv,
                "cycle_rate_hz": 1.0 / median,
                "source_element_instance_ids": source_element_ids,
                "source_cycle_instance_ids": [
                    str(row["cycle_instance_id"]) for row in selected
                ],
                "raw_support_references": [
                    deepcopy(reference)
                    for row in selected
                    for reference in row["raw_support_references"]
                ],
                "inference_source": "native_successive_cycle_instance_ledger",
                "candidate_semantics": "engineering_rhythmic_run_candidate_only",
                "spectral_peak_used": False,
                "autocorrelation_used": False,
                "clinical_rhythmic_term_authorized": False,
            }
            result.append(
                {
                    "instance_id": "S06RHYRUN-" + _canonical_sha256(body)[:24],
                    **body,
                }
            )
    return result


def _query_result(
    source: Mapping[str, Any],
    *,
    term_query_id: str,
) -> dict[str, Any]:
    if term_query_id not in _QUERY_SPECS:
        raise ValueError("unknown native S06 term query")
    term_id, claim_kind, family = _QUERY_SPECS[term_query_id]
    raw_by_element = _raw_support_by_element(source)
    if term_query_id == "TQ-EVENT-RHYTHMICITY-COURSE":
        trajectories = _rhythmicity_trajectories(source)
        instances: list[dict[str, Any]] = []
        ledger = source["cycle_instance_ledger"]
        qualification, opportunity, reasons = _query_opportunity(
            ledger, has_output=bool(trajectories)
        )
        assertion = "measured"
        used_element_ids = {
            str(reference["element_instance_id"])
            for cycle in ledger["instances"]
            for reference in cycle["raw_support_references"]
        }
        semantic_reasons = [
            "course_replayed_from_native_successive_cycle_instance_ledger",
            "clinical_rhythmicity_not_qualified",
        ]
    elif term_query_id == "TQ-PERIODIC-ELEMENT-INSTANCE":
        trajectories = []
        instances = _periodic_element_instances(source)
        ledger = source["element_instance_ledger"]
        qualification, opportunity, reasons = _query_opportunity(
            ledger,
            has_output=bool(instances),
            uncertain_with_evidence=bool(instances),
        )
        assertion = "model_candidate"
        used_element_ids = {
            str(row["element_instance"]["element_instance_id"])
            for row in instances
        }
        semantic_reasons = [
            "explicit_element_and_numerical_component_instances_replayed",
            "not_a_clinical_periodic_discharge_claim",
        ]
    else:
        trajectories = []
        instances = _rhythmic_run_instances(source)
        ledger = source["cycle_instance_ledger"]
        cycle_evidence_exists = bool(ledger["instances"])
        qualification, opportunity, reasons = _query_opportunity(
            ledger,
            has_output=bool(instances),
            uncertain_with_evidence=cycle_evidence_exists,
        )
        assertion = "model_candidate"
        used_element_ids = {
            str(element_id)
            for row in instances
            for element_id in row["source_element_instance_ids"]
        }
        semantic_reasons = [
            "engineering_run_rule_bound_to_native_cycle_instance_ledger",
            "not_a_clinical_rhythmic_pattern_claim",
        ]
    raw_support = [raw_by_element[key] for key in sorted(used_element_ids)]
    raw_hashes = sorted(
        {str(row["dependency_sha256"]) for row in raw_support}
    )
    result: dict[str, Any] = {
        "term_query_id": term_query_id,
        "term_id": term_id,
        "claim_kind": claim_kind,
        "family": family,
        "assertion_level": assertion,
        "qualification_status": qualification,
        "opportunity": {
            "status": opportunity,
            "reason_codes": reasons,
            "not_evaluable_is_negative": False,
            "absence_inference_authorized": False,
        },
        "measurements": [],
        "trajectories": trajectories,
        "instances": instances,
        "raw_support_receipts": raw_support,
        "raw_dependency_sha256s": raw_hashes,
        "source_artifact_bindings": [
            {
                "source_kind": "event_component_cycle_element_ledger_v1",
                "source_artifact_id": str(source["event_id"]),
                "source_artifact_sha256": str(source["receipt_sha256"]),
            }
        ],
        "projection_method_id": (
            EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_METHOD_ID
        ),
        "projection_policy": deepcopy(_RUN_POLICY),
        "whole_output_unit_identity_preserved": True,
        "bipolar_endpoint_fact_projection_authorized": False,
        "negative_assertion_authorized": False,
        "clinical_term_qualification_authorized": False,
        "report_promotion_authorized": False,
        "onset_support_eligible": False,
        "soz_support_eligible": False,
        "reason_codes": semantic_reasons,
        "query_result_sha256": "",
    }
    result["query_result_sha256"] = _self_hash(result, "query_result_sha256")
    return result


def _body(source: Mapping[str, Any]) -> dict[str, Any]:
    queries = [
        _query_result(source, term_query_id=query_id)
        for query_id in S06_REQUIRED_QUERY_IDS
    ]
    return {
        "schema_version": (
            EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_SCHEMA_VERSION
        ),
        "method_id": EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_METHOD_ID,
        "event_id": str(source["event_id"]),
        "recording_id": str(source["recording_id"]),
        "canonical_signal_id": str(source["canonical_signal_id"]),
        "canonical_receipt_sha256": str(source["canonical_receipt_sha256"]),
        "source_signal_sha256": str(source["source_signal_sha256"]),
        "analysis_interval_seconds": deepcopy(source["analysis_interval_seconds"]),
        "coordinate_system": "recording_relative_seconds",
        "event_card_slot_id": "S06_RHYTHMICITY_PERIODICITY",
        "source_component_cycle_element_ledger_receipt": deepcopy(dict(source)),
        "source_component_cycle_element_ledger_receipt_sha256": str(
            source["receipt_sha256"]
        ),
        "query_results": queries,
        "query_result_roster_sha256": _canonical_sha256(queries),
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
    }


def materialize_event_component_cycle_element_query_adapter_v1(
    component_cycle_element_ledger_receipt: object,
) -> dict[str, Any]:
    """Project one native S06 receipt into exactly three term queries."""

    source = validate_event_component_cycle_element_ledger_v1(
        component_cycle_element_ledger_receipt
    )
    result = _body(source)
    result["receipt_sha256"] = _self_hash(result, "receipt_sha256")
    return validate_event_component_cycle_element_query_adapter_v1(result)


def validate_event_component_cycle_element_query_adapter_v1(
    value: object,
) -> dict[str, Any]:
    """Replay all three query payloads from the embedded native S06 source."""

    if type(value) is not dict:
        raise TypeError("native S06 query adapter must be an object")
    candidate = deepcopy(value)
    _reject_nonfinite(candidate)
    required = {
        "schema_version",
        "method_id",
        "event_id",
        "recording_id",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "analysis_interval_seconds",
        "coordinate_system",
        "event_card_slot_id",
        "source_component_cycle_element_ledger_receipt",
        "source_component_cycle_element_ledger_receipt_sha256",
        "query_results",
        "query_result_roster_sha256",
        "firewall",
        "authorization",
        "receipt_sha256",
    }
    if set(candidate) != required:
        raise ValueError("native S06 query adapter fields drifted")
    if (
        candidate["schema_version"]
        != EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_SCHEMA_VERSION
        or candidate["method_id"]
        != EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_METHOD_ID
        or candidate["event_card_slot_id"] != "S06_RHYTHMICITY_PERIODICITY"
    ):
        raise ValueError("native S06 query adapter identity drifted")
    for field in ("event_id", "recording_id", "canonical_signal_id"):
        _identifier(candidate[field], field)
    for field in (
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "source_component_cycle_element_ledger_receipt_sha256",
        "query_result_roster_sha256",
        "receipt_sha256",
    ):
        _sha(candidate[field], field)
    if (
        candidate["coordinate_system"] != "recording_relative_seconds"
        or candidate["firewall"] != _FIREWALL
        or candidate["authorization"] != _AUTHORIZATION
    ):
        raise ValueError("native S06 query adapter permissions drifted")
    source = validate_event_component_cycle_element_ledger_v1(
        candidate["source_component_cycle_element_ledger_receipt"]
    )
    if (
        candidate["source_component_cycle_element_ledger_receipt_sha256"]
        != source["receipt_sha256"]
    ):
        raise ValueError("native S06 query adapter source binding drifted")
    expected = _body(source)
    actual = deepcopy(candidate)
    actual.pop("receipt_sha256")
    if actual != expected:
        raise ValueError("native S06 query adapter does not replay from source")
    if candidate["receipt_sha256"] != _self_hash(candidate, "receipt_sha256"):
        raise ValueError("native S06 query adapter self hash drifted")
    return candidate


__all__ = [
    "EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_METHOD_ID",
    "EVENT_COMPONENT_CYCLE_ELEMENT_QUERY_ADAPTER_SCHEMA_VERSION",
    "S06_REQUIRED_QUERY_IDS",
    "materialize_event_component_cycle_element_query_adapter_v1",
    "validate_event_component_cycle_element_query_adapter_v1",
]
