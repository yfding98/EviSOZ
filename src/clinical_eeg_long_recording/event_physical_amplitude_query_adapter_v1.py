"""Replayable adapter from native S04 evidence into Findings-v1 queries.

The original Findings-v1 composer obtains its two S04 query payloads from the
broader morphology primitive sidecar.  The dedicated physical-amplitude
producer is intentionally additive, so this adapter provides a narrow bridge
without changing that legacy producer or its hashes.  It projects one
validated ``event_physical_amplitude_findings_v1`` receipt into exactly:

* ``TQ-PHYSICAL-AMPLITUDE-PROFILE``; and
* ``TQ-EVENT-AMPLITUDE-COURSE``.

The projection keeps every typed opportunity, QC reason, physical-unit
measurement, relative-amplitude ratio, trajectory, and canonical raw-sample
support receipt.  A bipolar output remains one whole lead.  Neither its name
nor its source-channel calibration lineage is interpreted as endpoint-level
evidence.  Amplitude or a low relative ratio is never promoted to clinical
attenuation/electrodecrement, evolution, onset, SOZ/EZ, or report text.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

from .event_physical_amplitude_findings_v1 import (
    validate_event_physical_amplitude_findings_v1,
)


EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_event_physical_amplitude_query_adapter_v1"
)
EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_METHOD_ID: Final[str] = (
    "EVENT-PHYSICAL-AMPLITUDE-QUERY-ADAPTER-V1"
)

S04_REQUIRED_QUERY_IDS: Final[tuple[str, str]] = (
    "TQ-EVENT-AMPLITUDE-COURSE",
    "TQ-PHYSICAL-AMPLITUDE-PROFILE",
)

_QUERY_SPECS: Final[dict[str, tuple[str, str, str]]] = {
    "TQ-EVENT-AMPLITUDE-COURSE": (
        "event_amplitude_course_profile",
        "measurement",
        "evolution",
    ),
    "TQ-PHYSICAL-AMPLITUDE-PROFILE": (
        "deterministic_event_physical_amplitude_profile",
        "measurement",
        "spectral",
    ),
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")

_FIREWALL: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}
_AUTHORIZATION: Final[dict[str, bool | str]] = {
    "event_card_slot_id": "S04_PHYSICAL_AMPLITUDE",
    "projection_scope": "physical_measurements_relative_ratios_and_course_only",
    "whole_bipolar_lead_identity_required": True,
    "bipolar_endpoint_fact_projection_authorized": False,
    "clinical_attenuation_or_electrodecrement_authorized": False,
    "amplitude_change_as_evolution_authorized": False,
    "clinical_term_qualification_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
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
        raise ValueError("opportunity reason codes must be non-empty and trimmed")
    return result


def _aggregate_opportunity(
    rows: Sequence[Mapping[str, Any]],
    *,
    evaluable_statuses: frozenset[str],
    complete_status: str,
    empty_reason: str,
    limited_reason: str,
) -> tuple[str, str, list[str]]:
    reasons = _sorted_reasons(
        [
            reason
            for row in rows
            for reason in row["opportunity"]["reason_codes"]
        ]
    )
    if rows and all(str(row["status"]) == complete_status for row in rows):
        return "measured", "sufficient", reasons
    if any(str(row["status"]) in evaluable_statuses for row in rows):
        return (
            "measured",
            "limited",
            _sorted_reasons([*reasons, limited_reason]),
        )
    return (
        "not_evaluable",
        "not_evaluable",
        _sorted_reasons([*reasons, empty_reason]),
    )


def _raw_support_receipts(
    source: Mapping[str, Any],
    *,
    measurement_role: str | None,
) -> list[dict[str, Any]]:
    physical = source["source_physical_amplitude_receipt"]
    measurements_by_source_id = {
        str(row["source_amplitude_row_id"]): row for row in source["measurements"]
    }
    query_by_hash = {
        _canonical_sha256(row): row for row in physical["query_roster"]
    }
    result: list[dict[str, Any]] = []
    for source_row in physical["rows"]:
        binding = source_row["source_binding"]
        query = query_by_hash[str(binding["query_binding_sha256"])]
        if measurement_role is not None and query["measurement_role"] != measurement_role:
            continue
        measurement = measurements_by_source_id[str(source_row["row_id"])]
        if (
            measurement["view_id"] != binding["view_id"]
            or measurement["unit_id"] != binding["unit_id"]
            or measurement["unit_type"] != binding["unit_type"]
        ):
            raise ValueError("S04 measurement/raw-support unit identity drifted")
        receipt: dict[str, Any] = {
            "source_row_id": str(source_row["row_id"]),
            "source_row_binding_sha256": str(source_row["row_binding_sha256"]),
            "source_binding_sha256": str(source_row["source_binding_sha256"]),
            "view_id": str(binding["view_id"]),
            "unit_id": str(binding["unit_id"]),
            "unit_type": str(binding["unit_type"]),
            "whole_output_unit_identity_preserved": True,
            "bipolar_endpoint_fact_projection_authorized": False,
            "measurement_role": str(query["measurement_role"]),
            "requested_recording_interval_seconds": list(
                binding["requested_recording_interval_seconds"]
            ),
            "recording_interval_seconds": list(binding["recording_interval_seconds"]),
            "raw_sample_intervals": deepcopy(binding["raw_sample_intervals"]),
            "raw_support_includes_missing_carrier_as_zero_length_only": True,
            "opportunity": deepcopy(source_row["opportunity"]),
            "future_sample_access": False,
            "dependency_policy": "instantaneous",
            "raw_support_receipt_sha256": "",
        }
        receipt["raw_support_receipt_sha256"] = _self_hash(
            receipt, "raw_support_receipt_sha256"
        )
        result.append(receipt)
    return sorted(
        result,
        key=lambda row: (
            row["view_id"],
            row["unit_id"],
            row["recording_interval_seconds"][0],
            row["recording_interval_seconds"][1],
            row["source_row_id"],
        ),
    )


def _query_result(
    source: Mapping[str, Any],
    *,
    term_query_id: str,
) -> dict[str, Any]:
    if term_query_id not in _QUERY_SPECS:
        raise ValueError("unknown S04 required-query ID")
    term_id, claim_kind, family = _QUERY_SPECS[term_query_id]
    if term_query_id == "TQ-PHYSICAL-AMPLITUDE-PROFILE":
        measurements = deepcopy(source["measurements"])
        ratios = deepcopy(source["attenuation_ratios"])
        trajectories: list[dict[str, Any]] = []
        raw_support = _raw_support_receipts(source, measurement_role=None)
        status, opportunity, reasons = _aggregate_opportunity(
            measurements,
            evaluable_statuses=frozenset({"measured"}),
            complete_status="measured",
            empty_reason="no_evaluable_native_s04_physical_amplitude_measurement",
            limited_reason="some_native_s04_physical_amplitude_measurements_not_evaluable",
        )
        semantic_reasons = [
            "native_s04_physical_uv_measurements_replayed",
            "relative_ratio_is_measurement_not_clinical_attenuation",
        ]
    else:
        measurements = []
        ratios = []
        trajectories = deepcopy(source["amplitude_trajectories"])
        raw_support = _raw_support_receipts(source, measurement_role="event_course")
        status, opportunity, reasons = _aggregate_opportunity(
            trajectories,
            evaluable_statuses=frozenset({"measured", "partially_measured"}),
            complete_status="measured",
            empty_reason="no_evaluable_native_s04_amplitude_course",
            limited_reason="native_s04_amplitude_course_is_incomplete",
        )
        semantic_reasons = [
            "native_s04_course_replayed_from_adjacent_physical_time_windows",
            "amplitude_change_alone_is_not_ictal_evolution",
        ]
    raw_hashes = [str(row["raw_support_receipt_sha256"]) for row in raw_support]
    result: dict[str, Any] = {
        "term_query_id": term_query_id,
        "term_id": term_id,
        "claim_kind": claim_kind,
        "family": family,
        "assertion_level": "measured",
        "qualification_status": status,
        "opportunity": {
            "status": opportunity,
            "reason_codes": reasons,
            "not_evaluable_is_negative": False,
        },
        "measurements": measurements,
        "relative_amplitude_ratios": ratios,
        "trajectories": trajectories,
        "instances": [],
        "raw_support_receipts": raw_support,
        "raw_dependency_sha256s": raw_hashes,
        "source_artifact_bindings": [
            {
                "source_kind": "event_physical_amplitude_findings_v1",
                "source_artifact_id": str(source["event_id"]),
                "source_artifact_sha256": str(source["receipt_sha256"]),
            }
        ],
        "projection_method_id": EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_METHOD_ID,
        "whole_output_unit_identity_preserved": True,
        "bipolar_endpoint_fact_projection_authorized": False,
        "negative_assertion_authorized": False,
        "clinical_attenuation_or_electrodecrement_authorized": False,
        "amplitude_change_as_evolution_authorized": False,
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
    queries = [_query_result(source, term_query_id=item) for item in S04_REQUIRED_QUERY_IDS]
    return {
        "schema_version": EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_SCHEMA_VERSION,
        "method_id": EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_METHOD_ID,
        "event_id": str(source["event_id"]),
        "recording_id": str(source["recording_id"]),
        "canonical_signal_id": str(source["canonical_signal_id"]),
        "canonical_receipt_sha256": str(source["canonical_receipt_sha256"]),
        "source_signal_sha256": str(source["source_signal_sha256"]),
        "analysis_interval_seconds": list(source["analysis_interval_seconds"]),
        "coordinate_system": "recording_relative_seconds",
        "event_card_slot_id": "S04_PHYSICAL_AMPLITUDE",
        "source_physical_amplitude_findings_receipt": deepcopy(dict(source)),
        "source_physical_amplitude_findings_receipt_sha256": str(
            source["receipt_sha256"]
        ),
        "query_results": queries,
        "query_result_roster_sha256": _canonical_sha256(queries),
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
    }


def materialize_event_physical_amplitude_query_adapter_v1(
    physical_amplitude_findings_receipt: object,
) -> dict[str, Any]:
    """Project one validated native S04 receipt into its two Event-Card queries."""

    source = validate_event_physical_amplitude_findings_v1(
        physical_amplitude_findings_receipt
    )
    result = _body(source)
    result["receipt_sha256"] = _self_hash(result, "receipt_sha256")
    return validate_event_physical_amplitude_query_adapter_v1(result)


def validate_event_physical_amplitude_query_adapter_v1(
    value: object,
) -> dict[str, Any]:
    """Replay the complete adapter from its embedded typed native S04 source."""

    if type(value) is not dict:
        raise TypeError("S04 query adapter must be an object")
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
        "source_physical_amplitude_findings_receipt",
        "source_physical_amplitude_findings_receipt_sha256",
        "query_results",
        "query_result_roster_sha256",
        "firewall",
        "authorization",
        "receipt_sha256",
    }
    if set(candidate) != required:
        raise ValueError("S04 query adapter fields drifted")
    if (
        candidate["schema_version"]
        != EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_SCHEMA_VERSION
        or candidate["method_id"] != EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_METHOD_ID
        or candidate["event_card_slot_id"] != "S04_PHYSICAL_AMPLITUDE"
    ):
        raise ValueError("S04 query adapter identity drifted")
    for field in ("event_id", "recording_id", "canonical_signal_id"):
        _identifier(candidate[field], field)
    for field in (
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "source_physical_amplitude_findings_receipt_sha256",
        "query_result_roster_sha256",
        "receipt_sha256",
    ):
        _sha(candidate[field], field)
    if (
        candidate["coordinate_system"] != "recording_relative_seconds"
        or candidate["firewall"] != _FIREWALL
        or candidate["authorization"] != _AUTHORIZATION
    ):
        raise ValueError("S04 query adapter firewall/authorization drifted")
    source = validate_event_physical_amplitude_findings_v1(
        candidate["source_physical_amplitude_findings_receipt"]
    )
    if candidate["source_physical_amplitude_findings_receipt_sha256"] != source[
        "receipt_sha256"
    ]:
        raise ValueError("S04 query adapter source receipt binding drifted")
    expected = _body(source)
    actual = deepcopy(candidate)
    actual.pop("receipt_sha256")
    if actual != expected:
        raise ValueError("S04 query adapter does not replay from its source receipt")
    if candidate["receipt_sha256"] != _self_hash(candidate, "receipt_sha256"):
        raise ValueError("S04 query adapter self hash drifted")
    return candidate


__all__ = [
    "EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_METHOD_ID",
    "EVENT_PHYSICAL_AMPLITUDE_QUERY_ADAPTER_SCHEMA_VERSION",
    "S04_REQUIRED_QUERY_IDS",
    "materialize_event_physical_amplitude_query_adapter_v1",
    "validate_event_physical_amplitude_query_adapter_v1",
]
