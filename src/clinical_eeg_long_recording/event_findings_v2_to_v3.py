"""Explicit fail-closed migration from event Findings v2 to v3.

The adapter preserves the complete validated v2 projection.  v2 did not
distinguish rhythmicity from periodicity and did not record first-class
quantity/burden/variability, acquisition capability, or competing signal
hypotheses.  Those additions therefore remain ``not_evaluable``.  In
particular, a legacy ``rhythm`` Finding is never copied into either positive
v3 qualification gate and no new positive/negative Finding is manufactured.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping

from .event_findings_v2_validation import (
    validate_event_eeg_findings_v2_payload,
)
from .event_findings_v3_validation import (
    EVENT_FINDINGS_V2_TO_V3_MIGRATOR_ID,
    validate_event_eeg_findings_v3_payload,
)


_LOSS_CODES = [
    "acquisition_capabilities_not_recorded_in_v2",
    "competing_hypotheses_not_recorded_in_v2",
    "occurrence_burden_not_recorded_in_v2",
    "v2_rhythm_semantics_not_recoverable",
    "variability_not_recorded_in_v2",
]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identifier(value: object, *, prefix: str = "ID") -> str:
    text = re.sub(r"[^A-Za-z0-9._:-]+", "-", str(value)).strip("-._:")
    if not text:
        text = prefix
    if not text[0].isalnum():
        text = f"{prefix}-{text}"
    return text[:256]


def _sensitive_feature_class(finding: Mapping[str, Any]) -> str | None:
    if finding["family"] == "high_frequency":
        return "high_frequency_oscillation"
    term_id = str(finding["term"]["term_id"]).lower()
    if "dc_shift" in term_id or "direct_current_shift" in term_id:
        return "dc_shift"
    if any(
        token in term_id
        for token in (
            "high_frequency_oscillation",
            "hfo",
            "fast_ripple",
            "ripple",
        )
    ):
        return "high_frequency_oscillation"
    if "low_voltage_fast_activity" in term_id or "lvfa" in term_id:
        return "low_voltage_fast_activity"
    return None


def _acquisition_capabilities(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    term_to_class: dict[str, str] = {
        "dc_shift": "dc_shift",
        "high_frequency_oscillation": "high_frequency_oscillation",
        "low_voltage_fast_activity": "low_voltage_fast_activity",
    }
    for finding in payload["findings"]:
        feature_class = _sensitive_feature_class(finding)
        if feature_class is not None:
            term_to_class[str(finding["term"]["term_id"])] = feature_class
    return [
        {
            "capability_id": _identifier(f"MIG-ACQ-{term_id}"),
            "term_id": term_id,
            "feature_class": feature_class,
            "status": "not_evaluable",
            "source_view_ids": [],
            "effective_bandwidth_hz": None,
            "sample_rate_hz": None,
            "coupling": "unknown",
            "evaluation_opportunity_ids": [],
            "reason_codes": ["v2_acquisition_capability_not_recorded"],
        }
        for term_id, feature_class in sorted(term_to_class.items())
    ]


def migrate_event_eeg_findings_v2_to_v3(
    value: object,
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Migrate one validated v2 payload without upgrading missing evidence."""

    source = validate_event_eeg_findings_v2_payload(
        value,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    source_sha256 = _sha256(source)
    result: dict[str, Any] = deepcopy(source)
    result["schema_version"] = "event_eeg_findings_v3"
    result["occurrence_burden_variability"] = {
        "status": "not_evaluable",
        "analysis_scope": "signal_only_event_window",
        "summaries": [],
        "reason_codes": [
            "v2_occurrence_burden_variability_not_recorded"
        ],
    }
    empty_rhythm_gate = {
        "qualification_status": "not_evaluable",
        "term_ids": [],
        "finding_ids": [],
        "evaluation_opportunity_ids": [],
        "capability_receipt_ids": [],
        "term_decision_receipt_ids": [],
        "reason_codes": ["v2_rhythm_kind_not_recoverable"],
    }
    result["rhythm_periodicity_qualification"] = {
        "rhythmicity": deepcopy(empty_rhythm_gate),
        "periodicity": deepcopy(empty_rhythm_gate),
    }
    result["acquisition_capabilities"] = {
        "status": "not_evaluable",
        "capabilities": _acquisition_capabilities(source),
        "reason_codes": ["v2_acquisition_capability_not_recorded"],
    }
    result["competing_hypotheses"] = {
        "status": "not_evaluable",
        "selected_hypothesis_id": None,
        "hypotheses": [],
        "reason_codes": ["v2_competing_hypotheses_not_recorded"],
    }
    qualification_status = str(source["event_qualification"]["status"])
    outcome = {
        "qualified_electrographic_seizure": "qualified_electrographic_seizure",
        "qualified_electrographic_event": "qualified_electrographic_event",
        "unqualified_candidate": "candidate_only",
        "not_evaluable": "not_possible_to_determine",
    }[qualification_status]
    reason_codes: list[str]
    if outcome in {
        "qualified_electrographic_seizure",
        "qualified_electrographic_event",
    }:
        reason_codes = []
    else:
        reason_codes = [f"v2_{qualification_status}_preserved"]
    result["event_outcome"] = {
        "outcome": outcome,
        "evidence_ids": list(
            source["event_qualification"]["supporting_evidence_ids"]
        ),
        "competing_hypothesis_ids": [],
        "artifact_interval_indices": [],
        "reason_codes": reason_codes,
    }
    result["v3_migration"] = {
        "schema_version": "clinical_eeg_findings_v2_to_v3_migration_v1",
        "migrator_id": EVENT_FINDINGS_V2_TO_V3_MIGRATOR_ID,
        "source_schema_version": "event_eeg_findings_v2",
        "source_payload_sha256": source_sha256,
        "preserved_base_projection_sha256": source_sha256,
        "lossy": True,
        "loss_codes": list(_LOSS_CODES),
    }
    return validate_event_eeg_findings_v3_payload(
        result,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )


__all__ = [
    "EVENT_FINDINGS_V2_TO_V3_MIGRATOR_ID",
    "migrate_event_eeg_findings_v2_to_v3",
]
