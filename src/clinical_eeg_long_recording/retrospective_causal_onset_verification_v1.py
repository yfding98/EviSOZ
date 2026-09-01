"""Offline proposal -> causal remeasurement contract for ictal onset review.

Clinicians often inspect a complete seizure and then revisit an earlier time
or channel.  This is useful retrospective search, but the later course cannot
itself become positive onset evidence.  The contract below separates:

1. an offline selector that may propose only a bounded search scope;
2. a new measurement on a future-sample-excluding onset view;
3. calibration for the multiplicity introduced by retrospective search; and
4. a verified *candidate* leaf whose late decision time remains explicit.

No signal value, saliency, annotation, spreadsheet field, physician label or
clinical text is permitted in the proposal.  The current trusted report route
is deliberately disconnected until real patient-disjoint calibration exists.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence


RETROSPECTIVE_ONSET_PROPOSAL_SCHEMA_VERSION = "retrospective_onset_search_proposal_v1"
CAUSAL_EARLY_LEAF_REMEASUREMENT_SCHEMA_VERSION = "causal_early_leaf_remeasurement_v1"
RETROSPECTIVE_MULTIPLE_SEARCH_CALIBRATION_SCHEMA_VERSION = (
    "retrospective_multiple_search_calibration_v1"
)
RETROSPECTIVE_VERIFIED_ONSET_LEAF_SCHEMA_VERSION = (
    "retrospectively_proposed_onset_view_verified_leaf_v1"
)
RETROSPECTIVE_CAUSAL_ONSET_METHOD_ID = (
    "offline_search_proposal_then_future_free_causal_remeasurement_v1"
)
RETROSPECTIVE_VERIFIED_ONSET_TRUSTED_REPORT_ROUTE_CONNECTED = False

_HEX = frozenset("0123456789abcdef")
_RATIONALE_CODES: Final[frozenset[str]] = frozenset(
    {
        "boundary_left_tail_mass",
        "current_left_boundary_touched",
        "earlier_change_point_candidate",
        "early_field_instability",
        "course_consistent_earlier_candidate",
        "cross_reference_disagreement",
    }
)
_UNIT_TYPES = frozenset({"electrode", "lead", "region"})
_UNIT_ROSTER_AUTHORITIES = frozenset(
    {
        "predeclared_evaluable_unit_roster",
        "causal_coarse_candidate_roster",
        "offline_search_proposal_only",
    }
)
_MEASUREMENT_STATUSES = frozenset(
    {"present", "absent_with_opportunity", "uncertain", "not_evaluable"}
)
_CALIBRATED_CLAIM_TYPES = frozenset({"onset_time_support", "onset_topography_support"})
_MEASUREMENT_KINDS = frozenset(
    {
        "earliest_visible_change",
        "rhythmic_emergence_candidate",
        "field_emergence_candidate",
        "frequency_change_point_candidate",
        "morphology_change_point_candidate",
    }
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a non-empty trimmed identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{context} is outside its finite range")
    return result


def _interval(value: object, context: str, *, maximum: float) -> list[float]:
    if type(value) is not list or len(value) != 2:
        raise ValueError(f"{context} must be [start, stop]")
    start = _finite(value[0], f"{context} start", minimum=0.0)
    stop = _finite(value[1], f"{context} stop", minimum=0.0)
    if stop <= start or stop > maximum + 1e-9:
        raise ValueError(f"{context} is outside the recording")
    return [start, stop]


def _exact_keys(value: object, expected: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _unit_roster(value: object, context: str) -> list[dict[str, str]]:
    if type(value) is not list or not value:
        raise ValueError(f"{context} must be a non-empty unit roster")
    units: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        row = _exact_keys(raw, {"unit_type", "unit_id"}, f"{context} row {index}")
        if row["unit_type"] not in _UNIT_TYPES:
            raise ValueError(f"{context} row {index} unit type is invalid")
        _identifier(row["unit_id"], f"{context} row {index} ID")
        units.append(row)
    canonical = sorted(units, key=lambda row: (row["unit_type"], row["unit_id"]))
    if units != canonical or len(
        {(row["unit_type"], row["unit_id"]) for row in units}
    ) != len(units):
        raise ValueError(f"{context} must be sorted and unique")
    return units


def validate_retrospective_onset_search_proposal_v1(
    payload: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "proposal_id",
        "method_id",
        "recording_id",
        "patient_id",
        "event_id",
        "canonical_evidence_root_sha256",
        "recording_duration_seconds",
        "event_qualification_binding",
        "offline_selector_binding",
        "search_scope",
        "scope_receipt",
        "permissions",
        "receipt_sha256",
    }
    data = _exact_keys(payload, required, "retrospective onset proposal")
    if (
        data["schema_version"] != RETROSPECTIVE_ONSET_PROPOSAL_SCHEMA_VERSION
        or data["method_id"] != RETROSPECTIVE_CAUSAL_ONSET_METHOD_ID
    ):
        raise ValueError("retrospective onset proposal identity drifted")
    _identifier(data["recording_id"], "recording ID")
    _identifier(data["patient_id"], "patient ID")
    _identifier(data["event_id"], "event ID")
    _sha256(data["canonical_evidence_root_sha256"], "canonical evidence root")
    duration = _finite(
        data["recording_duration_seconds"], "recording duration", minimum=1e-9
    )
    qualification = _exact_keys(
        data["event_qualification_binding"],
        {"qualification_id", "qualification_receipt_sha256"},
        "event qualification binding",
    )
    _identifier(qualification["qualification_id"], "event qualification ID")
    _sha256(
        qualification["qualification_receipt_sha256"],
        "event qualification receipt",
    )
    selector = _exact_keys(
        data["offline_selector_binding"],
        {
            "offline_evidence_id",
            "offline_context_receipt_sha256",
            "view_role",
            "selector_available_recording_seconds",
            "proposal_policy_sha256",
        },
        "offline selector binding",
    )
    _identifier(selector["offline_evidence_id"], "offline evidence ID")
    _sha256(selector["offline_context_receipt_sha256"], "offline context receipt")
    _sha256(selector["proposal_policy_sha256"], "proposal policy")
    if selector["view_role"] != "context_offline":
        raise ValueError("retrospective selector must be an offline-context view")
    available = _finite(
        selector["selector_available_recording_seconds"],
        "selector-available time",
        minimum=0.0,
    )
    if available > duration + 1e-9:
        raise ValueError("selector-available time exceeds the recording")

    search = _exact_keys(
        data["search_scope"],
        {
            "interval_recording_seconds",
            "unit_candidates",
            "unit_roster_authority",
            "unit_roster_authority_receipt_sha256",
            "candidate_search_count",
            "candidate_roster_sha256",
            "multiple_search_family_id",
            "rationale_codes",
        },
        "retrospective search scope",
    )
    interval = _interval(
        search["interval_recording_seconds"],
        "retrospective search interval",
        maximum=duration,
    )
    if interval[1] > available + 1e-9:
        raise ValueError("retrospective search extends past selector availability")
    units = _unit_roster(search["unit_candidates"], "retrospective search units")
    if search["unit_roster_authority"] not in _UNIT_ROSTER_AUTHORITIES:
        raise ValueError("retrospective unit-roster authority is invalid")
    _sha256(
        search["unit_roster_authority_receipt_sha256"],
        "retrospective unit-roster authority receipt",
    )
    candidate_count = search["candidate_search_count"]
    if type(candidate_count) is not int or candidate_count < len(units):
        raise ValueError("retrospective candidate search count is too small")
    _sha256(search["candidate_roster_sha256"], "candidate roster")
    _identifier(search["multiple_search_family_id"], "multiple-search family")
    rationales = search["rationale_codes"]
    if (
        type(rationales) is not list
        or not rationales
        or rationales != sorted(set(rationales))
        or any(code not in _RATIONALE_CODES for code in rationales)
    ):
        raise ValueError("retrospective rationale roster is invalid")

    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "physician_labels_or_report_used": False,
        "clinical_context_used": False,
        "offline_signal_values_or_saliency_forwarded": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("retrospective proposal EEG-only scope widened")
    expected_permissions = {
        "may_control_additional_search": True,
        "may_supply_onset_time_support": False,
        "may_supply_onset_topography_support": False,
        "may_create_report_eligible_finding": False,
    }
    if data["permissions"] != expected_permissions:
        raise ValueError("retrospective proposal permissions widened")
    digest = deepcopy(data)
    digest["proposal_id"] = "RETRO-ONSET-PROPOSAL-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["proposal_id"] != "RETROPROP-" + _canonical_sha256(digest)[:24]:
        raise ValueError("retrospective proposal ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("retrospective proposal receipt hash drifted")
    return data


def build_retrospective_onset_search_proposal_v1(
    *,
    recording_id: str,
    patient_id: str,
    event_id: str,
    canonical_evidence_root_sha256: str,
    recording_duration_seconds: float,
    event_qualification_id: str,
    event_qualification_receipt_sha256: str,
    offline_evidence_id: str,
    offline_context_receipt_sha256: str,
    selector_available_recording_seconds: float,
    proposal_policy_sha256: str,
    search_interval_recording_seconds: Sequence[float],
    unit_candidates: Sequence[Mapping[str, str]],
    unit_roster_authority: str,
    unit_roster_authority_receipt_sha256: str,
    candidate_search_count: int,
    candidate_roster_sha256: str,
    multiple_search_family_id: str,
    rationale_codes: Sequence[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": RETROSPECTIVE_ONSET_PROPOSAL_SCHEMA_VERSION,
        "proposal_id": "RETRO-ONSET-PROPOSAL-PENDING",
        "method_id": RETROSPECTIVE_CAUSAL_ONSET_METHOD_ID,
        "recording_id": recording_id,
        "patient_id": patient_id,
        "event_id": event_id,
        "canonical_evidence_root_sha256": canonical_evidence_root_sha256,
        "recording_duration_seconds": float(recording_duration_seconds),
        "event_qualification_binding": {
            "qualification_id": event_qualification_id,
            "qualification_receipt_sha256": event_qualification_receipt_sha256,
        },
        "offline_selector_binding": {
            "offline_evidence_id": offline_evidence_id,
            "offline_context_receipt_sha256": offline_context_receipt_sha256,
            "view_role": "context_offline",
            "selector_available_recording_seconds": float(
                selector_available_recording_seconds
            ),
            "proposal_policy_sha256": proposal_policy_sha256,
        },
        "search_scope": {
            "interval_recording_seconds": [
                float(search_interval_recording_seconds[0]),
                float(search_interval_recording_seconds[1]),
            ],
            "unit_candidates": [dict(row) for row in unit_candidates],
            "unit_roster_authority": unit_roster_authority,
            "unit_roster_authority_receipt_sha256": (
                unit_roster_authority_receipt_sha256
            ),
            "candidate_search_count": candidate_search_count,
            "candidate_roster_sha256": candidate_roster_sha256,
            "multiple_search_family_id": multiple_search_family_id,
            "rationale_codes": sorted(set(rationale_codes)),
        },
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "physician_labels_or_report_used": False,
            "clinical_context_used": False,
            "offline_signal_values_or_saliency_forwarded": False,
        },
        "permissions": {
            "may_control_additional_search": True,
            "may_supply_onset_time_support": False,
            "may_supply_onset_topography_support": False,
            "may_create_report_eligible_finding": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["proposal_id"] = "RETROPROP-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_retrospective_onset_search_proposal_v1(body)


def validate_causal_early_leaf_remeasurement_v1(
    payload: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "measurement_id",
        "proposal_id",
        "canonical_evidence_root_sha256",
        "measurement_kind",
        "view_binding",
        "physical_support",
        "temporal_permissions",
        "spatial_receipts",
        "measurement_status",
        "assertion_level",
        "receipt_sha256",
    }
    data = _exact_keys(payload, required, "causal early-leaf remeasurement")
    if data["schema_version"] != CAUSAL_EARLY_LEAF_REMEASUREMENT_SCHEMA_VERSION:
        raise ValueError("causal early-leaf schema drifted")
    _identifier(data["measurement_id"], "measurement ID")
    _identifier(data["proposal_id"], "proposal ID")
    _sha256(data["canonical_evidence_root_sha256"], "canonical evidence root")
    if data["measurement_kind"] not in _MEASUREMENT_KINDS:
        raise ValueError("causal early-leaf measurement kind is invalid")
    if data["measurement_status"] not in _MEASUREMENT_STATUSES:
        raise ValueError("causal early-leaf measurement status is invalid")
    view = _exact_keys(
        data["view_binding"],
        {"view_id", "view_role", "view_receipt_sha256", "transform_sha256"},
        "causal view binding",
    )
    _identifier(view["view_id"], "causal view ID")
    if view["view_role"] != "onset_causal":
        raise ValueError("early-leaf remeasurement must use onset_causal")
    _sha256(view["view_receipt_sha256"], "causal view receipt")
    _sha256(view["transform_sha256"], "causal transform")
    support = _exact_keys(
        data["physical_support"],
        {
            "interval_recording_seconds",
            "transform_input_interval_recording_seconds",
            "signal_available_recording_seconds",
            "measured_units",
            "raw_sample_sha256",
            "measurement_payload_sha256",
            "quality_opportunity_complete",
        },
        "causal physical support",
    )
    interval = _interval(
        support["interval_recording_seconds"],
        "causal physical support interval",
        maximum=float("inf"),
    )
    transform_interval = _interval(
        support["transform_input_interval_recording_seconds"],
        "causal transform-input support interval",
        maximum=float("inf"),
    )
    if (
        transform_interval[0] > interval[0] + 1e-9
        or transform_interval[1] < interval[1] - 1e-9
    ):
        raise ValueError("causal transform support does not cover the measurement")
    available = _finite(
        support["signal_available_recording_seconds"],
        "causal signal-available time",
        minimum=0.0,
    )
    if available + 1e-9 < transform_interval[1]:
        raise ValueError("causal measurement is available before its support ends")
    _unit_roster(support["measured_units"], "causal measured units")
    _sha256(support["raw_sample_sha256"], "causal raw sample")
    _sha256(support["measurement_payload_sha256"], "measurement payload")
    quality_complete = support["quality_opportunity_complete"]
    if type(quality_complete) is not bool:
        raise ValueError("causal quality/evaluation opportunity must be boolean")
    expected_quality_complete = data["measurement_status"] != "not_evaluable"
    if quality_complete is not expected_quality_complete:
        raise ValueError("causal status and evaluation opportunity are inconsistent")
    onset_time_eligible = data["measurement_status"] == "present"
    expected_temporal = {
        "future_sample_access": False,
        "zero_phase_or_offline_filter_used": False,
        "filter_support_recorded": True,
        "onset_time_support_eligible": onset_time_eligible,
    }
    if data["temporal_permissions"] != expected_temporal:
        raise ValueError("causal early-leaf temporal permissions widened")
    spatial = _exact_keys(
        data["spatial_receipts"],
        {
            "onset_topography_support_eligible",
            "constructive_spatial_receipt_sha256",
            "reference_stability_receipt_sha256",
        },
        "causal spatial receipts",
    )
    eligible = spatial["onset_topography_support_eligible"]
    if type(eligible) is not bool:
        raise ValueError("topography eligibility must be boolean")
    if eligible and not onset_time_eligible:
        raise ValueError("non-present measurement cannot support onset topography")
    for name in (
        "constructive_spatial_receipt_sha256",
        "reference_stability_receipt_sha256",
    ):
        value = spatial[name]
        if eligible:
            _sha256(value, name)
        elif value is not None:
            raise ValueError("ineligible topography cannot retain spatial authority")
    if data["assertion_level"] != "measured":
        raise ValueError("causal remeasurement must remain a measured atom")
    digest = deepcopy(data)
    digest["measurement_id"] = "CAUSAL-REMEASUREMENT-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["measurement_id"] != "CAUSALREMEAS-" + _canonical_sha256(digest)[:24]:
        raise ValueError("causal remeasurement ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("causal remeasurement receipt hash drifted")
    return data


def build_causal_early_leaf_remeasurement_v1(
    *,
    proposal_id: str,
    canonical_evidence_root_sha256: str,
    measurement_kind: str,
    measurement_status: str,
    view_id: str,
    view_receipt_sha256: str,
    transform_sha256: str,
    interval_recording_seconds: Sequence[float],
    transform_input_interval_recording_seconds: Sequence[float],
    signal_available_recording_seconds: float,
    measured_units: Sequence[Mapping[str, str]],
    raw_sample_sha256: str,
    measurement_payload_sha256: str,
    onset_topography_support_eligible: bool,
    constructive_spatial_receipt_sha256: str | None = None,
    reference_stability_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": CAUSAL_EARLY_LEAF_REMEASUREMENT_SCHEMA_VERSION,
        "measurement_id": "CAUSAL-REMEASUREMENT-PENDING",
        "proposal_id": proposal_id,
        "canonical_evidence_root_sha256": canonical_evidence_root_sha256,
        "measurement_kind": measurement_kind,
        "measurement_status": measurement_status,
        "view_binding": {
            "view_id": view_id,
            "view_role": "onset_causal",
            "view_receipt_sha256": view_receipt_sha256,
            "transform_sha256": transform_sha256,
        },
        "physical_support": {
            "interval_recording_seconds": [
                float(interval_recording_seconds[0]),
                float(interval_recording_seconds[1]),
            ],
            "transform_input_interval_recording_seconds": [
                float(causal_transform_value)
                for causal_transform_value in transform_input_interval_recording_seconds
            ],
            "signal_available_recording_seconds": float(
                signal_available_recording_seconds
            ),
            "measured_units": [dict(row) for row in measured_units],
            "raw_sample_sha256": raw_sample_sha256,
            "measurement_payload_sha256": measurement_payload_sha256,
            "quality_opportunity_complete": measurement_status != "not_evaluable",
        },
        "temporal_permissions": {
            "future_sample_access": False,
            "zero_phase_or_offline_filter_used": False,
            "filter_support_recorded": True,
            "onset_time_support_eligible": measurement_status == "present",
        },
        "spatial_receipts": {
            "onset_topography_support_eligible": onset_topography_support_eligible,
            "constructive_spatial_receipt_sha256": constructive_spatial_receipt_sha256,
            "reference_stability_receipt_sha256": reference_stability_receipt_sha256,
        },
        "assertion_level": "measured",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["measurement_id"] = "CAUSALREMEAS-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_causal_early_leaf_remeasurement_v1(body)


def validate_retrospective_multiple_search_calibration_v1(
    payload: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "calibration_id",
        "multiple_search_family_id",
        "proposal_policy_sha256",
        "source_split",
        "calibration_manifest_sha256",
        "target_exclusion_binding",
        "selection_count_cap",
        "calibrated_claim_types",
        "eligible_measurement_kinds",
        "eligible_unit_roster_authorities",
        "precision_lower_bound",
        "required_precision_lower_bound",
        "patient_disjoint_from_current_recording",
        "current_record_excluded",
        "frozen_before_current_recording",
        "multiple_search_scope_calibrated",
        "private_or_source_eval_labels_used",
        "passed",
        "receipt_sha256",
    }
    data = _exact_keys(payload, required, "retrospective search calibration")
    if (
        data["schema_version"]
        != RETROSPECTIVE_MULTIPLE_SEARCH_CALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError("retrospective search calibration schema drifted")
    _identifier(data["calibration_id"], "calibration ID")
    _identifier(data["multiple_search_family_id"], "multiple-search family")
    _sha256(data["proposal_policy_sha256"], "proposal policy")
    if data["source_split"] != "source_dev":
        raise ValueError("retrospective search calibration must be source-dev only")
    _sha256(data["calibration_manifest_sha256"], "calibration manifest")
    target = _exact_keys(
        data["target_exclusion_binding"],
        {"recording_id", "patient_id", "exclusion_receipt_sha256"},
        "retrospective calibration target exclusion",
    )
    _identifier(target["recording_id"], "calibration target recording ID")
    _identifier(target["patient_id"], "calibration target patient ID")
    _sha256(
        target["exclusion_receipt_sha256"],
        "calibration target-exclusion receipt",
    )
    cap = data["selection_count_cap"]
    if type(cap) is not int or cap < 1:
        raise ValueError("retrospective selection count cap must be positive")
    claim_types = data["calibrated_claim_types"]
    if (
        type(claim_types) is not list
        or not claim_types
        or claim_types != sorted(set(claim_types))
        or any(value not in _CALIBRATED_CLAIM_TYPES for value in claim_types)
    ):
        raise ValueError("retrospective calibrated claim types are invalid")
    if (
        "onset_topography_support" in claim_types
        and "onset_time_support" not in claim_types
    ):
        raise ValueError("topography calibration must also cover onset time")
    measurement_kinds = data["eligible_measurement_kinds"]
    if (
        type(measurement_kinds) is not list
        or not measurement_kinds
        or measurement_kinds != sorted(set(measurement_kinds))
        or any(value not in _MEASUREMENT_KINDS for value in measurement_kinds)
    ):
        raise ValueError("retrospective calibrated measurement kinds are invalid")
    roster_authorities = data["eligible_unit_roster_authorities"]
    if (
        type(roster_authorities) is not list
        or not roster_authorities
        or roster_authorities != sorted(set(roster_authorities))
        or any(value not in _UNIT_ROSTER_AUTHORITIES for value in roster_authorities)
    ):
        raise ValueError("retrospective calibrated roster authorities are invalid")
    lower = _finite(data["precision_lower_bound"], "precision lower bound")
    required_lower = _finite(
        data["required_precision_lower_bound"], "required precision lower bound"
    )
    if not 0.0 <= lower <= 1.0 or not 0.0 <= required_lower <= 1.0:
        raise ValueError("retrospective precision bounds must lie in [0,1]")
    for name in (
        "patient_disjoint_from_current_recording",
        "current_record_excluded",
        "frozen_before_current_recording",
        "multiple_search_scope_calibrated",
    ):
        if data[name] is not True:
            raise ValueError(f"retrospective calibration lacks {name}")
    if data["private_or_source_eval_labels_used"] is not False:
        raise ValueError("private/source-eval labels entered calibration")
    if data["passed"] is not (lower >= required_lower):
        raise ValueError("retrospective calibration pass flag is inconsistent")
    digest = deepcopy(data)
    digest["calibration_id"] = "RETRO-CALIBRATION-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["calibration_id"] != "RETROCAL-" + _canonical_sha256(digest)[:24]:
        raise ValueError("retrospective calibration ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("retrospective calibration receipt hash drifted")
    return data


def build_retrospective_multiple_search_calibration_v1(
    *,
    multiple_search_family_id: str,
    proposal_policy_sha256: str,
    calibration_manifest_sha256: str,
    target_recording_id: str,
    target_patient_id: str,
    target_exclusion_receipt_sha256: str,
    selection_count_cap: int,
    calibrated_claim_types: Sequence[str],
    eligible_measurement_kinds: Sequence[str],
    eligible_unit_roster_authorities: Sequence[str],
    precision_lower_bound: float,
    required_precision_lower_bound: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": RETROSPECTIVE_MULTIPLE_SEARCH_CALIBRATION_SCHEMA_VERSION,
        "calibration_id": "RETRO-CALIBRATION-PENDING",
        "multiple_search_family_id": multiple_search_family_id,
        "proposal_policy_sha256": proposal_policy_sha256,
        "source_split": "source_dev",
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "target_exclusion_binding": {
            "recording_id": target_recording_id,
            "patient_id": target_patient_id,
            "exclusion_receipt_sha256": target_exclusion_receipt_sha256,
        },
        "selection_count_cap": selection_count_cap,
        "calibrated_claim_types": sorted(set(calibrated_claim_types)),
        "eligible_measurement_kinds": sorted(set(eligible_measurement_kinds)),
        "eligible_unit_roster_authorities": sorted(
            set(eligible_unit_roster_authorities)
        ),
        "precision_lower_bound": float(precision_lower_bound),
        "required_precision_lower_bound": float(required_precision_lower_bound),
        "patient_disjoint_from_current_recording": True,
        "current_record_excluded": True,
        "frozen_before_current_recording": True,
        "multiple_search_scope_calibrated": True,
        "private_or_source_eval_labels_used": False,
        "passed": float(precision_lower_bound) >= float(required_precision_lower_bound),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["calibration_id"] = "RETROCAL-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_retrospective_multiple_search_calibration_v1(body)


def _measurement_core_sha256(measurement: Mapping[str, Any]) -> str:
    """Hash only early physical support; no offline selector field enters."""

    return _canonical_sha256(
        {
            "canonical_evidence_root_sha256": measurement[
                "canonical_evidence_root_sha256"
            ],
            "measurement_kind": measurement["measurement_kind"],
            "measurement_status": measurement["measurement_status"],
            "view_binding": measurement["view_binding"],
            "physical_support": measurement["physical_support"],
            "temporal_permissions": measurement["temporal_permissions"],
            "spatial_receipts": measurement["spatial_receipts"],
        }
    )


def validate_retrospective_verified_onset_leaf_v1(
    payload: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "leaf_id",
        "method_id",
        "recording_id",
        "patient_id",
        "event_id",
        "proposal_binding",
        "measurement_binding",
        "calibration_binding",
        "event_qualification_binding",
        "proposal_origin",
        "final_decision_available_recording_seconds",
        "physical_support_interval_recording_seconds",
        "measured_units",
        "causal_measurement_status",
        "candidate_permissions",
        "verification_status",
        "assertion_level",
        "report_authorization",
        "receipt_sha256",
    }
    data = _exact_keys(payload, required, "retrospective verified onset leaf")
    if (
        data["schema_version"] != RETROSPECTIVE_VERIFIED_ONSET_LEAF_SCHEMA_VERSION
        or data["method_id"] != RETROSPECTIVE_CAUSAL_ONSET_METHOD_ID
        or data["proposal_origin"]
        != "offline_retrospective_search_then_causal_view_remeasurement"
    ):
        raise ValueError("retrospective verified onset leaf identity drifted")
    _identifier(data["recording_id"], "recording ID")
    _identifier(data["patient_id"], "patient ID")
    _identifier(data["event_id"], "event ID")
    for field, names in (
        (
            "proposal_binding",
            {"proposal_id", "proposal_receipt_sha256"},
        ),
        (
            "measurement_binding",
            {
                "measurement_id",
                "measurement_receipt_sha256",
                "early_measurement_core_sha256",
            },
        ),
        (
            "calibration_binding",
            {"calibration_id", "calibration_receipt_sha256"},
        ),
        (
            "event_qualification_binding",
            {"qualification_id", "qualification_receipt_sha256"},
        ),
    ):
        binding = _exact_keys(data[field], names, field)
        for name, value in binding.items():
            if name.endswith("_sha256"):
                _sha256(value, name)
            else:
                _identifier(value, name)
    decision = _finite(
        data["final_decision_available_recording_seconds"],
        "final decision-available time",
        minimum=0.0,
    )
    interval = _interval(
        data["physical_support_interval_recording_seconds"],
        "verified physical support",
        maximum=float("inf"),
    )
    if decision + 1e-9 < interval[1]:
        raise ValueError("verified leaf decision predates its physical support")
    _unit_roster(data["measured_units"], "verified measured units")
    if data["causal_measurement_status"] not in _MEASUREMENT_STATUSES:
        raise ValueError("verified causal measurement status is invalid")
    permissions = _exact_keys(
        data["candidate_permissions"],
        {
            "candidate_onset_time_support",
            "candidate_onset_topography_support",
            "course_or_spread_support_used_as_positive_onset",
        },
        "verified candidate permissions",
    )
    if any(type(value) is not bool for value in permissions.values()):
        raise ValueError("verified candidate permissions must be boolean")
    if permissions["course_or_spread_support_used_as_positive_onset"] is not False:
        raise ValueError("late course/spread was promoted to positive onset")
    if data["verification_status"] not in {
        "candidate_verified_multiple_search_calibrated",
        "candidate_not_authorized_multiple_search_calibration_failed",
        "candidate_not_supported_by_causal_remeasurement",
    }:
        raise ValueError("retrospective verification status is invalid")
    passed = (
        data["verification_status"] == "candidate_verified_multiple_search_calibrated"
    )
    if permissions["candidate_onset_time_support"] is not passed:
        raise ValueError("retrospective time-support status is inconsistent")
    if (
        permissions["candidate_onset_topography_support"]
        and not permissions["candidate_onset_time_support"]
    ):
        raise ValueError("topography candidate lacks authorized onset-time support")
    if (
        permissions["candidate_onset_topography_support"]
        and len({row["unit_type"] for row in data["measured_units"]}) != 1
    ):
        raise ValueError("topography candidate mixes incomparable unit types")
    if passed and data["causal_measurement_status"] != "present":
        raise ValueError("verified onset candidate lacks a present causal measurement")
    if (
        data["verification_status"] == "candidate_not_supported_by_causal_remeasurement"
        and data["causal_measurement_status"] == "present"
    ):
        raise ValueError("unsupported onset status contradicts the causal measurement")
    if data["assertion_level"] != "model_candidate":
        raise ValueError("retrospective verified leaf exceeded candidate status")
    expected_report = {
        "trusted_registry_route_connected": False,
        "report_eligible": False,
        "qwen_or_renderer_authorized": False,
    }
    if data["report_authorization"] != expected_report:
        raise ValueError("retrospective verified leaf report route widened")
    digest = deepcopy(data)
    digest["leaf_id"] = "RETRO-VERIFIED-LEAF-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["leaf_id"] != "RETROLEAF-" + _canonical_sha256(digest)[:24]:
        raise ValueError("retrospective verified leaf ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("retrospective verified leaf receipt hash drifted")
    return data


def verify_retrospective_onset_search_proposal_v1(
    proposal_payload: Mapping[str, Any],
    measurement_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
) -> dict[str, Any]:
    proposal = validate_retrospective_onset_search_proposal_v1(dict(proposal_payload))
    measurement = validate_causal_early_leaf_remeasurement_v1(dict(measurement_payload))
    calibration = validate_retrospective_multiple_search_calibration_v1(
        dict(calibration_payload)
    )
    if measurement["proposal_id"] != proposal["proposal_id"]:
        raise ValueError("causal remeasurement is bound to another proposal")
    if (
        measurement["canonical_evidence_root_sha256"]
        != proposal["canonical_evidence_root_sha256"]
    ):
        raise ValueError("proposal and causal measurement use different EEG roots")
    search = proposal["search_scope"]
    support = measurement["physical_support"]
    search_interval = search["interval_recording_seconds"]
    measured_interval = support["interval_recording_seconds"]
    transform_interval = support["transform_input_interval_recording_seconds"]
    if (
        measured_interval[0] < search_interval[0] - 1e-9
        or measured_interval[1] > search_interval[1] + 1e-9
    ):
        raise ValueError("causal remeasurement escaped the proposed search interval")
    if (
        transform_interval[0] < search_interval[0] - 1e-9
        or transform_interval[1] > search_interval[1] + 1e-9
    ):
        raise ValueError(
            "causal transform support escaped the proposed search interval"
        )
    proposed_units = {
        (row["unit_type"], row["unit_id"]) for row in search["unit_candidates"]
    }
    measured_units = {
        (row["unit_type"], row["unit_id"]) for row in support["measured_units"]
    }
    if not measured_units.issubset(proposed_units):
        raise ValueError("causal remeasurement escaped the proposed unit roster")
    selector = proposal["offline_selector_binding"]
    if (
        support["signal_available_recording_seconds"]
        > selector["selector_available_recording_seconds"] + 1e-9
    ):
        raise ValueError("causal signal support was unavailable to the selector")
    if (
        calibration["multiple_search_family_id"] != search["multiple_search_family_id"]
        or calibration["proposal_policy_sha256"] != selector["proposal_policy_sha256"]
        or calibration["selection_count_cap"] < search["candidate_search_count"]
        or measurement["measurement_kind"]
        not in calibration["eligible_measurement_kinds"]
        or search["unit_roster_authority"]
        not in calibration["eligible_unit_roster_authorities"]
        or calibration["target_exclusion_binding"]["recording_id"]
        != proposal["recording_id"]
        or calibration["target_exclusion_binding"]["patient_id"]
        != proposal["patient_id"]
    ):
        raise ValueError("multiple-search calibration does not cover the proposal")
    calibration_passed = bool(calibration["passed"])
    passed = bool(
        calibration_passed
        and measurement["temporal_permissions"]["onset_time_support_eligible"]
        and "onset_time_support" in calibration["calibrated_claim_types"]
    )
    topography = bool(
        passed
        and "onset_topography_support" in calibration["calibrated_claim_types"]
        and measurement["spatial_receipts"]["onset_topography_support_eligible"]
        # An offline-selected or coarse subset can encode late saliency by the
        # mere choice of channels.  V1 therefore authorizes spatial comparison
        # only when every unit in an a-priori evaluable roster was remeasured.
        and search["unit_roster_authority"] == "predeclared_evaluable_unit_roster"
        and measured_units == proposed_units
        and len({unit_type for unit_type, _ in measured_units}) == 1
    )
    body: dict[str, Any] = {
        "schema_version": RETROSPECTIVE_VERIFIED_ONSET_LEAF_SCHEMA_VERSION,
        "leaf_id": "RETRO-VERIFIED-LEAF-PENDING",
        "method_id": RETROSPECTIVE_CAUSAL_ONSET_METHOD_ID,
        "recording_id": proposal["recording_id"],
        "patient_id": proposal["patient_id"],
        "event_id": proposal["event_id"],
        "proposal_binding": {
            "proposal_id": proposal["proposal_id"],
            "proposal_receipt_sha256": proposal["receipt_sha256"],
        },
        "measurement_binding": {
            "measurement_id": measurement["measurement_id"],
            "measurement_receipt_sha256": measurement["receipt_sha256"],
            "early_measurement_core_sha256": _measurement_core_sha256(measurement),
        },
        "calibration_binding": {
            "calibration_id": calibration["calibration_id"],
            "calibration_receipt_sha256": calibration["receipt_sha256"],
        },
        "event_qualification_binding": deepcopy(
            proposal["event_qualification_binding"]
        ),
        "proposal_origin": (
            "offline_retrospective_search_then_causal_view_remeasurement"
        ),
        "final_decision_available_recording_seconds": selector[
            "selector_available_recording_seconds"
        ],
        "physical_support_interval_recording_seconds": deepcopy(measured_interval),
        "measured_units": deepcopy(support["measured_units"]),
        "causal_measurement_status": measurement["measurement_status"],
        "candidate_permissions": {
            "candidate_onset_time_support": passed,
            "candidate_onset_topography_support": topography,
            "course_or_spread_support_used_as_positive_onset": False,
        },
        "verification_status": (
            "candidate_verified_multiple_search_calibrated"
            if passed
            else (
                "candidate_not_authorized_multiple_search_calibration_failed"
                if not calibration_passed
                else "candidate_not_supported_by_causal_remeasurement"
            )
        ),
        "assertion_level": "model_candidate",
        "report_authorization": {
            "trusted_registry_route_connected": (
                RETROSPECTIVE_VERIFIED_ONSET_TRUSTED_REPORT_ROUTE_CONNECTED
            ),
            "report_eligible": False,
            "qwen_or_renderer_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["leaf_id"] = "RETROLEAF-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_retrospective_verified_onset_leaf_v1(body)


__all__ = [
    "CAUSAL_EARLY_LEAF_REMEASUREMENT_SCHEMA_VERSION",
    "RETROSPECTIVE_CAUSAL_ONSET_METHOD_ID",
    "RETROSPECTIVE_MULTIPLE_SEARCH_CALIBRATION_SCHEMA_VERSION",
    "RETROSPECTIVE_ONSET_PROPOSAL_SCHEMA_VERSION",
    "RETROSPECTIVE_VERIFIED_ONSET_LEAF_SCHEMA_VERSION",
    "RETROSPECTIVE_VERIFIED_ONSET_TRUSTED_REPORT_ROUTE_CONNECTED",
    "build_causal_early_leaf_remeasurement_v1",
    "build_retrospective_multiple_search_calibration_v1",
    "build_retrospective_onset_search_proposal_v1",
    "validate_causal_early_leaf_remeasurement_v1",
    "validate_retrospective_multiple_search_calibration_v1",
    "validate_retrospective_onset_search_proposal_v1",
    "validate_retrospective_verified_onset_leaf_v1",
    "verify_retrospective_onset_search_proposal_v1",
]
