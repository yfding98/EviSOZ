"""Content-addressed per-event completion ledger for the EEG-only v2 route.

The detector-selected roster is the complete event universe.  Every selected
event must have exactly one terminal processing outcome; missing, duplicate,
or extra outcomes fail closed.  A single event failure therefore cannot be
hidden by shrinking the original detector count, and it need not cancel
successfully processed events from the same recording.

This module is deliberately independent from the production batch route and
from any event-Findings wire schema.  It never reads EDF files, annotations,
spreadsheets, clinical text, physician labels, or source-eval.  Stage hashes
are nullable content-address references for later evidence-flow accounting:
``None`` means that the stage was not materialized and must never be promoted
to success by the ledger.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


EVENT_PROCESSING_LEDGER_SCHEMA_VERSION = "clinical_eeg_event_processing_ledger_v2"

COMPLETED_FINDINGS = "completed_findings"
NOT_EVALUABLE_WINDOW_UNAVAILABLE = "not_evaluable_window_unavailable"
COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE = "completed_findings_onset_nonlocalizable"
ABSTAINED_SIGNAL_QUALITY = "abstained_signal_quality"
TECHNICAL_FAILURE_EVENT = "technical_failure_event"

EVENT_OUTCOME_STATUSES = (
    COMPLETED_FINDINGS,
    NOT_EVALUABLE_WINDOW_UNAVAILABLE,
    COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE,
    ABSTAINED_SIGNAL_QUALITY,
    TECHNICAL_FAILURE_EVENT,
)

SUCCESS_STATUSES = frozenset(
    {COMPLETED_FINDINGS, COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE}
)
SKIPPED_STATUSES = frozenset(
    {NOT_EVALUABLE_WINDOW_UNAVAILABLE, ABSTAINED_SIGNAL_QUALITY}
)
FAILURE_STATUSES = frozenset({TECHNICAL_FAILURE_EVENT})

ZERO_DETECTOR_EVENTS = "zero_detector_events"
ALL_EVENTS_COMPLETED_FINDINGS = "all_events_completed_findings"
MIXED_COMPLETED_AND_NOT_EVALUABLE = "mixed_completed_and_not_evaluable"
ALL_EVENTS_NOT_EVALUABLE = "all_events_not_evaluable"
PARTIAL_TECHNICAL_FAILURE = "partial_technical_failure"
ALL_EVENTS_TECHNICAL_FAILURE = "all_events_technical_failure"

RECORD_COMPLETION_CLASSES = (
    ZERO_DETECTOR_EVENTS,
    ALL_EVENTS_COMPLETED_FINDINGS,
    MIXED_COMPLETED_AND_NOT_EVALUABLE,
    ALL_EVENTS_NOT_EVALUABLE,
    PARTIAL_TECHNICAL_FAILURE,
    ALL_EVENTS_TECHNICAL_FAILURE,
)

_STAGE_HASH_KEYS = {
    "detector_selection_sha256",
    "adaptive_search_sha256",
    "adaptive_window_sha256",
    "quality_assessment_sha256",
    "event_findings_sha256",
    "waveform_manifest_sha256",
    "record_claim_sha256",
    "rendered_claim_sha256",
    "technical_failure_receipt_sha256",
}
_CALLER_STAGE_HASH_KEYS = _STAGE_HASH_KEYS - {"detector_selection_sha256"}

_ROSTER_INPUT_KEYS = {
    "event_id",
    "detector_candidate_id",
    "detector_event_ordinal",
    "candidate_start_seconds",
    "candidate_stop_seconds",
    "candidate_anchor_seconds",
    "source_detector_candidate_sha256",
}
_ROSTER_OUTPUT_KEYS = _ROSTER_INPUT_KEYS | {"roster_entry_sha256"}
_OUTCOME_INPUT_KEYS = {
    "event_id",
    "detector_candidate_id",
    "outcome_status",
    "reason_codes",
    "stage_hashes",
}
_OUTCOME_OUTPUT_KEYS = _OUTCOME_INPUT_KEYS | {
    "detector_event_ordinal",
    "eligibility",
    "stage_presence",
    "event_outcome_sha256",
}
_ELIGIBILITY_KEYS = {
    "findings_artifact_completed",
    "eligible_for_record_aggregation",
    "permitted_to_contribute_onset_positive_evidence",
    "technical_failure",
}
_SOURCE_BINDING_KEYS = {
    "recording_id",
    "recording_duration_seconds",
    "canonical_signal_sha256",
    "canonical_materialization_receipt_sha256",
    "detection_manifest_sha256",
    "detector_selected_roster_sha256",
}
_COUNT_KEYS = {
    "original_detector_event_count",
    "processed_outcome_count",
    "successful_event_count",
    "skipped_event_count",
    "failed_event_count",
    "completed_findings_event_count",
    "onset_nonlocalizable_findings_event_count",
    "window_unavailable_event_count",
    "signal_quality_abstention_event_count",
    "technical_failure_event_count",
    "eligible_for_record_aggregation_event_count",
    "permitted_onset_positive_event_count",
}
_COMPLETION_KEYS = {
    "completion_class",
    "report_materialization_route",
    "report_artifact_required",
    "invoke_nonempty_findings_aggregator",
    "dedicated_no_eligible_event_report_required",
    "technical_unassessable_report_required",
    "detector_zero_watchdog_required",
    "successful_event_ids",
    "skipped_event_ids",
    "failed_event_ids",
    "conclusion_may_use_event_ids",
    "forbidden_record_interpretations",
}
_LEDGER_KEYS = {
    "schema_version",
    "ledger_id",
    "source_binding",
    "detector_selected_roster",
    "event_outcomes",
    "event_status_counts",
    "record_counts",
    "record_completion_outcome",
    "eeg_only_firewall",
    "ledger_sha256",
}

_FIREWALL = {
    "eeg_signal_only": True,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "ecg_emg_eog_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "source_eval_used": False,
    "qwen_used": False,
    "diagnostic_claim_generated": False,
}

_FORBIDDEN_RECORD_INTERPRETATIONS = (
    "no_seizure",
    "normal_eeg",
    "negative_eeg",
    "generalized_onset_without_eeg_evidence",
    "diffuse_onset_without_eeg_evidence",
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_PENDING = "CONTENT-ADDRESS-PENDING"


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
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an opaque identifier")
    return value


def _sha256(value: object, context: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{context} must be finite and >= {minimum}")
    return result


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _content_address(value: Mapping[str, Any], field: str) -> str:
    body = deepcopy(dict(value))
    body[field] = _HASH_PENDING
    return _canonical_sha256(body)


def _validate_reason_codes(value: object, context: str, *, required: bool) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    result = [
        _identifier(item, f"{context}[{index}]") for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicates")
    if required and not result:
        raise ValueError(f"{context} requires at least one controlled reason code")
    return sorted(result)


def _roster_entry_hash(row: Mapping[str, Any]) -> str:
    body = deepcopy(dict(row))
    body.pop("roster_entry_sha256", None)
    return _canonical_sha256(body)


def _validate_roster_input_row(
    value: object,
    *,
    context: str,
    recording_duration_seconds: float,
) -> dict[str, Any]:
    row = _strict_object(value, _ROSTER_INPUT_KEYS, context)
    start = _finite(
        row["candidate_start_seconds"], f"{context}.candidate_start_seconds"
    )
    stop = _finite(row["candidate_stop_seconds"], f"{context}.candidate_stop_seconds")
    anchor = _finite(
        row["candidate_anchor_seconds"], f"{context}.candidate_anchor_seconds"
    )
    if not start < stop <= recording_duration_seconds:
        raise ValueError(f"{context} candidate interval is outside the recording")
    if not start <= anchor <= stop:
        raise ValueError(f"{context} candidate anchor is outside its interval")
    normalized = {
        "event_id": _identifier(row["event_id"], f"{context}.event_id"),
        "detector_candidate_id": _identifier(
            row["detector_candidate_id"], f"{context}.detector_candidate_id"
        ),
        "detector_event_ordinal": _integer(
            row["detector_event_ordinal"],
            f"{context}.detector_event_ordinal",
            minimum=1,
        ),
        "candidate_start_seconds": start,
        "candidate_stop_seconds": stop,
        "candidate_anchor_seconds": anchor,
        "source_detector_candidate_sha256": _sha256(
            row["source_detector_candidate_sha256"],
            f"{context}.source_detector_candidate_sha256",
        ),
    }
    return normalized


def _normalize_roster(
    value: object,
    *,
    recording_duration_seconds: float,
    output_rows: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("detector-selected roster must be an array")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if output_rows:
            full = _strict_object(
                item,
                _ROSTER_OUTPUT_KEYS,
                f"detector-selected roster[{index}]",
            )
            source = {key: full[key] for key in _ROSTER_INPUT_KEYS}
        else:
            full = None
            source = item
        normalized = _validate_roster_input_row(
            source,
            context=f"detector-selected roster[{index}]",
            recording_duration_seconds=recording_duration_seconds,
        )
        normalized["roster_entry_sha256"] = _roster_entry_hash(normalized)
        if (
            full is not None
            and full["roster_entry_sha256"] != normalized["roster_entry_sha256"]
        ):
            raise ValueError("detector-selected roster entry hash mismatch")
        rows.append(normalized)
    rows.sort(key=lambda item: int(item["detector_event_ordinal"]))
    event_ids = [str(item["event_id"]) for item in rows]
    candidate_ids = [str(item["detector_candidate_id"]) for item in rows]
    ordinals = [int(item["detector_event_ordinal"]) for item in rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("detector-selected roster contains duplicate event IDs")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("detector-selected roster contains duplicate candidate IDs")
    if ordinals != list(range(1, len(rows) + 1)):
        raise ValueError("detector-selected roster ordinals must be contiguous from 1")
    return rows


def _validate_caller_stage_hashes(value: object, context: str) -> dict[str, str | None]:
    data = _strict_object(value, _CALLER_STAGE_HASH_KEYS, context)
    return {
        key: _sha256(data[key], f"{context}.{key}", nullable=True)
        for key in sorted(_CALLER_STAGE_HASH_KEYS)
    }


def _derived_eligibility(status: str) -> dict[str, bool]:
    return {
        "findings_artifact_completed": status
        in {
            COMPLETED_FINDINGS,
            COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE,
            ABSTAINED_SIGNAL_QUALITY,
        },
        "eligible_for_record_aggregation": status in SUCCESS_STATUSES,
        "permitted_to_contribute_onset_positive_evidence": status == COMPLETED_FINDINGS,
        "technical_failure": status == TECHNICAL_FAILURE_EVENT,
    }


def _derived_stage_presence(stage_hashes: Mapping[str, Any]) -> dict[str, bool]:
    return {
        key.removesuffix("_sha256"): stage_hashes[key] is not None
        for key in sorted(_STAGE_HASH_KEYS)
    }


def _validate_stage_semantics(
    status: str,
    stage_hashes: Mapping[str, str | None],
    *,
    context: str,
) -> None:
    required: set[str]
    forbidden: set[str]
    if status == COMPLETED_FINDINGS:
        required = {
            "detector_selection_sha256",
            "adaptive_search_sha256",
            "adaptive_window_sha256",
            "quality_assessment_sha256",
            "event_findings_sha256",
        }
        forbidden = {"technical_failure_receipt_sha256"}
    elif status == COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE:
        required = {
            "detector_selection_sha256",
            "adaptive_search_sha256",
            "adaptive_window_sha256",
            "quality_assessment_sha256",
            "event_findings_sha256",
        }
        forbidden = {"technical_failure_receipt_sha256"}
    elif status == NOT_EVALUABLE_WINDOW_UNAVAILABLE:
        required = {"detector_selection_sha256"}
        forbidden = {
            "adaptive_window_sha256",
            "quality_assessment_sha256",
            "event_findings_sha256",
            "waveform_manifest_sha256",
            "record_claim_sha256",
            "rendered_claim_sha256",
            "technical_failure_receipt_sha256",
        }
    elif status == ABSTAINED_SIGNAL_QUALITY:
        required = {
            "detector_selection_sha256",
            "adaptive_search_sha256",
            "adaptive_window_sha256",
            "quality_assessment_sha256",
            "event_findings_sha256",
        }
        forbidden = {
            "record_claim_sha256",
            "rendered_claim_sha256",
            "technical_failure_receipt_sha256",
        }
    elif status == TECHNICAL_FAILURE_EVENT:
        required = {
            "detector_selection_sha256",
            "technical_failure_receipt_sha256",
        }
        forbidden = {
            "event_findings_sha256",
            "waveform_manifest_sha256",
            "record_claim_sha256",
            "rendered_claim_sha256",
        }
    else:
        raise ValueError(f"{context} has an unknown event outcome status")
    missing = sorted(key for key in required if stage_hashes[key] is None)
    if missing:
        raise ValueError(f"{context} status requires stage hashes: {missing}")
    present_forbidden = sorted(
        key for key in forbidden if stage_hashes[key] is not None
    )
    if present_forbidden:
        raise ValueError(f"{context} status forbids stage hashes: {present_forbidden}")

    # Ordered stage reachability.  A downstream content hash cannot resurrect
    # a stage that has no upstream receipt.
    dependencies = {
        "adaptive_window_sha256": "adaptive_search_sha256",
        "quality_assessment_sha256": "adaptive_window_sha256",
        "event_findings_sha256": "quality_assessment_sha256",
        "waveform_manifest_sha256": "event_findings_sha256",
        "record_claim_sha256": "event_findings_sha256",
        "rendered_claim_sha256": "record_claim_sha256",
    }
    for downstream, upstream in dependencies.items():
        if stage_hashes[downstream] is not None and stage_hashes[upstream] is None:
            raise ValueError(f"{context} {downstream} cannot exist without {upstream}")


def _validate_outcome_input(
    value: object,
    *,
    context: str,
    roster_by_event: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row = _strict_object(value, _OUTCOME_INPUT_KEYS, context)
    event_id = _identifier(row["event_id"], f"{context}.event_id")
    candidate_id = _identifier(
        row["detector_candidate_id"], f"{context}.detector_candidate_id"
    )
    roster = roster_by_event.get(event_id)
    if roster is None:
        raise ValueError(f"{context} references an event absent from detector roster")
    if candidate_id != roster["detector_candidate_id"]:
        raise ValueError(f"{context} detector candidate binding mismatch")
    status = row["outcome_status"]
    if status not in EVENT_OUTCOME_STATUSES:
        raise ValueError(f"{context}.outcome_status is invalid")
    reasons = _validate_reason_codes(
        row["reason_codes"],
        f"{context}.reason_codes",
        required=status != COMPLETED_FINDINGS,
    )
    caller_hashes = _validate_caller_stage_hashes(
        row["stage_hashes"], f"{context}.stage_hashes"
    )
    stage_hashes = {
        "detector_selection_sha256": str(roster["roster_entry_sha256"]),
        **caller_hashes,
    }
    _validate_stage_semantics(status, stage_hashes, context=context)
    normalized: dict[str, Any] = {
        "event_id": event_id,
        "detector_candidate_id": candidate_id,
        "detector_event_ordinal": int(roster["detector_event_ordinal"]),
        "outcome_status": status,
        "reason_codes": reasons,
        "stage_hashes": stage_hashes,
        "eligibility": _derived_eligibility(status),
        "stage_presence": _derived_stage_presence(stage_hashes),
        "event_outcome_sha256": _HASH_PENDING,
    }
    normalized["event_outcome_sha256"] = _content_address(
        normalized, "event_outcome_sha256"
    )
    return normalized


def _normalize_outcomes(
    value: object,
    *,
    roster: Sequence[Mapping[str, Any]],
    output_rows: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("event outcomes must be an array")
    roster_by_event = {str(row["event_id"]): row for row in roster}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        context = f"event outcomes[{index}]"
        if output_rows:
            full = _strict_object(item, _OUTCOME_OUTPUT_KEYS, context)
            source = {key: full[key] for key in _OUTCOME_INPUT_KEYS}
            full_stage_hashes = _strict_object(
                full["stage_hashes"], _STAGE_HASH_KEYS, f"{context}.stage_hashes"
            )
            source["stage_hashes"] = {
                key: full_stage_hashes[key] for key in _CALLER_STAGE_HASH_KEYS
            }
        else:
            full = None
            source = item
        normalized = _validate_outcome_input(
            source,
            context=context,
            roster_by_event=roster_by_event,
        )
        if full is not None:
            if (
                full_stage_hashes["detector_selection_sha256"]
                != normalized["stage_hashes"]["detector_selection_sha256"]
            ):
                raise ValueError(f"{context} detector selection stage hash mismatch")
            if full["detector_event_ordinal"] != normalized["detector_event_ordinal"]:
                raise ValueError(f"{context} detector ordinal mismatch")
            eligibility = _strict_object(
                full["eligibility"], _ELIGIBILITY_KEYS, f"{context}.eligibility"
            )
            if eligibility != normalized["eligibility"]:
                raise ValueError(f"{context} eligibility is not status-derived")
            expected_presence = normalized["stage_presence"]
            if full["stage_presence"] != expected_presence:
                raise ValueError(f"{context} stage presence does not match hashes")
            if full["event_outcome_sha256"] != normalized["event_outcome_sha256"]:
                raise ValueError(f"{context} event outcome hash mismatch")
        rows.append(normalized)
    event_ids = [str(row["event_id"]) for row in rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event outcomes contain duplicate event IDs")
    expected_ids = {str(row["event_id"]) for row in roster}
    actual_ids = set(event_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "event outcomes must exactly cover detector roster; "
            f"missing={missing}, extra={extra}"
        )
    rows.sort(key=lambda item: int(item["detector_event_ordinal"]))
    return rows


def _status_counts(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        status: sum(row["outcome_status"] == status for row in outcomes)
        for status in EVENT_OUTCOME_STATUSES
    }


def _record_counts(
    roster: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    status = _status_counts(outcomes)
    return {
        "original_detector_event_count": len(roster),
        "processed_outcome_count": len(outcomes),
        "successful_event_count": sum(status[item] for item in SUCCESS_STATUSES),
        "skipped_event_count": sum(status[item] for item in SKIPPED_STATUSES),
        "failed_event_count": sum(status[item] for item in FAILURE_STATUSES),
        "completed_findings_event_count": status[COMPLETED_FINDINGS],
        "onset_nonlocalizable_findings_event_count": status[
            COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE
        ],
        "window_unavailable_event_count": status[NOT_EVALUABLE_WINDOW_UNAVAILABLE],
        "signal_quality_abstention_event_count": status[ABSTAINED_SIGNAL_QUALITY],
        "technical_failure_event_count": status[TECHNICAL_FAILURE_EVENT],
        "eligible_for_record_aggregation_event_count": sum(
            row["eligibility"]["eligible_for_record_aggregation"] for row in outcomes
        ),
        "permitted_onset_positive_event_count": sum(
            row["eligibility"]["permitted_to_contribute_onset_positive_evidence"]
            for row in outcomes
        ),
    }


def _record_completion_outcome(
    outcomes: Sequence[Mapping[str, Any]], counts: Mapping[str, int]
) -> dict[str, Any]:
    original = int(counts["original_detector_event_count"])
    success = int(counts["successful_event_count"])
    skipped = int(counts["skipped_event_count"])
    failed = int(counts["failed_event_count"])
    if original == 0:
        completion_class = ZERO_DETECTOR_EVENTS
        route = "zero_detector_event_fallback"
    elif failed == original:
        completion_class = ALL_EVENTS_TECHNICAL_FAILURE
        route = "technical_unassessable_fallback"
    elif failed > 0:
        completion_class = PARTIAL_TECHNICAL_FAILURE
        route = (
            "partial_failure_with_available_findings"
            if success > 0
            else "partial_failure_without_available_findings"
        )
    elif skipped == original:
        completion_class = ALL_EVENTS_NOT_EVALUABLE
        route = "no_eligible_event_evidence_fallback"
    elif success == original:
        completion_class = ALL_EVENTS_COMPLETED_FINDINGS
        route = "aggregate_completed_findings"
    else:
        completion_class = MIXED_COMPLETED_AND_NOT_EVALUABLE
        route = "aggregate_available_findings_with_skipped_events"
    successful_ids = sorted(
        str(row["event_id"])
        for row in outcomes
        if row["outcome_status"] in SUCCESS_STATUSES
    )
    skipped_ids = sorted(
        str(row["event_id"])
        for row in outcomes
        if row["outcome_status"] in SKIPPED_STATUSES
    )
    failed_ids = sorted(
        str(row["event_id"])
        for row in outcomes
        if row["outcome_status"] in FAILURE_STATUSES
    )
    return {
        "completion_class": completion_class,
        "report_materialization_route": route,
        "report_artifact_required": True,
        "invoke_nonempty_findings_aggregator": success > 0,
        "dedicated_no_eligible_event_report_required": success == 0
        and completion_class
        in {
            ZERO_DETECTOR_EVENTS,
            ALL_EVENTS_NOT_EVALUABLE,
            PARTIAL_TECHNICAL_FAILURE,
        },
        "technical_unassessable_report_required": completion_class
        == ALL_EVENTS_TECHNICAL_FAILURE,
        "detector_zero_watchdog_required": completion_class == ZERO_DETECTOR_EVENTS,
        "successful_event_ids": successful_ids,
        "skipped_event_ids": skipped_ids,
        "failed_event_ids": failed_ids,
        "conclusion_may_use_event_ids": successful_ids,
        "forbidden_record_interpretations": list(_FORBIDDEN_RECORD_INTERPRETATIONS),
    }


def _ledger_id(source_binding: Mapping[str, Any]) -> str:
    identity = {
        "recording_id": source_binding["recording_id"],
        "canonical_signal_sha256": source_binding["canonical_signal_sha256"],
        "detection_manifest_sha256": source_binding["detection_manifest_sha256"],
        "detector_selected_roster_sha256": source_binding[
            "detector_selected_roster_sha256"
        ],
    }
    return f"EPL2-{_canonical_sha256(identity)[:24]}"


def _validate_source_binding(value: object) -> dict[str, Any]:
    data = _strict_object(value, _SOURCE_BINDING_KEYS, "event ledger source_binding")
    return {
        "recording_id": _identifier(
            data["recording_id"], "source_binding.recording_id"
        ),
        "recording_duration_seconds": _finite(
            data["recording_duration_seconds"],
            "source_binding.recording_duration_seconds",
            minimum=1e-12,
        ),
        "canonical_signal_sha256": _sha256(
            data["canonical_signal_sha256"], "source_binding.canonical_signal_sha256"
        ),
        "canonical_materialization_receipt_sha256": _sha256(
            data["canonical_materialization_receipt_sha256"],
            "source_binding.canonical_materialization_receipt_sha256",
        ),
        "detection_manifest_sha256": _sha256(
            data["detection_manifest_sha256"],
            "source_binding.detection_manifest_sha256",
        ),
        "detector_selected_roster_sha256": _sha256(
            data["detector_selected_roster_sha256"],
            "source_binding.detector_selected_roster_sha256",
        ),
    }


def _validate_counts(
    value: object, expected: Mapping[str, int], context: str
) -> dict[str, int]:
    data = _strict_object(value, _COUNT_KEYS, context)
    normalized = {
        key: _integer(data[key], f"{context}.{key}") for key in sorted(_COUNT_KEYS)
    }
    if normalized != dict(expected):
        raise ValueError(f"{context} does not match the event partition")
    if (
        normalized["successful_event_count"]
        + normalized["skipped_event_count"]
        + normalized["failed_event_count"]
        != normalized["original_detector_event_count"]
    ):
        raise ValueError(f"{context} success/skip/failure partition is incomplete")
    return normalized


def _validate_completion(value: object, expected: Mapping[str, Any]) -> dict[str, Any]:
    data = _strict_object(value, _COMPLETION_KEYS, "record completion outcome")
    if data["completion_class"] not in RECORD_COMPLETION_CLASSES:
        raise ValueError("record completion outcome class is invalid")
    for key in (
        "report_artifact_required",
        "invoke_nonempty_findings_aggregator",
        "dedicated_no_eligible_event_report_required",
        "technical_unassessable_report_required",
        "detector_zero_watchdog_required",
    ):
        if type(data[key]) is not bool:
            raise TypeError(f"record completion outcome.{key} must be boolean")
    for key in (
        "successful_event_ids",
        "skipped_event_ids",
        "failed_event_ids",
        "conclusion_may_use_event_ids",
        "forbidden_record_interpretations",
    ):
        if not isinstance(data[key], list):
            raise TypeError(f"record completion outcome.{key} must be an array")
    if data != dict(expected):
        raise ValueError(
            "record completion outcome does not match event counts/statuses"
        )
    return deepcopy(data)


def build_event_processing_ledger_v2(
    *,
    recording_id: str,
    recording_duration_seconds: float,
    canonical_signal_sha256: str,
    canonical_materialization_receipt_sha256: str,
    detection_manifest_sha256: str,
    detector_selected_roster: Sequence[Mapping[str, Any]],
    event_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a complete detector-roster partition and record outcome."""

    duration = _finite(
        recording_duration_seconds, "recording_duration_seconds", minimum=1e-12
    )
    roster = _normalize_roster(
        list(detector_selected_roster),
        recording_duration_seconds=duration,
        output_rows=False,
    )
    roster_sha256 = _canonical_sha256(roster)
    outcomes = _normalize_outcomes(
        list(event_outcomes), roster=roster, output_rows=False
    )
    source_binding = {
        "recording_id": _identifier(recording_id, "recording_id"),
        "recording_duration_seconds": duration,
        "canonical_signal_sha256": _sha256(
            canonical_signal_sha256, "canonical_signal_sha256"
        ),
        "canonical_materialization_receipt_sha256": _sha256(
            canonical_materialization_receipt_sha256,
            "canonical_materialization_receipt_sha256",
        ),
        "detection_manifest_sha256": _sha256(
            detection_manifest_sha256, "detection_manifest_sha256"
        ),
        "detector_selected_roster_sha256": roster_sha256,
    }
    status_counts = _status_counts(outcomes)
    counts = _record_counts(roster, outcomes)
    completion = _record_completion_outcome(outcomes, counts)
    ledger: dict[str, Any] = {
        "schema_version": EVENT_PROCESSING_LEDGER_SCHEMA_VERSION,
        "ledger_id": _ledger_id(source_binding),
        "source_binding": source_binding,
        "detector_selected_roster": roster,
        "event_outcomes": outcomes,
        "event_status_counts": status_counts,
        "record_counts": counts,
        "record_completion_outcome": completion,
        "eeg_only_firewall": dict(_FIREWALL),
        "ledger_sha256": _HASH_PENDING,
    }
    ledger["ledger_sha256"] = _content_address(ledger, "ledger_sha256")
    return validate_event_processing_ledger_v2(ledger)


def validate_event_processing_ledger_v2(value: object) -> dict[str, Any]:
    """Validate hashes, exact roster coverage, stage truth, and record class."""

    data = _strict_object(value, _LEDGER_KEYS, "event processing ledger v2")
    if data["schema_version"] != EVENT_PROCESSING_LEDGER_SCHEMA_VERSION:
        raise ValueError("event processing ledger v2 schema_version mismatch")
    source = _validate_source_binding(data["source_binding"])
    roster = _normalize_roster(
        data["detector_selected_roster"],
        recording_duration_seconds=float(source["recording_duration_seconds"]),
        output_rows=True,
    )
    roster_hash = _canonical_sha256(roster)
    if source["detector_selected_roster_sha256"] != roster_hash:
        raise ValueError("source binding detector roster hash mismatch")
    outcomes = _normalize_outcomes(
        data["event_outcomes"], roster=roster, output_rows=True
    )
    expected_status_counts = _status_counts(outcomes)
    status_counts = _strict_object(
        data["event_status_counts"],
        set(EVENT_OUTCOME_STATUSES),
        "event_status_counts",
    )
    normalized_status_counts = {
        key: _integer(status_counts[key], f"event_status_counts.{key}")
        for key in EVENT_OUTCOME_STATUSES
    }
    if normalized_status_counts != expected_status_counts:
        raise ValueError("event_status_counts does not match event outcomes")
    expected_counts = _record_counts(roster, outcomes)
    counts = _validate_counts(data["record_counts"], expected_counts, "record_counts")
    expected_completion = _record_completion_outcome(outcomes, counts)
    completion = _validate_completion(
        data["record_completion_outcome"], expected_completion
    )
    firewall = _strict_object(
        data["eeg_only_firewall"], set(_FIREWALL), "eeg_only_firewall"
    )
    if firewall != _FIREWALL:
        raise ValueError("event processing ledger EEG-only firewall mismatch")
    if data["ledger_id"] != _ledger_id(source):
        raise ValueError("event processing ledger ID does not match source binding")
    ledger_sha = _sha256(data["ledger_sha256"], "event processing ledger.ledger_sha256")
    normalized: dict[str, Any] = {
        "schema_version": EVENT_PROCESSING_LEDGER_SCHEMA_VERSION,
        "ledger_id": data["ledger_id"],
        "source_binding": source,
        "detector_selected_roster": roster,
        "event_outcomes": outcomes,
        "event_status_counts": normalized_status_counts,
        "record_counts": counts,
        "record_completion_outcome": completion,
        "eeg_only_firewall": dict(_FIREWALL),
        "ledger_sha256": ledger_sha,
    }
    if ledger_sha != _content_address(normalized, "ledger_sha256"):
        raise ValueError("event processing ledger content hash mismatch")
    return normalized


__all__ = [
    "ABSTAINED_SIGNAL_QUALITY",
    "ALL_EVENTS_COMPLETED_FINDINGS",
    "ALL_EVENTS_NOT_EVALUABLE",
    "ALL_EVENTS_TECHNICAL_FAILURE",
    "COMPLETED_FINDINGS",
    "COMPLETED_FINDINGS_ONSET_NONLOCALIZABLE",
    "EVENT_OUTCOME_STATUSES",
    "EVENT_PROCESSING_LEDGER_SCHEMA_VERSION",
    "MIXED_COMPLETED_AND_NOT_EVALUABLE",
    "NOT_EVALUABLE_WINDOW_UNAVAILABLE",
    "PARTIAL_TECHNICAL_FAILURE",
    "TECHNICAL_FAILURE_EVENT",
    "ZERO_DETECTOR_EVENTS",
    "build_event_processing_ledger_v2",
    "validate_event_processing_ledger_v2",
]
