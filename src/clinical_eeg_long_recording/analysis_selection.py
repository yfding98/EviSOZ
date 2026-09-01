"""Signal-only partition of detector candidates for downstream SOZ analysis.

The detector manifest is immutable: a candidate that cannot be replayed safely
is neither deleted nor relabelled as a non-seizure.  Instead this artifact
partitions every detector-selected candidate into exactly one of two sets:

* ``analyzable`` -- a content-bound preprocessing window exists; or
* ``rejected_signal_eligibility`` -- signal/time-support eligibility prevented
  downstream ranking and report-fact extraction.

No annotation, spreadsheet, clinical history, target or ground truth is an
input.  Rejection receipts intentionally contain closed reason codes and
structured signal-QC fields rather than exception prose.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .schema import (
    CANDIDATE_SEMANTICS,
    _finite_number,
    _identifier,
    _integer,
    _same_number,
    _sha256,
    _strict_object,
    canonical_payload_sha256,
    validate_long_term_seizure_detection_manifest,
)


ANALYSIS_SELECTION_SCHEMA_VERSION = "long_term_eeg_analysis_selection_v1"
ANALYSIS_REJECTION_SCHEMA_VERSION = "long_term_eeg_analysis_rejection_v1"
ANALYSIS_SELECTION_ID_PREFIX = "EEG-ANSEL-"
ANALYSIS_REJECTION_ID_PREFIX = "EEG-ANREJ-"

ANALYSIS_DISPOSITIONS = (
    "analyzable",
    "rejected_signal_eligibility",
)

ELIGIBILITY_REASON_CODES = (
    "ambiguous_standard19",
    "invalid_sfreq",
    "mixed_sfreq",
    "sample_count_mismatch",
    "insufficient_warmup",
    "insufficient_post",
    "payload_shape",
    "signal_qc",
    "reference_or_signal_contract",
)

QC_STAGES = (
    "pre_filter_raw_physical",
    "post_preprocessing_physical_contract",
    "not_available_for_eligibility_code",
)

QC_FAILED_CHECKS = (
    "flatline_run",
    "extreme_value_run",
    "downstream_physical_signal_contract",
)


def _nullable_sha256(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, context)


def _nullable_nonnegative_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, context, minimum=0.0)


def _closed_string_list(
    value: object,
    *,
    allowed: Sequence[str] | None,
    context: str,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip():
            raise TypeError(f"{context} values must be non-empty trimmed strings")
        if allowed is not None and item not in allowed:
            raise ValueError(f"{context} contains an unsupported value")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicates")
    return result


def _signal_qc_details(value: object) -> dict[str, Any]:
    data = _strict_object(
        value,
        required=(
            "qc_stage",
            "failed_checks",
            "flatline_channels",
            "clipping_channels",
            "flatline_run_threshold_seconds",
            "clipping_run_threshold_seconds",
            "qc_tolerance_volts",
            "edf_gap_annotations_used",
        ),
        context="analysis rejection signal_qc_details",
    )
    stage = data["qc_stage"]
    if stage not in QC_STAGES:
        raise ValueError("analysis rejection QC stage is unsupported")
    failed_checks = _closed_string_list(
        data["failed_checks"],
        allowed=QC_FAILED_CHECKS,
        context="analysis rejection failed_checks",
    )
    flatline = _closed_string_list(
        data["flatline_channels"],
        allowed=None,
        context="analysis rejection flatline_channels",
    )
    clipping = _closed_string_list(
        data["clipping_channels"],
        allowed=None,
        context="analysis rejection clipping_channels",
    )
    for channel in (*flatline, *clipping):
        _identifier(channel, "analysis rejection channel")
    if data["edf_gap_annotations_used"] is not False:
        raise ValueError("analysis selection must not use EDF gap annotations")
    details = {
        "qc_stage": stage,
        "failed_checks": failed_checks,
        "flatline_channels": flatline,
        "clipping_channels": clipping,
        "flatline_run_threshold_seconds": _nullable_nonnegative_number(
            data["flatline_run_threshold_seconds"],
            "analysis rejection flatline threshold",
        ),
        "clipping_run_threshold_seconds": _nullable_nonnegative_number(
            data["clipping_run_threshold_seconds"],
            "analysis rejection clipping threshold",
        ),
        "qc_tolerance_volts": _nullable_nonnegative_number(
            data["qc_tolerance_volts"],
            "analysis rejection QC tolerance",
        ),
        "edf_gap_annotations_used": False,
    }
    if stage == "not_available_for_eligibility_code" and any(
        (
            failed_checks,
            flatline,
            clipping,
            details["flatline_run_threshold_seconds"] is not None,
            details["clipping_run_threshold_seconds"] is not None,
            details["qc_tolerance_volts"] is not None,
        )
    ):
        raise ValueError("non-QC eligibility rejection must not invent QC details")
    return details


def validate_analysis_rejection_receipt(payload: object) -> dict[str, Any]:
    """Validate one content-addressed signal-only rejection receipt."""

    data = _strict_object(
        payload,
        required=(
            "schema_version",
            "rejection_receipt_id",
            "candidate_id",
            "eeg_event_id",
            "candidate_anchor_offset_seconds",
            "eligibility_code",
            "reason_code",
            "signal_qc_details",
            "scope_receipt",
            "claim_boundary",
        ),
        context="analysis rejection receipt",
    )
    if data["schema_version"] != ANALYSIS_REJECTION_SCHEMA_VERSION:
        raise ValueError("analysis rejection receipt schema mismatch")
    receipt_id = _identifier(
        data["rejection_receipt_id"], "analysis rejection receipt ID"
    )
    candidate_id = _identifier(data["candidate_id"], "analysis rejection candidate")
    event_id = _identifier(data["eeg_event_id"], "analysis rejection EEG event")
    anchor = _finite_number(
        data["candidate_anchor_offset_seconds"],
        "analysis rejection anchor",
        minimum=0.0,
    )
    eligibility_code = data["eligibility_code"]
    reason_code = data["reason_code"]
    if eligibility_code not in ELIGIBILITY_REASON_CODES or reason_code != (
        "edf_event_eligibility_" + str(eligibility_code)
    ):
        raise ValueError("analysis rejection eligibility/reason code is unsupported")
    details = _signal_qc_details(data["signal_qc_details"])
    if eligibility_code == "signal_qc":
        if details["qc_stage"] == "not_available_for_eligibility_code":
            raise ValueError("signal_qc rejection requires a QC-stage receipt")
    elif details["qc_stage"] != "not_available_for_eligibility_code":
        raise ValueError("non-QC eligibility rejection must not claim QC measurements")
    scope = _strict_object(
        data["scope_receipt"],
        required=(
            "eeg_signal_or_physical_metadata_used",
            "edf_annotations_used",
            "excel_used",
            "clinical_context_used",
            "labels_or_ground_truth_used",
        ),
        context="analysis rejection scope receipt",
    )
    if scope != {
        "eeg_signal_or_physical_metadata_used": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
    }:
        raise ValueError("analysis rejection violates the EEG-only scope")
    claims = _strict_object(
        data["claim_boundary"],
        required=(
            "rejection_is_not_no_seizure",
            "candidate_is_confirmed_seizure",
            "candidate_is_confirmed_nonseizure",
            "soz_conclusion_generated",
        ),
        context="analysis rejection claim boundary",
    )
    if claims != {
        "rejection_is_not_no_seizure": True,
        "candidate_is_confirmed_seizure": False,
        "candidate_is_confirmed_nonseizure": False,
        "soz_conclusion_generated": False,
    }:
        raise ValueError("analysis rejection crossed its claim boundary")
    result = {
        "schema_version": ANALYSIS_REJECTION_SCHEMA_VERSION,
        "rejection_receipt_id": receipt_id,
        "candidate_id": candidate_id,
        "eeg_event_id": event_id,
        "candidate_anchor_offset_seconds": anchor,
        "eligibility_code": eligibility_code,
        "reason_code": reason_code,
        "signal_qc_details": details,
        "scope_receipt": scope,
        "claim_boundary": claims,
    }
    digest_source = deepcopy(result)
    digest_source["rejection_receipt_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = ANALYSIS_REJECTION_ID_PREFIX + canonical_payload_sha256(
        digest_source
    )[:20]
    if receipt_id != expected_id:
        raise ValueError("analysis rejection receipt ID does not bind its content")
    return result


def validate_long_term_eeg_analysis_selection(payload: object) -> dict[str, Any]:
    """Validate a content-addressed detector-candidate partition."""

    data = _strict_object(
        payload,
        required=(
            "schema_version",
            "selection_id",
            "recording_id",
            "patient_pseudonym",
            "source_signal_sha256",
            "recording_duration_seconds",
            "detection_manifest_sha256",
            "event_id_assignment_sha256",
            "candidate_semantics",
            "detector_selected_count",
            "analyzable_count",
            "rejected_count",
            "events",
            "scope_receipt",
        ),
        context="long-term EEG analysis selection",
    )
    if data["schema_version"] != ANALYSIS_SELECTION_SCHEMA_VERSION:
        raise ValueError("analysis selection schema mismatch")
    selection_id = _identifier(data["selection_id"], "analysis selection ID")
    recording_id = _identifier(data["recording_id"], "analysis selection recording")
    patient = _identifier(data["patient_pseudonym"], "analysis selection patient")
    source_hash = _sha256(
        data["source_signal_sha256"], "analysis selection source hash"
    )
    duration = _finite_number(
        data["recording_duration_seconds"],
        "analysis selection recording duration",
        exclusive_minimum=0.0,
    )
    detection_hash = _sha256(
        data["detection_manifest_sha256"], "analysis selection detection hash"
    )
    assignment_hash = _sha256(
        data["event_id_assignment_sha256"], "analysis selection assignment hash"
    )
    if data["candidate_semantics"] != CANDIDATE_SEMANTICS:
        raise ValueError("analysis selection promotes detector candidates")
    selected_count = _integer(
        data["detector_selected_count"],
        "analysis selection selected count",
        minimum=0,
    )
    analyzable_count = _integer(
        data["analyzable_count"], "analysis selection analyzable count", minimum=0
    )
    rejected_count = _integer(
        data["rejected_count"], "analysis selection rejected count", minimum=0
    )
    raw_events = data["events"]
    if not isinstance(raw_events, list) or len(raw_events) != selected_count:
        raise ValueError("analysis selection events do not match selected count")
    events: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    event_ids: set[str] = set()
    for index, raw in enumerate(raw_events):
        event = _strict_object(
            raw,
            required=(
                "candidate_id",
                "eeg_event_id",
                "candidate_anchor_offset_seconds",
                "analysis_disposition",
                "pre_ranking_window_receipt_sha256",
                "processed_window_sha256",
                "preprocessing_receipt_sha256",
                "rejection_receipt",
            ),
            context=f"analysis selection event {index}",
        )
        candidate_id = _identifier(
            event["candidate_id"], "analysis selection candidate"
        )
        event_id = _identifier(event["eeg_event_id"], "analysis selection EEG event")
        anchor = _finite_number(
            event["candidate_anchor_offset_seconds"],
            "analysis selection candidate anchor",
            minimum=0.0,
            maximum=duration,
        )
        disposition = event["analysis_disposition"]
        if disposition not in ANALYSIS_DISPOSITIONS:
            raise ValueError("analysis selection disposition is unsupported")
        window_hash = _nullable_sha256(
            event["pre_ranking_window_receipt_sha256"],
            "analysis selection window receipt hash",
        )
        processed_hash = _nullable_sha256(
            event["processed_window_sha256"],
            "analysis selection processed window hash",
        )
        preprocessing_hash = _nullable_sha256(
            event["preprocessing_receipt_sha256"],
            "analysis selection preprocessing hash",
        )
        rejection_raw = event["rejection_receipt"]
        if disposition == "analyzable":
            if None in (window_hash, processed_hash, preprocessing_hash):
                raise ValueError("analyzable selection event lacks signal hashes")
            if rejection_raw is not None:
                raise ValueError("analyzable selection event has a rejection receipt")
            rejection = None
        else:
            if any(
                item is not None
                for item in (window_hash, processed_hash, preprocessing_hash)
            ):
                raise ValueError("rejected selection event claims an analyzable window")
            rejection = validate_analysis_rejection_receipt(rejection_raw)
            if (
                rejection["candidate_id"] != candidate_id
                or rejection["eeg_event_id"] != event_id
                or not _same_number(
                    rejection["candidate_anchor_offset_seconds"], anchor
                )
            ):
                raise ValueError("analysis rejection identity differs from its event")
        if candidate_id in candidate_ids or event_id in event_ids:
            raise ValueError("analysis selection repeats candidate or EEG event identity")
        candidate_ids.add(candidate_id)
        event_ids.add(event_id)
        events.append(
            {
                "candidate_id": candidate_id,
                "eeg_event_id": event_id,
                "candidate_anchor_offset_seconds": anchor,
                "analysis_disposition": disposition,
                "pre_ranking_window_receipt_sha256": window_hash,
                "processed_window_sha256": processed_hash,
                "preprocessing_receipt_sha256": preprocessing_hash,
                "rejection_receipt": rejection,
            }
        )
    if events != sorted(
        events,
        key=lambda item: (
            item["candidate_anchor_offset_seconds"],
            item["eeg_event_id"],
        ),
    ):
        raise ValueError("analysis selection events are not in recording-time order")
    observed_analyzable = sum(
        item["analysis_disposition"] == "analyzable" for item in events
    )
    if (
        analyzable_count != observed_analyzable
        or rejected_count != selected_count - observed_analyzable
        or analyzable_count + rejected_count != selected_count
    ):
        raise ValueError("analysis selection counts do not form an exact partition")
    scope = _strict_object(
        data["scope_receipt"],
        required=(
            "physical_edf_signal_or_metadata_used",
            "edf_annotations_used",
            "excel_used",
            "clinical_context_used",
            "labels_or_ground_truth_used",
            "detector_decisions_modified",
            "rejected_candidates_silently_dropped",
        ),
        context="analysis selection scope receipt",
    )
    if scope != {
        "physical_edf_signal_or_metadata_used": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "detector_decisions_modified": False,
        "rejected_candidates_silently_dropped": False,
    }:
        raise ValueError("analysis selection violates its scope")
    result = {
        "schema_version": ANALYSIS_SELECTION_SCHEMA_VERSION,
        "selection_id": selection_id,
        "recording_id": recording_id,
        "patient_pseudonym": patient,
        "source_signal_sha256": source_hash,
        "recording_duration_seconds": duration,
        "detection_manifest_sha256": detection_hash,
        "event_id_assignment_sha256": assignment_hash,
        "candidate_semantics": CANDIDATE_SEMANTICS,
        "detector_selected_count": selected_count,
        "analyzable_count": analyzable_count,
        "rejected_count": rejected_count,
        "events": events,
        "scope_receipt": scope,
    }
    digest_source = deepcopy(result)
    digest_source["selection_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = ANALYSIS_SELECTION_ID_PREFIX + canonical_payload_sha256(
        digest_source
    )[:20]
    if selection_id != expected_id:
        raise ValueError("analysis selection ID does not bind its content")
    return result


def bind_long_term_eeg_analysis_selection(
    selection: object,
    detection_manifest: object,
    *,
    event_id_assignment: object | None = None,
) -> dict[str, Any]:
    """Validate exact candidate partition and optional event-ID assignment binding."""

    result = validate_long_term_eeg_analysis_selection(selection)
    detection = validate_long_term_seizure_detection_manifest(detection_manifest)
    for key in (
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
    ):
        if result[key] != detection[key]:
            raise ValueError(f"analysis selection {key} differs from detection")
    if not _same_number(
        result["recording_duration_seconds"],
        detection["recording_duration_seconds"],
    ):
        raise ValueError("analysis selection duration differs from detection")
    if result["detection_manifest_sha256"] != canonical_payload_sha256(detection):
        raise ValueError("analysis selection hash does not bind detection manifest")
    selected = {
        item["candidate_id"]: item
        for item in detection["merge_candidates"]
        if item["decision_available"] is True
        and item["decision"] == "selected_for_event_analysis"
    }
    events_by_candidate = {item["candidate_id"]: item for item in result["events"]}
    if set(events_by_candidate) != set(selected):
        raise ValueError("analysis selection does not exactly cover detector selection")
    for candidate_id, event in events_by_candidate.items():
        if not _same_number(
            event["candidate_anchor_offset_seconds"],
            selected[candidate_id]["anchor_offset_seconds"],
        ):
            raise ValueError("analysis selection anchor differs from detector candidate")

    if event_id_assignment is not None:
        if type(event_id_assignment) is not dict:
            raise TypeError("event ID assignment must be a canonical object")
        assignment = deepcopy(event_id_assignment)
        if result["event_id_assignment_sha256"] != canonical_payload_sha256(assignment):
            raise ValueError("analysis selection hash does not bind event ID assignment")
        raw_assignments = assignment.get("assignments")
        if not isinstance(raw_assignments, list):
            raise TypeError("event ID assignment has no assignments array")
        assignment_by_candidate: dict[str, str] = {}
        for raw in raw_assignments:
            if not isinstance(raw, Mapping):
                raise TypeError("event ID assignment entry must be an object")
            candidate_id = _identifier(
                raw.get("candidate_id"), "event ID assignment candidate"
            )
            event_id = _identifier(raw.get("eeg_event_id"), "event ID assignment event")
            if candidate_id in assignment_by_candidate:
                raise ValueError("event ID assignment repeats a candidate")
            assignment_by_candidate[candidate_id] = event_id
        if set(assignment_by_candidate) != set(events_by_candidate):
            raise ValueError("event ID assignment does not cover analysis selection")
        if any(
            events_by_candidate[candidate_id]["eeg_event_id"] != event_id
            for candidate_id, event_id in assignment_by_candidate.items()
        ):
            raise ValueError("analysis selection EEG event identity drifted")
    return result


__all__ = [
    "ANALYSIS_DISPOSITIONS",
    "ANALYSIS_REJECTION_SCHEMA_VERSION",
    "ANALYSIS_SELECTION_SCHEMA_VERSION",
    "ELIGIBILITY_REASON_CODES",
    "bind_long_term_eeg_analysis_selection",
    "validate_analysis_rejection_receipt",
    "validate_long_term_eeg_analysis_selection",
]
