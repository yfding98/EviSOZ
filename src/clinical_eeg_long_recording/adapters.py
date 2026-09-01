"""Adapters from the existing event-level trustworthy artifacts.

The existing private pipeline already materializes one facts-locked
``clinical_eeg_report_v1`` payload, one waveform receipt and one research
electrode ranking per frozen 60-second event.  This module binds those three
artifacts to an original-recording candidate anchor without copying legacy
clinical prose, raw paths, or source annotation facts into the recording
bundle.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.clinical_eeg_report.schema import validate_report_payload

from .schema import (
    BOUNDARY_POLICY,
    EVENT_SEGMENT_RECEIPT_SCHEMA_VERSION,
    FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
    FIXED_EVENT_WINDOW_SECONDS,
    SOZ_INTERPRETATION_STATUS,
    WAVEFORM_SELECTION_POLICY,
    canonical_payload_sha256,
    validate_long_term_event_segment_receipt,
)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return value


def _single_occurrence(report: Mapping[str, Any], event_id: str) -> dict[str, Any]:
    matches = [
        fact
        for fact in report["facts"]
        if fact.get("fact_type") == "electrographic_event_occurrence"
        and fact.get("eeg_event_id") == event_id
    ]
    if len(matches) != 1:
        raise ValueError("legacy event report requires exactly one occurrence fact")
    return matches[0]


def _clean_event_report(
    payload: object,
    *,
    event_id: str,
    patient_pseudonym: str,
) -> dict[str, Any]:
    """Remove source-context facts and state the 60-second timebase.

    ``source_eeg_annotation_timing`` belongs to
    ``clinical_eeg_long_term_context_v1`` in recording mode.  It is removed
    before the content hash is recomputed, so context can never affect the EEG
    fact ledger or a downstream language prompt.
    """

    report = validate_report_payload(payload).to_dict()
    if report["patient_pseudonym"] != patient_pseudonym:
        raise ValueError("legacy report patient binding mismatch")
    if report["eeg_event_ids"] != [event_id]:
        raise ValueError("legacy report must contain exactly the requested event")
    report["facts"] = [
        fact
        for fact in report["facts"]
        if fact["fact_type"] != "source_eeg_annotation_timing"
    ]
    occurrence = _single_occurrence(report, event_id)
    occurrence["value"]["time_coordinate"] = "segment_start_seconds"
    digest_source = deepcopy(report)
    digest_source["report_id"] = "CONTENT-ADDRESS-PENDING"
    report["report_id"] = f"CER-LONG-{canonical_payload_sha256(digest_source)[:24]}"
    return validate_report_payload(report).to_dict()


def adapt_legacy_event_to_long_term_segment(
    *,
    event_report_payload: object,
    legacy_waveform_attachment: object,
    qualified_report: object,
    recording_id: str,
    patient_pseudonym: str,
    source_signal_sha256: str,
    recording_duration_seconds: float,
    candidate_id: str,
    candidate_anchor_offset_seconds: float,
    eeg_event_id: str,
    portable_figure_file: str,
    ranker_method_id: str,
    ranker_model_sha256: str,
) -> dict[str, Any]:
    """Create one strict long-recording event segment receipt.

    The adapter is intentionally a migration bridge.  It does not claim that
    a historical frozen event roster was produced by the new detector.  The
    caller must separately provide a detection manifest whose selected
    candidate ID and anchor exactly match this receipt.
    """

    report = _clean_event_report(
        event_report_payload,
        event_id=eeg_event_id,
        patient_pseudonym=patient_pseudonym,
    )
    occurrence = _single_occurrence(report, eeg_event_id)["value"]
    local_start = float(occurrence["start_offset_seconds"])
    local_stop = local_start + float(occurrence["duration_seconds"])
    relative_interval = [
        local_start - FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
        local_stop - FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
    ]

    legacy_waveform = _mapping(legacy_waveform_attachment, "legacy waveform")
    required_waveform = {
        "evidence_id",
        "fact_ids",
        "eeg_event_id",
        "figure_sha256",
        "source_signal_sha256",
        "preprocessing_receipt_sha256",
        "processed_window_sha256",
        "event_window_seconds",
        "event_anchor_offset_seconds",
    }
    missing_waveform = required_waveform.difference(legacy_waveform)
    if missing_waveform:
        raise ValueError(
            f"legacy waveform is missing receipt fields: {sorted(missing_waveform)}"
        )
    if legacy_waveform["eeg_event_id"] != eeg_event_id:
        raise ValueError("legacy waveform event binding mismatch")
    if legacy_waveform["source_signal_sha256"] != source_signal_sha256:
        raise ValueError("legacy waveform signal binding mismatch")
    if list(legacy_waveform["event_window_seconds"]) != list(
        FIXED_EVENT_WINDOW_SECONDS
    ):
        raise ValueError("legacy waveform does not use the fixed event window")
    if float(legacy_waveform["event_anchor_offset_seconds"]) != float(
        FIXED_EVENT_ANCHOR_OFFSET_SECONDS
    ):
        raise ValueError("legacy waveform anchor is not 12 seconds")
    waveform_fact_ids = [
        fact_id
        for fact_id in legacy_waveform["fact_ids"]
        if any(fact["fact_id"] == fact_id for fact in report["facts"])
    ]
    if not waveform_fact_ids:
        raise ValueError("legacy waveform has no retained EEG fact binding")
    waveform = {
        "attachment_id": f"WAVE-LONG-{canonical_payload_sha256(dict(legacy_waveform))[:20]}",
        "evidence_id": legacy_waveform["evidence_id"],
        "fact_ids": waveform_fact_ids,
        "eeg_event_id": eeg_event_id,
        "figure_file": portable_figure_file,
        "figure_sha256": legacy_waveform["figure_sha256"],
        "source_signal_sha256": source_signal_sha256,
        "preprocessing_receipt_sha256": legacy_waveform[
            "preprocessing_receipt_sha256"
        ],
        "processed_window_sha256": legacy_waveform["processed_window_sha256"],
        "event_window_seconds": list(FIXED_EVENT_WINDOW_SECONDS),
        "event_anchor_offset_seconds": FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
        "evidence_interval_seconds_relative_to_anchor": relative_interval,
        "selection_policy": WAVEFORM_SELECTION_POLICY,
        "sent_to_llm": False,
    }

    qualified = _mapping(qualified_report, "qualified report")
    if qualified.get("unit_id") != eeg_event_id:
        raise ValueError("qualified report event binding mismatch")
    if qualified.get("patient_id") != patient_pseudonym:
        raise ValueError("qualified report patient binding mismatch")
    localization = _mapping(qualified.get("localization"), "qualified localization")
    if localization.get("action") != "display_candidate":
        raise ValueError("qualified report has no displayable research ranking")
    displayed = localization.get("displayed_candidates")
    if not isinstance(displayed, list) or not displayed:
        raise ValueError("qualified report research ranking is empty")
    ranked_electrodes = []
    for rank, raw in enumerate(displayed, start=1):
        item = _mapping(raw, f"qualified localization candidate {rank}")
        if set(item) != {"channel", "normalized_candidate_score"}:
            raise ValueError("qualified localization candidate schema drifted")
        ranked_electrodes.append(
            {
                "rank": rank,
                "electrode": item["channel"],
                "score": float(item["normalized_candidate_score"]),
            }
        )
    ranking = {
        "receipt_id": f"SOZRANK-{canonical_payload_sha256(localization)[:24]}",
        "method_id": ranker_method_id,
        "model_sha256": ranker_model_sha256,
        "input_processed_window_sha256": legacy_waveform[
            "processed_window_sha256"
        ],
        "interpretation_status": SOZ_INTERPRETATION_STATUS,
        "ranked_electrodes": ranked_electrodes,
        "used_in_clinical_facts": False,
        "used_in_impression": False,
        "sent_to_llm": False,
    }

    anchor = float(candidate_anchor_offset_seconds)
    duration = float(recording_duration_seconds)
    identity = {
        "recording_id": recording_id,
        "candidate_id": candidate_id,
        "eeg_event_id": eeg_event_id,
        "anchor": anchor,
        "processed_window_sha256": legacy_waveform["processed_window_sha256"],
    }
    segment = {
        "schema_version": EVENT_SEGMENT_RECEIPT_SCHEMA_VERSION,
        "segment_receipt_id": f"SEG-{canonical_payload_sha256(identity)[:24]}",
        "recording_id": recording_id,
        "patient_pseudonym": patient_pseudonym,
        "source_signal_sha256": source_signal_sha256,
        "recording_duration_seconds": duration,
        "candidate_id": candidate_id,
        "eeg_event_id": eeg_event_id,
        "candidate_anchor_offset_seconds": anchor,
        "requested_window_seconds": list(FIXED_EVENT_WINDOW_SECONDS),
        "segment_start_offset_seconds": anchor + FIXED_EVENT_WINDOW_SECONDS[0],
        "segment_stop_offset_seconds": anchor + FIXED_EVENT_WINDOW_SECONDS[1],
        "warmup_seconds_available": anchor,
        "post_anchor_seconds_available": duration - anchor,
        "boundary_policy": BOUNDARY_POLICY,
        "processed_window_sha256": legacy_waveform["processed_window_sha256"],
        "preprocessing_receipt_sha256": legacy_waveform[
            "preprocessing_receipt_sha256"
        ],
        "event_report_payload": report,
        "waveform_attachment": waveform,
        "research_soz_ranking_receipt": ranking,
    }
    return validate_long_term_event_segment_receipt(segment)


__all__ = ["adapt_legacy_event_to_long_term_segment"]
