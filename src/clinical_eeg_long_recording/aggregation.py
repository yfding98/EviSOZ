"""Recording-level aggregation for fixed-window clinical EEG event analyses."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report import validate_report_payload

from .schema import (
    CANDIDATE_SEMANTICS,
    FIXED_EVENT_ANCHOR_OFFSET_SECONDS,
    FIXED_EVENT_WINDOW_SECONDS,
    FIXED_SEGMENT_DURATION_SECONDS,
    FILTERED_LONG_TERM_BUNDLE_SCHEMA_VERSION,
    LONG_TERM_BUNDLE_SCHEMA_VERSION,
    _enum,
    _finite_number,
    _identifier,
    _integer,
    _same_number,
    _sha256,
    _strict_object,
    _validate_segment_components,
    canonical_payload_sha256,
    validate_long_term_event_segment_receipt,
    validate_long_term_seizure_detection_manifest,
)
from .analysis_selection import bind_long_term_eeg_analysis_selection


AGGREGATION_POLICY = "full_record_scan_then_fixed_window_event_aggregation_v1"
EVENT_ORDERING_POLICY = "recording_event_start_then_anchor_then_eeg_event_id"
EVENT_REPORT_OCCURRENCE_TIMEBASE = "segment_start_seconds"
WAVEFORM_EVIDENCE_TIMEBASE = "candidate_anchor_seconds"
OUTPUT_EVENT_TIMEBASE = "recording_start_seconds"

_TOLERANCE_SECONDS = 1e-6


def _occurrence_fact(
    report_payload: Mapping[str, Any], eeg_event_id: str
) -> dict[str, Any]:
    matches = [
        fact
        for fact in report_payload["facts"]
        if fact.get("eeg_event_id") == eeg_event_id
        and fact.get("fact_type") == "electrographic_event_occurrence"
    ]
    if len(matches) != 1:  # segment validation already guards this
        raise ValueError("event report requires exactly one occurrence fact")
    return matches[0]


def _set_event_number(
    report_payload: Mapping[str, Any],
    *,
    eeg_event_id: str,
    event_number: int,
) -> dict[str, Any]:
    report = deepcopy(report_payload)
    occurrence = _occurrence_fact(report, eeg_event_id)
    occurrence["value"]["event_number"] = event_number
    return validate_report_payload(report).to_dict()


def _coordinates_from_report(
    report_payload: Mapping[str, Any],
    *,
    eeg_event_id: str,
) -> tuple[float, float]:
    value = _occurrence_fact(report_payload, eeg_event_id)["value"]
    start = float(value["start_offset_seconds"])
    return start, start + float(value["duration_seconds"])


def _aggregation_receipt() -> dict[str, Any]:
    return {
        "aggregation_policy": AGGREGATION_POLICY,
        "event_ordering_policy": EVENT_ORDERING_POLICY,
        "fixed_window_seconds": list(FIXED_EVENT_WINDOW_SECONDS),
        "event_report_occurrence_timebase": EVENT_REPORT_OCCURRENCE_TIMEBASE,
        "waveform_evidence_timebase": WAVEFORM_EVIDENCE_TIMEBASE,
        "output_event_timebase": OUTPUT_EVENT_TIMEBASE,
        "clinical_context_included": False,
        "research_soz_used_in_clinical_facts": False,
    }


def _canonical_bundle_event(
    value: object,
    *,
    index: int,
    manifest: Mapping[str, Any],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    context = f"bundle.events[{index}]"
    data = _strict_object(
        value,
        required=(
            "event_number",
            "eeg_event_id",
            "candidate_id",
            "candidate_anchor_offset_seconds",
            "segment_start_offset_seconds",
            "segment_stop_offset_seconds",
            "recording_event_start_offset_seconds",
            "recording_event_stop_offset_seconds",
            "event_interval_relative_to_anchor_seconds",
            "event_report_payload",
            "waveform_attachment",
            "research_soz_ranking_receipt",
        ),
        context=context,
    )
    event_number = _integer(data["event_number"], f"{context}.event_number", minimum=1)
    event_id = _identifier(data["eeg_event_id"], f"{context}.eeg_event_id")
    candidate_id = _identifier(data["candidate_id"], f"{context}.candidate_id")
    candidate = candidates_by_id.get(candidate_id)
    if candidate is None:
        raise ValueError(f"{context} references an unknown selected candidate")
    anchor = _finite_number(
        data["candidate_anchor_offset_seconds"],
        f"{context}.candidate_anchor_offset_seconds",
        minimum=0,
        maximum=float(manifest["recording_duration_seconds"]),
    )
    if not _same_number(anchor, float(candidate["anchor_offset_seconds"])):
        raise ValueError(f"{context} candidate anchor does not match detection manifest")
    segment_start = _finite_number(
        data["segment_start_offset_seconds"],
        f"{context}.segment_start_offset_seconds",
        minimum=0,
        maximum=float(manifest["recording_duration_seconds"]),
    )
    segment_stop = _finite_number(
        data["segment_stop_offset_seconds"],
        f"{context}.segment_stop_offset_seconds",
        minimum=0,
        maximum=float(manifest["recording_duration_seconds"]),
    )
    if not _same_number(segment_start, anchor + FIXED_EVENT_WINDOW_SECONDS[0]):
        raise ValueError(f"{context} segment start is not anchor minus 12 seconds")
    if not _same_number(segment_stop, anchor + FIXED_EVENT_WINDOW_SECONDS[1]):
        raise ValueError(f"{context} segment stop is not anchor plus 48 seconds")
    if not _same_number(segment_stop - segment_start, FIXED_SEGMENT_DURATION_SECONDS):
        raise ValueError(f"{context} must describe a 60-second segment")

    raw_waveform = data["waveform_attachment"]
    if not isinstance(raw_waveform, Mapping):
        raise TypeError(f"{context}.waveform_attachment must be an object")
    source_hash = _sha256(
        raw_waveform.get("source_signal_sha256"),
        f"{context}.waveform_attachment.source_signal_sha256",
    )
    preprocess_hash = _sha256(
        raw_waveform.get("preprocessing_receipt_sha256"),
        f"{context}.waveform_attachment.preprocessing_receipt_sha256",
    )
    processed_hash = _sha256(
        raw_waveform.get("processed_window_sha256"),
        f"{context}.waveform_attachment.processed_window_sha256",
    )
    if source_hash != manifest["source_signal_sha256"]:
        raise ValueError(f"{context} source signal hash does not match the recording")
    report, waveform, ranking, local_start, local_stop = _validate_segment_components(
        event_report_payload=data["event_report_payload"],
        waveform_attachment=data["waveform_attachment"],
        research_soz_ranking_receipt=data["research_soz_ranking_receipt"],
        patient_pseudonym=str(manifest["patient_pseudonym"]),
        eeg_event_id=event_id,
        source_signal_sha256=source_hash,
        preprocessing_receipt_sha256=preprocess_hash,
        processed_window_sha256=processed_hash,
    )
    occurrence_number = _occurrence_fact(report, event_id)["value"]["event_number"]
    if occurrence_number != event_number:
        raise ValueError(f"{context} event number does not match its occurrence fact")

    relative_interval = data["event_interval_relative_to_anchor_seconds"]
    if not isinstance(relative_interval, list) or len(relative_interval) != 2:
        raise TypeError(f"{context}.event_interval_relative_to_anchor_seconds must be a pair")
    relative_start = _finite_number(
        relative_interval[0],
        f"{context}.event_interval_relative_to_anchor_seconds[0]",
    )
    relative_stop = _finite_number(
        relative_interval[1],
        f"{context}.event_interval_relative_to_anchor_seconds[1]",
    )
    evidence_interval = waveform["evidence_interval_seconds_relative_to_anchor"]
    if not _same_number(relative_start, evidence_interval[0]) or not _same_number(
        relative_stop, evidence_interval[1]
    ):
        raise ValueError(f"{context} anchor-relative interval does not match waveform evidence")
    if not _same_number(local_start, relative_start + FIXED_EVENT_ANCHOR_OFFSET_SECONDS):
        raise ValueError(f"{context} segment and anchor timebases do not close")
    if not _same_number(local_stop, relative_stop + FIXED_EVENT_ANCHOR_OFFSET_SECONDS):
        raise ValueError(f"{context} segment and anchor stop timebases do not close")

    recording_start = _finite_number(
        data["recording_event_start_offset_seconds"],
        f"{context}.recording_event_start_offset_seconds",
        minimum=0,
        maximum=float(manifest["recording_duration_seconds"]),
    )
    recording_stop = _finite_number(
        data["recording_event_stop_offset_seconds"],
        f"{context}.recording_event_stop_offset_seconds",
        minimum=0,
        maximum=float(manifest["recording_duration_seconds"]),
    )
    if recording_stop <= recording_start:
        raise ValueError(f"{context} recording event interval must have positive duration")
    if not _same_number(recording_start, segment_start + local_start):
        raise ValueError(f"{context} recording start does not equal segment start plus local start")
    if not _same_number(recording_stop, segment_start + local_stop):
        raise ValueError(f"{context} recording stop does not equal segment start plus local stop")
    if not _same_number(recording_start, anchor + relative_start):
        raise ValueError(f"{context} recording start does not equal anchor plus relative start")
    if not _same_number(recording_stop, anchor + relative_stop):
        raise ValueError(f"{context} recording stop does not equal anchor plus relative stop")

    return {
        "event_number": event_number,
        "eeg_event_id": event_id,
        "candidate_id": candidate_id,
        "candidate_anchor_offset_seconds": anchor,
        "segment_start_offset_seconds": segment_start,
        "segment_stop_offset_seconds": segment_stop,
        "recording_event_start_offset_seconds": recording_start,
        "recording_event_stop_offset_seconds": recording_stop,
        "event_interval_relative_to_anchor_seconds": [relative_start, relative_stop],
        "event_report_payload": report,
        "waveform_attachment": waveform,
        "research_soz_ranking_receipt": ranking,
    }


def validate_trustworthy_long_term_clinical_eeg_bundle(payload: object) -> dict[str, Any]:
    """Validate an aggregated recording bundle and return a canonical copy."""

    if type(payload) is not dict:
        raise TypeError("trustworthy long-term clinical EEG bundle must be an object")
    schema_raw = payload.get("schema_version")
    filtered = schema_raw == FILTERED_LONG_TERM_BUNDLE_SCHEMA_VERSION
    if schema_raw not in (
        LONG_TERM_BUNDLE_SCHEMA_VERSION,
        FILTERED_LONG_TERM_BUNDLE_SCHEMA_VERSION,
    ):
        raise ValueError("bundle.schema_version is unsupported")
    required = [
        "schema_version",
        "bundle_id",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "detection_manifest_sha256",
        "detection_manifest",
        "event_count",
        "events",
        "aggregation_receipt",
    ]
    if filtered:
        required.extend(
            (
                "analysis_selection_sha256",
                "analysis_selection",
                "detector_selected_candidate_count",
                "analysis_analyzable_candidate_count",
                "analysis_rejected_candidate_count",
            )
        )
    data = _strict_object(
        payload,
        required=tuple(required),
        context="trustworthy long-term clinical EEG bundle",
    )
    schema_version = _enum(
        data["schema_version"],
        (
            LONG_TERM_BUNDLE_SCHEMA_VERSION,
            FILTERED_LONG_TERM_BUNDLE_SCHEMA_VERSION,
        ),
        "bundle.schema_version",
    )
    manifest = validate_long_term_seizure_detection_manifest(data["detection_manifest"])
    recording_id = _identifier(data["recording_id"], "bundle.recording_id")
    patient = _identifier(data["patient_pseudonym"], "bundle.patient_pseudonym")
    source_hash = _sha256(data["source_signal_sha256"], "bundle.source_signal_sha256")
    duration = _finite_number(
        data["recording_duration_seconds"],
        "bundle.recording_duration_seconds",
        exclusive_minimum=0,
    )
    if recording_id != manifest["recording_id"]:
        raise ValueError("bundle recording_id does not match detection manifest")
    if patient != manifest["patient_pseudonym"]:
        raise ValueError("bundle patient_pseudonym does not match detection manifest")
    if source_hash != manifest["source_signal_sha256"]:
        raise ValueError("bundle source signal hash does not match detection manifest")
    if not _same_number(duration, manifest["recording_duration_seconds"]):
        raise ValueError("bundle duration does not match detection manifest")
    manifest_hash = _sha256(
        data["detection_manifest_sha256"], "bundle.detection_manifest_sha256"
    )
    if manifest_hash != canonical_payload_sha256(manifest):
        raise ValueError("bundle detection manifest hash does not match its payload")

    detector_selected_candidates = {
        item["candidate_id"]: item
        for item in manifest["merge_candidates"]
        if item["decision_available"] is True
        and item["decision"] == "selected_for_event_analysis"
    }
    selection = None
    selection_hash = None
    detector_selected_count = len(detector_selected_candidates)
    rejected_count = 0
    if filtered:
        selection = bind_long_term_eeg_analysis_selection(
            data["analysis_selection"], manifest
        )
        selection_hash = _sha256(
            data["analysis_selection_sha256"],
            "bundle.analysis_selection_sha256",
        )
        if selection_hash != canonical_payload_sha256(selection):
            raise ValueError("bundle analysis selection hash does not match payload")
        detector_selected_count = _integer(
            data["detector_selected_candidate_count"],
            "bundle detector-selected candidate count",
            minimum=0,
        )
        analyzable_count = _integer(
            data["analysis_analyzable_candidate_count"],
            "bundle analyzable candidate count",
            minimum=0,
        )
        rejected_count = _integer(
            data["analysis_rejected_candidate_count"],
            "bundle rejected candidate count",
            minimum=0,
        )
        if (
            detector_selected_count != selection["detector_selected_count"]
            or analyzable_count != selection["analyzable_count"]
            or rejected_count != selection["rejected_count"]
        ):
            raise ValueError("bundle analysis-selection counts drifted")
        selected_candidates = {
            item["candidate_id"]: detector_selected_candidates[item["candidate_id"]]
            for item in selection["events"]
            if item["analysis_disposition"] == "analyzable"
        }
    else:
        selected_candidates = detector_selected_candidates
        analyzable_count = len(selected_candidates)
    raw_events = data["events"]
    if not isinstance(raw_events, list):
        raise TypeError("bundle.events must be an array")
    events = [
        _canonical_bundle_event(
            raw,
            index=index,
            manifest=manifest,
            candidates_by_id=selected_candidates,
        )
        for index, raw in enumerate(raw_events)
    ]
    event_count = _integer(data["event_count"], "bundle.event_count")
    if event_count != len(events):
        raise ValueError("bundle.event_count must equal the number of events")
    if {event["candidate_id"] for event in events} != set(selected_candidates):
        raise ValueError("bundle events must exactly cover selected detector candidates")
    for key, label in (
        ("eeg_event_id", "EEG event IDs"),
        ("candidate_id", "candidate IDs"),
    ):
        values = [event[key] for event in events]
        if len(values) != len(set(values)):
            raise ValueError(f"bundle contains duplicate {label}")
    attachment_ids = [event["waveform_attachment"]["attachment_id"] for event in events]
    figure_files = [event["waveform_attachment"]["figure_file"] for event in events]
    processed_hashes = [
        event["waveform_attachment"]["processed_window_sha256"] for event in events
    ]
    if len(attachment_ids) != len(set(attachment_ids)):
        raise ValueError("bundle contains duplicate waveform attachment IDs")
    if len(figure_files) != len(set(figure_files)):
        raise ValueError("bundle waveform figure_file values must be unique")
    if len(processed_hashes) != len(set(processed_hashes)):
        raise ValueError("bundle contains duplicate processed event windows")
    expected_order = sorted(
        events,
        key=lambda event: (
            event["recording_event_start_offset_seconds"],
            event["candidate_anchor_offset_seconds"],
            event["eeg_event_id"],
        ),
    )
    if events != expected_order:
        raise ValueError("bundle events are not in original-recording time order")
    if [event["event_number"] for event in events] != list(range(1, len(events) + 1)):
        raise ValueError("bundle event numbers must be contiguous in recording order")

    receipt_raw = _strict_object(
        data["aggregation_receipt"],
        required=(
            "aggregation_policy",
            "event_ordering_policy",
            "fixed_window_seconds",
            "event_report_occurrence_timebase",
            "waveform_evidence_timebase",
            "output_event_timebase",
            "clinical_context_included",
            "research_soz_used_in_clinical_facts",
        ),
        context="bundle.aggregation_receipt",
    )
    expected_receipt = _aggregation_receipt()
    if receipt_raw != expected_receipt:
        raise ValueError("bundle.aggregation_receipt does not match the frozen policy")

    result = {
        "schema_version": schema_version,
        "bundle_id": _identifier(data["bundle_id"], "bundle.bundle_id"),
        "recording_id": recording_id,
        "patient_pseudonym": patient,
        "source_signal_sha256": source_hash,
        "recording_duration_seconds": duration,
        "detection_manifest_sha256": manifest_hash,
        "detection_manifest": manifest,
        "event_count": event_count,
        "events": events,
        "aggregation_receipt": expected_receipt,
    }
    if filtered:
        result.update(
            {
                "analysis_selection_sha256": selection_hash,
                "analysis_selection": selection,
                "detector_selected_candidate_count": detector_selected_count,
                "analysis_analyzable_candidate_count": analyzable_count,
                "analysis_rejected_candidate_count": rejected_count,
            }
        )
    return result


def aggregate_long_term_event_segments(
    detection_manifest: object,
    segment_receipts: Sequence[object],
    bundle_id: str,
    *,
    analysis_selection: object | None = None,
) -> dict[str, Any]:
    """Aggregate validated event segments into one recording-level bundle.

    ``clinical_eeg_report_v1`` occurrence offsets remain segment-local.  This
    function derives recording coordinates as ``segment_start + local`` and
    independently checks them against ``candidate_anchor + waveform-relative``.
    Event numbers are then rewritten in original-recording time order.
    """

    manifest = validate_long_term_seizure_detection_manifest(detection_manifest)
    if isinstance(segment_receipts, (str, bytes)) or not isinstance(
        segment_receipts, Sequence
    ):
        raise TypeError("segment_receipts must be an array")
    segments = [validate_long_term_event_segment_receipt(item) for item in segment_receipts]
    detector_selected_candidates = {
        item["candidate_id"]: item
        for item in manifest["merge_candidates"]
        if item["decision_available"] is True
        and item["decision"] == "selected_for_event_analysis"
    }
    selection = (
        bind_long_term_eeg_analysis_selection(analysis_selection, manifest)
        if analysis_selection is not None
        else None
    )
    if selection is None:
        selected_candidates = detector_selected_candidates
    else:
        selected_candidates = {
            item["candidate_id"]: detector_selected_candidates[item["candidate_id"]]
            for item in selection["events"]
            if item["analysis_disposition"] == "analyzable"
        }
    segment_candidate_ids = [segment["candidate_id"] for segment in segments]
    if len(segment_candidate_ids) != len(set(segment_candidate_ids)):
        raise ValueError("segment receipts contain duplicate candidate_id values")
    if set(segment_candidate_ids) != set(selected_candidates):
        raise ValueError("segments must exactly cover selected detector candidates")
    event_ids = [segment["eeg_event_id"] for segment in segments]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("segment receipts contain duplicate eeg_event_id values")
    receipt_ids = [segment["segment_receipt_id"] for segment in segments]
    if len(receipt_ids) != len(set(receipt_ids)):
        raise ValueError("segment receipts contain duplicate segment_receipt_id values")

    provisional: list[tuple[tuple[float, float, str], dict[str, Any]]] = []
    for segment in segments:
        for identity_key in (
            "recording_id",
            "patient_pseudonym",
            "source_signal_sha256",
        ):
            if segment[identity_key] != manifest[identity_key]:
                raise ValueError(f"segment {identity_key} does not match detection manifest")
        if not _same_number(
            segment["recording_duration_seconds"], manifest["recording_duration_seconds"]
        ):
            raise ValueError("segment duration binding does not match detection manifest")
        candidate = selected_candidates[segment["candidate_id"]]
        if not _same_number(
            segment["candidate_anchor_offset_seconds"], candidate["anchor_offset_seconds"]
        ):
            raise ValueError("segment candidate anchor does not match detection manifest")
        local_start, local_stop = _coordinates_from_report(
            segment["event_report_payload"], eeg_event_id=segment["eeg_event_id"]
        )
        recording_start = segment["segment_start_offset_seconds"] + local_start
        recording_stop = segment["segment_start_offset_seconds"] + local_stop
        relative_interval = segment["waveform_attachment"][
            "evidence_interval_seconds_relative_to_anchor"
        ]
        if not _same_number(
            recording_start,
            segment["candidate_anchor_offset_seconds"] + relative_interval[0],
        ) or not _same_number(
            recording_stop,
            segment["candidate_anchor_offset_seconds"] + relative_interval[1],
        ):
            raise ValueError("segment, anchor, and recording timebases do not close")
        duration = float(manifest["recording_duration_seconds"])
        if (
            recording_start < -_TOLERANCE_SECONDS
            or recording_stop > duration + _TOLERANCE_SECONDS
        ):
            raise ValueError("derived recording event interval is out of bounds")
        event = {
            "event_number": 0,
            "eeg_event_id": segment["eeg_event_id"],
            "candidate_id": segment["candidate_id"],
            "candidate_anchor_offset_seconds": segment[
                "candidate_anchor_offset_seconds"
            ],
            "segment_start_offset_seconds": segment["segment_start_offset_seconds"],
            "segment_stop_offset_seconds": segment["segment_stop_offset_seconds"],
            "recording_event_start_offset_seconds": recording_start,
            "recording_event_stop_offset_seconds": recording_stop,
            "event_interval_relative_to_anchor_seconds": list(relative_interval),
            "event_report_payload": segment["event_report_payload"],
            "waveform_attachment": segment["waveform_attachment"],
            "research_soz_ranking_receipt": segment[
                "research_soz_ranking_receipt"
            ],
        }
        order_key = (
            recording_start,
            float(segment["candidate_anchor_offset_seconds"]),
            segment["eeg_event_id"],
        )
        provisional.append((order_key, event))

    provisional.sort(key=lambda item: item[0])
    events: list[dict[str, Any]] = []
    for event_number, (_, event) in enumerate(provisional, start=1):
        event["event_number"] = event_number
        event["event_report_payload"] = _set_event_number(
            event["event_report_payload"],
            eeg_event_id=event["eeg_event_id"],
            event_number=event_number,
        )
        events.append(event)

    bundle = {
        "schema_version": (
            FILTERED_LONG_TERM_BUNDLE_SCHEMA_VERSION
            if selection is not None
            else LONG_TERM_BUNDLE_SCHEMA_VERSION
        ),
        "bundle_id": _identifier(bundle_id, "bundle_id"),
        "recording_id": manifest["recording_id"],
        "patient_pseudonym": manifest["patient_pseudonym"],
        "source_signal_sha256": manifest["source_signal_sha256"],
        "recording_duration_seconds": manifest["recording_duration_seconds"],
        "detection_manifest_sha256": canonical_payload_sha256(manifest),
        "detection_manifest": manifest,
        "event_count": len(events),
        "events": events,
        "aggregation_receipt": _aggregation_receipt(),
    }
    if selection is not None:
        bundle.update(
            {
                "analysis_selection_sha256": canonical_payload_sha256(selection),
                "analysis_selection": selection,
                "detector_selected_candidate_count": selection[
                    "detector_selected_count"
                ],
                "analysis_analyzable_candidate_count": selection[
                    "analyzable_count"
                ],
                "analysis_rejected_candidate_count": selection[
                    "rejected_count"
                ],
            }
        )
    return validate_trustworthy_long_term_clinical_eeg_bundle(bundle)


validate_trustworthy_long_term_clinical_eeg_bundle_payload = (
    validate_trustworthy_long_term_clinical_eeg_bundle
)


__all__ = [
    "AGGREGATION_POLICY",
    "EVENT_ORDERING_POLICY",
    "EVENT_REPORT_OCCURRENCE_TIMEBASE",
    "OUTPUT_EVENT_TIMEBASE",
    "WAVEFORM_EVIDENCE_TIMEBASE",
    "aggregate_long_term_event_segments",
    "validate_trustworthy_long_term_clinical_eeg_bundle",
    "validate_trustworthy_long_term_clinical_eeg_bundle_payload",
]
