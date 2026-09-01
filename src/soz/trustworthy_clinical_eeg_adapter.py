"""Fail-closed adapter from trustworthy typed EEG facts to the clinical ledger.

The adapter deliberately consumes only target-blind waveform observations.
Candidate localization, diagnoses, clinical context, sleep/activation data,
and unqualified morphology/rhythm/artifact descriptors are neither inspected
nor copied. One invocation adapts one 60-second EEG segment and returns a
validated clinical_eeg_report_v1 payload plus its report-specific waveform
evidence manifest.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report.evidence import WAVEFORM_SELECTION_POLICY
from src.clinical_eeg_report.schema import (
    canonicalize_derivation,
    canonicalize_electrode,
    validate_report_payload,
)


QUALIFIED_REPORT_SCHEMA = "trustworthy_soz_qualified_report_v24"
PUBLIC_TYPED_SOURCE_SCHEMA = "soz_target_free_oof_report_assembler_v3"
PUBLIC_TYPED_RECORD_STATUS = "assembled_abstained_target_free_oof_draft"
WAVEFORM_MANIFEST_SCHEMA = "clinical_eeg_waveform_manifest_v1"
PRIVATE_ANNOTATION_EVENT_SCHEMA = "private_clinical_eeg_annotation_event_v1"

STANDARD_19 = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
)
EVENT_WINDOW_RELATIVE_SECONDS = (-12.0, 48.0)
EVENT_ANCHOR_OFFSET_SECONDS = 12.0
SEGMENT_DURATION_SECONDS = 60.0
SAMPLING_RATE_HZ = 200.0
FILTER_HZ = (0.5, 45.0)
REFERENCE = "common_average_standard19"
PUBLIC_SELECTION_POLICY = "lexicographically_first_reportable_target_blind_event"
PRIVATE_SELECTION_POLICY = "exact_private_event"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PRIVATE_EVENT_RE = re.compile(r"^PRIV-E\d{4}$")
_PRIVATE_PATIENT_RE = re.compile(r"^PRIV-P\d{3}$")
_PUBLIC_PATIENT_RE = re.compile(r"^\d+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = set(keys).difference(value)
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")


def _exact_keys(value: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    required = set(keys)
    missing = required.difference(value)
    extra = set(value).difference(required)
    if missing or extra:
        raise ValueError(
            f"{name} shape drifted: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is not a safe pseudonymous identifier")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{name} must be a two-number list")
    start = _number(value[0], f"{name}[0]")
    stop = _number(value[1], f"{name}[1]")
    if stop <= start:
        raise ValueError(f"{name} must have a strictly increasing interval")
    return start, stop


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)


def _same_pair(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return _same_number(left[0], right[0]) and _same_number(left[1], right[1])


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_figure_file(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("waveform.figure_file must be a non-empty string")
    if "\\" in value:
        raise ValueError("waveform.figure_file must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".png":
        raise ValueError("waveform.figure_file must be a safe relative PNG path")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError("waveform.figure_file is not normalized")
    return path.as_posix()


def _canonical_derivations(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty list")
    result = [canonicalize_derivation(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate derivations")
    return result


def _format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _waveform_contract(entry: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "scope",
        "unit_id",
        "patient_id",
        "event_id",
        "figure_file",
        "representative_event",
        "event_window_sec",
        "sampling_rate_hz",
        "filter_hz",
        "reference",
        "channel_order",
        "event_anchor_offset_seconds",
        "evidence_interval_sec",
        "figure_sha256",
        "source_signal_sha256",
        "preprocessing_receipt_sha256",
        "processed_window_sha256",
    )
    _require_keys(entry, required, "waveform entry")
    scope = entry["scope"]
    if scope not in {"private_event", "public_patient"}:
        raise ValueError("waveform.scope is unsupported")
    unit_id = _identifier(entry["unit_id"], "waveform.unit_id")
    patient_id = _identifier(entry["patient_id"], "waveform.patient_id")
    event_id = _identifier(entry["event_id"], "waveform.event_id")
    representative = entry["representative_event"]
    if not isinstance(representative, bool):
        raise TypeError("waveform.representative_event must be boolean")
    window = _pair(entry["event_window_sec"], "waveform.event_window_sec")
    if not _same_pair(window, EVENT_WINDOW_RELATIVE_SECONDS):
        raise ValueError("waveform must use the frozen [-12,48] second event window")
    sampling_rate = _positive(entry["sampling_rate_hz"], "waveform.sampling_rate_hz")
    if not _same_number(sampling_rate, SAMPLING_RATE_HZ):
        raise ValueError("waveform must use 200 Hz")
    filter_hz = _pair(entry["filter_hz"], "waveform.filter_hz")
    if not _same_pair(filter_hz, FILTER_HZ):
        raise ValueError("waveform must use the frozen 0.5-45 Hz filter")
    if entry["reference"] != REFERENCE:
        raise ValueError("waveform must use standard-19 common-average reference")
    channels = entry["channel_order"]
    if not isinstance(channels, list) or tuple(channels) != STANDARD_19:
        raise ValueError("waveform must use the exact standard-19 channel order")
    anchor = _number(
        entry["event_anchor_offset_seconds"],
        "waveform.event_anchor_offset_seconds",
    )
    if not _same_number(anchor, EVENT_ANCHOR_OFFSET_SECONDS):
        raise ValueError("waveform event anchor must be 12 seconds from segment start")
    interval = _pair(entry["evidence_interval_sec"], "waveform.evidence_interval_sec")
    if interval[0] < window[0] or interval[1] > window[1]:
        raise ValueError("waveform evidence interval falls outside the event window")
    return {
        "scope": scope,
        "unit_id": unit_id,
        "patient_id": patient_id,
        "event_id": event_id,
        "figure_file": _safe_figure_file(entry["figure_file"]),
        "representative_event": representative,
        "event_window_sec": window,
        "sampling_rate_hz": sampling_rate,
        "filter_hz": filter_hz,
        "reference": REFERENCE,
        "channel_order": list(STANDARD_19),
        "event_anchor_offset_seconds": anchor,
        "evidence_interval_sec": interval,
        "figure_sha256": _sha256(entry["figure_sha256"], "waveform.figure_sha256"),
        "source_signal_sha256": _sha256(
            entry["source_signal_sha256"], "waveform.source_signal_sha256"
        ),
        "preprocessing_receipt_sha256": _sha256(
            entry["preprocessing_receipt_sha256"],
            "waveform.preprocessing_receipt_sha256",
        ),
        "processed_window_sha256": _sha256(
            entry["processed_window_sha256"], "waveform.processed_window_sha256"
        ),
        "selection_policy": entry.get("selection_policy"),
    }


def _private_descriptor(
    report: Mapping[str, Any],
    *,
    patient_id: str,
    event_id: str,
) -> tuple[tuple[float, float], list[str], list[dict[str, Any]]]:
    descriptor = _mapping(report.get("private_event_descriptor"), "private descriptor")
    _exact_keys(
        descriptor,
        (
            "schema_version",
            "event_id",
            "patient_id",
            "algorithmic_sustained_change",
            "later_scalp_visible_change_candidates",
            "qualification",
            "lineage",
        ),
        "private descriptor",
    )
    if descriptor["schema_version"] != "soz_private_event_descriptor_target_blind_v24":
        raise ValueError("private descriptor schema drifted")
    if descriptor["event_id"] != event_id or descriptor["patient_id"] != patient_id:
        raise ValueError("private report/descriptor identity mismatch")
    change = _mapping(
        descriptor["algorithmic_sustained_change"],
        "private algorithmic sustained change",
    )
    _exact_keys(
        change,
        (
            "status",
            "support_interval_sec_relative_to_clinical_event_anchor",
            "bipolar_derivation_candidates",
            "physical_electrode_onset_truth",
            "soz_onset_truth",
        ),
        "private algorithmic sustained change",
    )
    if change["status"] != "detected":
        raise ValueError("private descriptor has no detected sustained EEG change")
    if (
        change["physical_electrode_onset_truth"] is not False
        or change["soz_onset_truth"] is not False
    ):
        raise ValueError("private descriptor improperly promotes a candidate to onset truth")
    interval = _pair(
        change["support_interval_sec_relative_to_clinical_event_anchor"],
        "private descriptor support interval",
    )
    derivations = _canonical_derivations(
        change["bipolar_derivation_candidates"],
        "private descriptor derivations",
    )
    raw_later = descriptor["later_scalp_visible_change_candidates"]
    if not isinstance(raw_later, list):
        raise TypeError("private later-visible candidates must be a list")
    observations: list[dict[str, Any]] = []
    seen_electrodes: set[str] = set()
    for index, raw in enumerate(raw_later):
        candidate = _mapping(raw, f"private later-visible candidate {index}")
        _exact_keys(
            candidate,
            ("channel", "delay_sec"),
            f"private later-visible candidate {index}",
        )
        electrode = canonicalize_electrode(candidate["channel"])
        delay = _positive(candidate["delay_sec"], f"private later delay {index}")
        if electrode in seen_electrodes:
            raise ValueError("private later-visible candidates contain duplicates")
        seen_electrodes.add(electrode)
        observations.append({"electrode": electrode, "delay_seconds": delay})
    lineage = _mapping(descriptor["lineage"], "private descriptor lineage")
    _exact_keys(
        lineage,
        (
            "source",
            "private_soz_target_used",
            "deepsoz_target_used",
            "model_prediction_used",
        ),
        "private descriptor lineage",
    )
    if any(
        lineage[key] is not False
        for key in (
            "private_soz_target_used",
            "deepsoz_target_used",
            "model_prediction_used",
        )
    ):
        raise ValueError("private descriptor is not target-blind")
    return interval, derivations, observations


def _select_public_record(
    source: Mapping[str, Any],
    *,
    patient_id: str,
    event_id: str,
) -> Mapping[str, Any]:
    if source.get("schema_version") == PUBLIC_TYPED_SOURCE_SCHEMA:
        records = source.get("records")
        if not isinstance(records, list):
            raise TypeError("public typed source records must be a list")
        matches = [
            item
            for item in records
            if isinstance(item, Mapping)
            and item.get("patient_id") == patient_id
            and item.get("event_id") == event_id
        ]
        if len(matches) != 1:
            raise ValueError("public typed source must contain exactly one matching event")
        return matches[0]
    return source


def _public_descriptor(
    source: Mapping[str, Any],
    *,
    patient_id: str,
    event_id: str,
    waveform: Mapping[str, Any],
) -> tuple[tuple[float, float], list[str], list[dict[str, Any]]]:
    record = _select_public_record(source, patient_id=patient_id, event_id=event_id)
    _require_keys(
        record,
        ("event_id", "patient_id", "status", "typed_facts", "assembly_receipt"),
        "public typed record",
    )
    if record["event_id"] != event_id or record["patient_id"] != patient_id:
        raise ValueError("public typed record identity mismatch")
    if record["status"] != PUBLIC_TYPED_RECORD_STATUS:
        raise ValueError("public typed record is not a qualified assembled draft")
    assembly = _mapping(record["assembly_receipt"], "public assembly receipt")
    _require_keys(
        assembly,
        (
            "event_id",
            "patient_id",
            "global_t0_sec",
            "global_stop_sec",
            "edf_sha256",
            "processed_window_sha256",
        ),
        "public assembly receipt",
    )
    if assembly["event_id"] != event_id or assembly["patient_id"] != patient_id:
        raise ValueError("public assembly identity mismatch")
    if _sha256(assembly["edf_sha256"], "public assembly EDF SHA-256") != waveform[
        "source_signal_sha256"
    ]:
        raise ValueError("public source signal hash does not match waveform")
    if _sha256(
        assembly["processed_window_sha256"],
        "public assembly processed-window SHA-256",
    ) != waveform["processed_window_sha256"]:
        raise ValueError("public processed-window hash does not match waveform")
    anchor_recording_time = _number(
        assembly["global_t0_sec"], "public assembly event anchor"
    )
    global_stop = _number(assembly["global_stop_sec"], "public assembly event stop")
    if global_stop <= anchor_recording_time:
        raise ValueError("public assembly event interval is invalid")

    typed = _mapping(record["typed_facts"], "public typed facts")
    phenotype = _mapping(typed.get("event_phenotype"), "public event phenotype")
    _require_keys(
        phenotype,
        (
            "onset_start_sec",
            "onset_end_sec",
            "first_visible_derivations",
            "later_visible_delay_sec",
            "later_visible_derivations",
            "receipt",
        ),
        "public event phenotype",
    )
    onset_start = _number(phenotype["onset_start_sec"], "public phenotype start")
    onset_end = _number(phenotype["onset_end_sec"], "public phenotype end")
    if onset_end <= onset_start or onset_start < anchor_recording_time or onset_end > global_stop:
        raise ValueError("public phenotype interval is inconsistent with the event")
    relative_interval = (
        onset_start - anchor_recording_time,
        onset_end - anchor_recording_time,
    )
    if not _same_pair(relative_interval, waveform["evidence_interval_sec"]):
        raise ValueError("public phenotype interval does not match waveform evidence")
    derivations = _canonical_derivations(
        phenotype["first_visible_derivations"],
        "public first-visible derivations",
    )
    later_derivations = phenotype["later_visible_derivations"]
    if not isinstance(later_derivations, list):
        raise TypeError("public later-visible derivations must be a list")
    observations: list[dict[str, Any]] = []
    if later_derivations:
        delay = _positive(
            phenotype["later_visible_delay_sec"],
            "public later-visible delay",
        )
        canonical_later = _canonical_derivations(
            later_derivations,
            "public later-visible derivations",
        )
        seen_electrodes: set[str] = set()
        for derivation in canonical_later:
            for electrode in derivation.split("-"):
                if electrode not in seen_electrodes:
                    seen_electrodes.add(electrode)
                    observations.append(
                        {"electrode": electrode, "delay_seconds": delay}
                    )
    elif phenotype["later_visible_delay_sec"] is not None:
        raise ValueError("public later-visible delay exists without derivations")
    receipt = _mapping(phenotype["receipt"], "public phenotype receipt")
    _require_keys(
        receipt,
        (
            "event_pseudonym",
            "patient_pseudonym",
            "signal_artifact_sha256",
            "causal_prefix_safe",
            "soz_labels_used_for_event_evidence",
            "private_labels_used_for_event_evidence",
            "time_coordinate_semantics",
        ),
        "public phenotype receipt",
    )
    if receipt["event_pseudonym"] != event_id or receipt["patient_pseudonym"] != patient_id:
        raise ValueError("public phenotype receipt identity mismatch")
    if _sha256(
        receipt["signal_artifact_sha256"],
        "public phenotype source-signal SHA-256",
    ) != waveform["source_signal_sha256"]:
        raise ValueError("public phenotype source hash does not match waveform")
    if (
        receipt["causal_prefix_safe"] is not True
        or receipt["soz_labels_used_for_event_evidence"] is not False
        or receipt["private_labels_used_for_event_evidence"] is not False
        or receipt["time_coordinate_semantics"] != "recording_start_seconds"
    ):
        raise ValueError("public phenotype receipt is not target-blind and coordinate-safe")
    return relative_interval, derivations, observations


def _metadata_fact(
    *,
    fact_id: str,
    fact_type: str,
    value: Mapping[str, Any],
    source_id: str,
    evidence_id: str,
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "section": "metadata",
        "fact_type": fact_type,
        "state": "present",
        "value": dict(value),
        "provenance": {
            "source_type": "signal_algorithm",
            "source_id": source_id,
            "method": "目标盲头皮脑电波形绑定",
        },
        "verification": {"status": "algorithm_candidate"},
        "evidence_ids": [evidence_id],
    }


def _event_fact(
    *,
    fact_id: str,
    fact_type: str,
    value: Mapping[str, Any],
    event_id: str,
    source_id: str,
    evidence_id: str,
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "section": "ictal",
        "fact_type": fact_type,
        "state": "uncertain",
        "value": dict(value),
        "provenance": {
            "source_type": "signal_algorithm",
            "source_id": source_id,
            "method": "目标盲头皮脑电波形候选提取",
        },
        "verification": {"status": "algorithm_candidate"},
        "evidence_ids": [evidence_id],
        "eeg_event_id": event_id,
    }


def _source_annotation_timing(
    value: Mapping[str, Any] | None,
    *,
    event_id: str,
    patient_id: str,
    source_signal_sha256: str,
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Project only controlled EDF point-marker coordinates.

    The private annotation ledger has already discarded raw descriptions and
    paths.  Patient-event/clip-note markers remain review context and are not
    projected.  Even EEG/SZ/end marker labels retain point-marker semantics;
    they do not become onset, termination, duration, or seizure facts.
    """

    if value is None:
        return None
    event = _mapping(value, "private annotation event")
    _exact_keys(
        event,
        (
            "schema_version",
            "event_id",
            "patient_id",
            "source_signal_sha256",
            "event_anchor_recording_seconds",
            "event_anchor_source",
            "markers",
            "binding_receipt",
        ),
        "private annotation event",
    )
    if event["schema_version"] != PRIVATE_ANNOTATION_EVENT_SCHEMA:
        raise ValueError("private annotation event schema drifted")
    if event["event_id"] != event_id or event["patient_id"] != patient_id:
        raise ValueError("private annotation event identity mismatch")
    if _sha256(
        event["source_signal_sha256"],
        "private annotation source signal SHA-256",
    ) != source_signal_sha256:
        raise ValueError("private annotation source signal does not match waveform")
    raw_markers = event["markers"]
    if not isinstance(raw_markers, list):
        raise TypeError("private annotation markers must be a list")
    kind_map = {
        "sz_index_point": "event_marker",
        "eeg_sz_point": "eeg_event_marker",
        "end_point": "end_marker",
    }
    projected: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_markers):
        marker = _mapping(raw, f"private annotation marker {index}")
        source_kind = marker.get("marker_kind")
        if source_kind not in kind_map:
            continue
        target_kind = kind_map[str(source_kind)]
        if target_kind in seen:
            raise ValueError(
                "private annotation contains ambiguous duplicate reportable marker kinds"
            )
        if (
            marker.get("semantics") != "point_marker_only_not_event_interval"
            or marker.get("requires_human_review") is not True
            or marker.get("clinical_fact_eligible") is not False
            or marker.get("deterministic_report_receipt_eligible") is not True
            or marker.get("llm_eligible") is not False
        ):
            raise ValueError("private annotation marker was promoted beyond source semantics")
        marker_id = _identifier(
            marker.get("marker_id"),
            f"private annotation marker {index} ID",
        )
        projected.append(
            {
                "marker_kind": target_kind,
                "recording_offset_seconds": _number(
                    marker.get("recording_offset_seconds"),
                    f"private annotation marker {index} recording offset",
                ),
                "point_marker": True,
            }
        )
        evidence_ids.append(marker_id)
        seen.add(target_kind)
    if not projected:
        return None
    projected.sort(
        key=lambda item: (
            float(item["recording_offset_seconds"]),
            str(item["marker_kind"]),
        )
    )
    return projected, evidence_ids


def adapt_trustworthy_clinical_eeg(
    qualified_report: Mapping[str, Any],
    waveform_entry: Mapping[str, Any],
    *,
    public_typed_source: Mapping[str, Any] | None = None,
    private_annotation_event: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt one qualified trustworthy-AI EEG unit into two strict payloads."""

    report = _mapping(qualified_report, "qualified report")
    waveform = _waveform_contract(_mapping(waveform_entry, "waveform entry"))
    _require_keys(
        report,
        (
            "schema_version",
            "cohort",
            "unit_id",
            "patient_id",
            "report_status",
            "facts_locked",
            "llm_used",
        ),
        "qualified report",
    )
    if report["schema_version"] != QUALIFIED_REPORT_SCHEMA:
        raise ValueError("qualified report schema drifted")
    if report["facts_locked"] is not True or report["llm_used"] is not False:
        raise ValueError("qualified report is not a deterministic facts-locked source")
    status = report["report_status"]
    if not isinstance(status, str) or not status.endswith("_facts_locked"):
        raise ValueError("qualified report status is not facts-locked")
    unit_id = _identifier(report["unit_id"], "qualified report unit_id")
    patient_id = _identifier(report["patient_id"], "qualified report patient_id")
    cohort = report["cohort"]

    if waveform["unit_id"] != unit_id or waveform["patient_id"] != patient_id:
        raise ValueError("qualified report/waveform identity mismatch")
    event_id = waveform["event_id"]
    if cohort == "private_post_open_target_blind_event_report":
        if (
            _PRIVATE_EVENT_RE.fullmatch(unit_id) is None
            or _PRIVATE_PATIENT_RE.fullmatch(patient_id) is None
            or event_id != unit_id
            or waveform["scope"] != "private_event"
            or waveform["representative_event"] is not False
        ):
            raise ValueError("private-event report/waveform contract mismatch")
        if public_typed_source is not None:
            raise ValueError("public typed source is forbidden for a private event")
        relative_interval, derivations, observations = _private_descriptor(
            report,
            patient_id=patient_id,
            event_id=event_id,
        )
        annotation_timing = _source_annotation_timing(
            private_annotation_event,
            event_id=event_id,
            patient_id=patient_id,
            source_signal_sha256=waveform["source_signal_sha256"],
        )
    elif cohort == "public_deepsoz_development_patient_report":
        if (
            _PUBLIC_PATIENT_RE.fullmatch(patient_id) is None
            or unit_id != patient_id
            or waveform["scope"] != "public_patient"
            or waveform["representative_event"] is not True
            or waveform["selection_policy"] != PUBLIC_SELECTION_POLICY
        ):
            raise ValueError("public-patient representative waveform contract mismatch")
        if public_typed_source is None:
            raise ValueError("public patient adaptation requires its typed event source")
        if private_annotation_event is not None:
            raise ValueError("private annotation event is forbidden for a public report")
        relative_interval, derivations, observations = _public_descriptor(
            _mapping(public_typed_source, "public typed source"),
            patient_id=patient_id,
            event_id=event_id,
            waveform=waveform,
        )
        annotation_timing = None
    else:
        raise ValueError(
            "only private_event and safely bound public_patient reports are supported"
        )

    if not _same_pair(relative_interval, waveform["evidence_interval_sec"]):
        raise ValueError("source descriptor interval does not match waveform evidence")
    window_start, window_stop = waveform["event_window_sec"]
    if relative_interval[0] < window_start or relative_interval[1] > window_stop:
        raise ValueError("source descriptor interval falls outside the 60-second segment")
    start_offset = relative_interval[0] + waveform["event_anchor_offset_seconds"]
    end_offset = relative_interval[1] + waveform["event_anchor_offset_seconds"]
    if (
        start_offset < 0
        or end_offset > SEGMENT_DURATION_SECONDS
        or end_offset <= start_offset
    ):
        raise ValueError("converted descriptor interval is outside the segment")
    if any(
        start_offset + float(observation["delay_seconds"])
        > SEGMENT_DURATION_SECONDS
        for observation in observations
    ):
        raise ValueError("later-visible observation is outside the 60-second segment")

    binding = {
        "scope": waveform["scope"],
        "unit_id": unit_id,
        "patient_id": patient_id,
        "event_id": event_id,
        "figure_file": waveform["figure_file"],
        "figure_sha256": waveform["figure_sha256"],
        "source_signal_sha256": waveform["source_signal_sha256"],
        "preprocessing_receipt_sha256": waveform[
            "preprocessing_receipt_sha256"
        ],
        "processed_window_sha256": waveform["processed_window_sha256"],
        "event_window_sec": list(waveform["event_window_sec"]),
        "evidence_interval_sec": list(relative_interval),
        "source_annotation_event_sha256": (
            _canonical_sha256(private_annotation_event)
            if private_annotation_event is not None
            else None
        ),
    }
    binding_digest = _canonical_sha256(binding)
    report_id = f"CER-{binding_digest[:24]}"
    evidence_id = f"EEG-WAVE-{binding_digest[:24]}"
    metadata_evidence_id = f"EEG-PROC-{binding_digest[:24]}"
    source_id = f"EEG-ALG-{binding_digest[:24]}"
    fact_suffix = binding_digest[:12].upper()

    occurrence_id = f"F-OCC-{fact_suffix}"
    sustained_id = f"F-SUST-{fact_suffix}"
    event_fact_ids = [occurrence_id, sustained_id]
    duration = end_offset - start_offset
    facts: list[dict[str, Any]] = [
        _metadata_fact(
            fact_id=f"F-DUR-{fact_suffix}",
            fact_type="recording_duration",
            value={"duration_seconds": SEGMENT_DURATION_SECONDS},
            source_id=source_id,
            evidence_id=metadata_evidence_id,
        ),
        _metadata_fact(
            fact_id=f"F-SETUP-{fact_suffix}",
            fact_type="electrode_setup",
            value={
                "system": "international_10_20",
                "electrodes": list(STANDARD_19),
                "montages": ["common_average"],
                "reference": "average",
            },
            source_id=source_id,
            evidence_id=metadata_evidence_id,
        ),
        _metadata_fact(
            fact_id=f"F-ACQ-{fact_suffix}",
            fact_type="acquisition_settings",
            value={
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "low_cut_hz": FILTER_HZ[0],
                "high_cut_hz": FILTER_HZ[1],
            },
            source_id=source_id,
            evidence_id=metadata_evidence_id,
        ),
        _event_fact(
            fact_id=occurrence_id,
            fact_type="electrographic_event_occurrence",
            value={
                "event_number": 1,
                "start_offset_seconds": start_offset,
                "duration_seconds": duration,
                "event_class": "uncertain_electrographic_pattern",
                "time_coordinate": "segment_start_seconds",
                "text_zh": "算法标记到意义不确定的持续脑电变化候选。",
            },
            event_id=event_id,
            source_id=source_id,
            evidence_id=evidence_id,
        ),
        _event_fact(
            fact_id=sustained_id,
            fact_type="algorithmic_sustained_eeg_change",
            value={
                "start_offset_seconds": start_offset,
                "end_offset_seconds": end_offset,
                "derivations": derivations,
                "text_zh": (
                    f"片段起点后{_format_number(start_offset)}至"
                    f"{_format_number(end_offset)}秒，"
                    f"{'、'.join(derivations)}导联出现算法标记的持续脑电变化候选。"
                ),
            },
            event_id=event_id,
            source_id=source_id,
            evidence_id=evidence_id,
        ),
    ]
    if annotation_timing is not None:
        markers, annotation_evidence_ids = annotation_timing
        facts.append(
            {
                "fact_id": f"F-EDFANN-{fact_suffix}",
                "section": "ictal",
                "fact_type": "source_eeg_annotation_timing",
                "state": "uncertain",
                "value": {
                    "time_coordinate": "original_recording_start_seconds",
                    "markers": markers,
                    "point_markers_only": True,
                    "onset_confirmed": False,
                    "termination_confirmed": False,
                    "duration_derived": False,
                },
                "provenance": {
                    "source_type": "acquisition_system",
                    "source_id": f"EEG-EDFANN-{binding_digest[:24]}",
                    "method": "原EDF点标记的受控类型投影；不推断起始、终止或时长",
                },
                "verification": {"status": "unverified"},
                "evidence_ids": annotation_evidence_ids,
                "eeg_event_id": event_id,
            }
        )
    if observations:
        later_id = f"F-LATER-{fact_suffix}"
        event_fact_ids.append(later_id)
        clauses = [
            f"{item['electrode']}在首段持续变化后"
            f"{_format_number(float(item['delay_seconds']))}秒可见后续头皮脑电变化候选"
            for item in observations
        ]
        facts.append(
            _event_fact(
                fact_id=later_id,
                fact_type="later_scalp_visible_eeg_change",
                value={
                    "observations": observations,
                    "temporal_relation_only": True,
                    "text_zh": "；".join(clauses) + "；仅表示时间先后关系。",
                },
                event_id=event_id,
                source_id=source_id,
                evidence_id=evidence_id,
            )
        )

    report_payload = {
        "schema_version": "clinical_eeg_report_v1",
        "report_id": report_id,
        "patient_pseudonym": patient_id,
        "facts": facts,
        "eeg_event_ids": [event_id],
        "impression_fact_ids": [],
    }
    validated_report = validate_report_payload(report_payload).to_dict()
    attachment = {
        "evidence_id": evidence_id,
        "fact_ids": event_fact_ids,
        "eeg_event_id": event_id,
        "figure_file": waveform["figure_file"],
        "figure_sha256": waveform["figure_sha256"],
        "source_signal_sha256": waveform["source_signal_sha256"],
        "preprocessing_receipt_sha256": waveform[
            "preprocessing_receipt_sha256"
        ],
        "processed_window_sha256": waveform["processed_window_sha256"],
        "channel_order": list(STANDARD_19),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "filter_hz": list(FILTER_HZ),
        "reference": REFERENCE,
        "event_window_seconds": list(EVENT_WINDOW_RELATIVE_SECONDS),
        "event_anchor_offset_seconds": EVENT_ANCHOR_OFFSET_SECONDS,
        "representative_event": waveform["representative_event"],
        "caption_zh": (
            "代表性事件的60秒处理后头皮脑电波形；"
            "标记区间为算法持续变化候选，需人工复核。"
            if waveform["representative_event"]
            else "当前报告事件的60秒处理后头皮脑电波形；"
            "标记区间为算法持续变化候选，需人工复核。"
        ),
    }
    waveform_manifest = {
        "schema_version": WAVEFORM_MANIFEST_SCHEMA,
        "report_id": report_id,
        "patient_pseudonym": patient_id,
        "selection_policy": WAVEFORM_SELECTION_POLICY,
        "attachments": [attachment],
    }
    return (
        deepcopy(validated_report),
        json.loads(
            json.dumps(
                waveform_manifest,
                ensure_ascii=False,
                allow_nan=False,
            )
        ),
    )


__all__ = [
    "EVENT_ANCHOR_OFFSET_SECONDS",
    "EVENT_WINDOW_RELATIVE_SECONDS",
    "FILTER_HZ",
    "PRIVATE_SELECTION_POLICY",
    "PUBLIC_SELECTION_POLICY",
    "REFERENCE",
    "SAMPLING_RATE_HZ",
    "SEGMENT_DURATION_SECONDS",
    "STANDARD_19",
    "WAVEFORM_MANIFEST_SCHEMA",
    "adapt_trustworthy_clinical_eeg",
]
