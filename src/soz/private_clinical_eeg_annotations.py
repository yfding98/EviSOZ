"""De-identified, fail-closed private EEG annotation ledger.

This module is intentionally separate from both the target-blind SOZ
descriptor path and ``clinical_eeg_report_v1``.  It reduces raw EDF annotation
descriptions to a small vocabulary of *point-marker* semantics and turns
spreadsheet rows into an unbound manual-review queue.  It never emits raw
descriptions, patient names, source paths, spreadsheet cell values, a seizure
duration, an EEG-onset assertion, or a physician-verification assertion.

The ledger can therefore be bound to a report event by pseudonymous identity
and source-signal SHA-256 without opening the raw annotation sources in the
report or LLM process.  A separate, explicitly authorized review step is
required before any spreadsheet content can become a clinical fact.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PRIVATE_ANNOTATION_LEDGER_SCHEMA = "private_clinical_eeg_annotation_ledger_v1"
PRIVATE_ANNOTATION_EVENT_SCHEMA = "private_clinical_eeg_annotation_event_v1"
PRIVATE_EXCEL_PENDING_SCHEMA = "private_clinical_eeg_excel_pending_review_v1"
PRIVATE_BUNDLE_SCHEMA = "soz_private_zero_adaptation_bundle_v18"

EVENT_WINDOW_SECONDS = (-12.0, 48.0)
SEGMENT_ANCHOR_OFFSET_SECONDS = 12.0

MARKER_KINDS = frozenset(
    {
        "eeg_sz_point",
        "sz_index_point",
        "end_point",
        "patient_event_point",
        "clip_note_point",
    }
)
DETERMINISTIC_REPORT_MARKER_KINDS = frozenset(
    {"eeg_sz_point", "sz_index_point", "end_point"}
)
MARKER_SCOPES = {
    "eeg_sz_point": "eeg_timing_annotation_candidate",
    "end_point": "ambiguous_timing_annotation",
    "sz_index_point": "event_index_annotation",
    "patient_event_point": "context_annotation",
    "clip_note_point": "context_annotation",
}

_EVENT_ID_RE = re.compile(r"^PRIV-E\d{4,}$")
_PATIENT_ID_RE = re.compile(r"^PRIV-P\d{3,}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SZ_LABEL_RE = re.compile(r"^SZ\d+(?:[-_]\d+)?$", re.IGNORECASE)
_SZ_MARKER_RE = re.compile(r"^SZ\s*\d+(?:[-_]\d+)?$", re.IGNORECASE)
_EEG_SZ_RE = re.compile(r"(?<![A-Z0-9])EEG\s*SZ(?![A-Z0-9])", re.IGNORECASE)
_PATIENT_EVENT_RE = re.compile(
    r"^PATIENT\s*EVENT(?:\s*[-_:]?\s*\d+)?$", re.IGNORECASE
)
_CLIP_NOTE_RE = re.compile(r"^CLIP\s*NOTE(?:\s*[-_:]?\s*\d+)?$", re.IGNORECASE)
_END_RE = re.compile(r"^(?:END|STOP|结束|终止)$", re.IGNORECASE)

_LEDGER_KEYS = {
    "schema_version",
    "status",
    "source_receipts",
    "events",
    "pending_excel_review",
    "exclusion_summary",
    "claim_boundary",
}
_EVENT_KEYS = {
    "schema_version",
    "event_id",
    "patient_id",
    "source_signal_sha256",
    "event_anchor_recording_seconds",
    "event_anchor_source",
    "markers",
    "binding_receipt",
}
_MARKER_KEYS = {
    "marker_id",
    "marker_kind",
    "marker_scope",
    "recording_offset_seconds",
    "annotation_duration_seconds",
    "is_point_marker",
    "event_relative_offset_seconds",
    "segment_offset_seconds",
    "semantics",
    "source_row",
    "source_row_sha256",
    "source_file_sha256",
    "source_signal_sha256",
    "requires_human_review",
    "clinical_fact_eligible",
    "deterministic_report_receipt_eligible",
    "llm_eligible",
}
_PENDING_KEYS = {
    "schema_version",
    "pending_id",
    "status",
    "source_workbook_sha256",
    "source_sheet_index",
    "source_row_number",
    "source_event_label",
    "source_row_sha256",
    "fields_present",
    "raw_content_included",
    "automatic_event_binding_allowed",
    "clinical_fact_eligible",
    "llm_eligible",
}
_FIELD_PRESENCE_KEYS = {
    "onset_description",
    "significant_electrodes",
    "early_spread",
    "all_channel_coverage",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be an integer")
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be an integer") from exc
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        as_float = float(number)
    if not math.isfinite(as_float) or not math.isclose(
        as_float, float(number), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(f"{context} must be an integer")
    if number < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return number


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{context} is not a supported pseudonymous identifier")
    return value


def _normalized_relative_edf(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} is missing")
    normalized = value.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        relative.is_absolute()
        or relative.suffix.lower() != ".edf"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{context} is unsafe")
    return relative.as_posix()


def _safe_source_edf(root: Path, relative_value: object, event_id: str) -> Path:
    relative = _normalized_relative_edf(relative_value, "private EDF path")
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"private source for {event_id} traverses a symlink")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"private source for {event_id} is not a regular file")
    return resolved


def _marker_kind(description: object) -> str | None:
    if not isinstance(description, str):
        return None
    text = re.sub(r"\s+", " ", description.strip())
    if not text:
        return None
    if _END_RE.fullmatch(text):
        return "end_point"
    if _PATIENT_EVENT_RE.fullmatch(text):
        return "patient_event_point"
    if _CLIP_NOTE_RE.fullmatch(text):
        return "clip_note_point"
    if _SZ_MARKER_RE.fullmatch(text):
        return "sz_index_point"
    if _EEG_SZ_RE.search(text):
        # Any morphology or clinical prose surrounding the token is discarded.
        return "eeg_sz_point"
    return None


def _source_row_hash(row: Mapping[str, object]) -> str:
    serializable = {
        str(key): "" if value is None else str(value)
        for key, value in row.items()
    }
    return _canonical_sha256(serializable)


def _private_patient_map(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    raw_patients = sorted({str(row.get("base_patient_id", "")).strip() for row in rows})
    if not raw_patients or raw_patients[0] == "":
        raise ValueError("private source manifest contains an empty patient identity")
    return {
        raw: f"PRIV-P{index + 1:03d}" for index, raw in enumerate(raw_patients)
    }


def _source_receipts(value: Mapping[str, object]) -> dict[str, Any]:
    required = ("edf_annotations_sha256", "private_manifest_sha256", "signal_roster_sha256")
    if set(value) != {*required, "workbook_sha256s"}:
        raise ValueError("source receipt shape drifted")
    result = {key: _sha256(value[key], f"source_receipts.{key}") for key in required}
    workbook_hashes = value["workbook_sha256s"]
    if not isinstance(workbook_hashes, list):
        raise TypeError("source_receipts.workbook_sha256s must be a list")
    normalized = [_sha256(item, "source workbook SHA-256") for item in workbook_hashes]
    if len(normalized) != len(set(normalized)):
        raise ValueError("source workbook SHA-256 values repeat")
    result["workbook_sha256s"] = normalized
    return result


def _validate_marker(
    value: object,
    *,
    expected_event_id: str,
    expected_source_signal_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _MARKER_KEYS:
        raise ValueError("annotation marker shape drifted")
    marker_id = value["marker_id"]
    if not isinstance(marker_id, str) or re.fullmatch(r"EDFANN-[0-9a-f]{24}", marker_id) is None:
        raise ValueError("annotation marker ID is invalid")
    kind = value["marker_kind"]
    if kind not in MARKER_KINDS:
        raise ValueError("annotation marker kind is unsupported")
    if value["marker_scope"] != MARKER_SCOPES[kind]:
        raise ValueError("annotation marker scope does not match its kind")
    recording = _finite_number(value["recording_offset_seconds"], "marker recording offset")
    if recording < 0:
        raise ValueError("marker recording offset must be nonnegative")
    annotation_duration = _finite_number(
        value["annotation_duration_seconds"], "marker annotation duration"
    )
    if not math.isclose(annotation_duration, 0.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("only zero-duration point annotations are supported")
    if value["is_point_marker"] is not True:
        raise ValueError("annotation must remain an explicit point marker")
    relative = _finite_number(value["event_relative_offset_seconds"], "marker relative offset")
    segment = _finite_number(value["segment_offset_seconds"], "marker segment offset")
    if not math.isclose(
        segment,
        relative + SEGMENT_ANCHOR_OFFSET_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("marker segment and event-relative offsets disagree")
    if segment < -1e-6 or segment > 60.0 + 1e-6:
        raise ValueError("marker falls outside the report segment")
    if value["semantics"] != "point_marker_only_not_event_interval":
        raise ValueError("annotation marker was promoted beyond point semantics")
    if value["requires_human_review"] is not True:
        raise ValueError("annotation marker must require human review")
    if value["clinical_fact_eligible"] is not False or value["llm_eligible"] is not False:
        raise ValueError("source point markers cannot directly enter facts or the LLM")
    report_receipt_eligible = kind in DETERMINISTIC_REPORT_MARKER_KINDS
    if value["deterministic_report_receipt_eligible"] is not report_receipt_eligible:
        raise ValueError("annotation marker report-receipt eligibility drifted")
    source_row = _integer(value["source_row"], "marker source row", minimum=1)
    source_row_sha256 = _sha256(value["source_row_sha256"], "marker row SHA-256")
    source_file_sha256 = _sha256(value["source_file_sha256"], "marker source SHA-256")
    source_signal_sha256 = _sha256(
        value["source_signal_sha256"], "marker source signal SHA-256"
    )
    expected_marker_digest = _canonical_sha256(
        {
            "event_id": expected_event_id,
            "marker_kind": kind,
            "recording_offset_seconds": recording,
            "source_row": source_row,
            "source_row_sha256": source_row_sha256,
            "source_file_sha256": source_file_sha256,
            "source_signal_sha256": source_signal_sha256,
        }
    )
    if marker_id != f"EDFANN-{expected_marker_digest[:24]}":
        raise ValueError("annotation marker ID does not bind its complete source receipt")
    return {
        "marker_id": marker_id,
        "marker_kind": kind,
        "marker_scope": value["marker_scope"],
        "recording_offset_seconds": recording,
        "annotation_duration_seconds": 0.0,
        "is_point_marker": True,
        "event_relative_offset_seconds": relative,
        "segment_offset_seconds": segment,
        "semantics": "point_marker_only_not_event_interval",
        "source_row": source_row,
        "source_row_sha256": source_row_sha256,
        "source_file_sha256": source_file_sha256,
        "source_signal_sha256": source_signal_sha256,
        "requires_human_review": True,
        "clinical_fact_eligible": False,
        "deterministic_report_receipt_eligible": report_receipt_eligible,
        "llm_eligible": False,
    }


def _validate_event(value: object, receipts: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _EVENT_KEYS:
        raise ValueError("private annotation event shape drifted")
    if value["schema_version"] != PRIVATE_ANNOTATION_EVENT_SCHEMA:
        raise ValueError("private annotation event schema drifted")
    event_id = _identifier(value["event_id"], "event_id", _EVENT_ID_RE)
    patient_id = _identifier(value["patient_id"], "patient_id", _PATIENT_ID_RE)
    anchor = _finite_number(value["event_anchor_recording_seconds"], "event anchor")
    if anchor < 0:
        raise ValueError("event anchor must be nonnegative")
    anchor_source = value["event_anchor_source"]
    if anchor_source not in {"exact_sz_marker", "first_sz_marker"}:
        raise ValueError("event anchor source is unsupported")
    source_signal_sha256 = _sha256(
        value["source_signal_sha256"], "source signal SHA-256"
    )
    markers = value["markers"]
    if not isinstance(markers, list):
        raise TypeError("event markers must be a list")
    normalized_markers = [
        _validate_marker(
            item,
            expected_event_id=event_id,
            expected_source_signal_sha256=source_signal_sha256,
        )
        for item in markers
    ]
    marker_ids = [item["marker_id"] for item in normalized_markers]
    if len(marker_ids) != len(set(marker_ids)):
        raise ValueError("event contains duplicate marker IDs")
    for marker in normalized_markers:
        if marker["source_file_sha256"] != receipts["edf_annotations_sha256"]:
            raise ValueError("marker source file receipt is inconsistent")
        if marker["source_signal_sha256"] != source_signal_sha256:
            raise ValueError("marker source-signal receipt is inconsistent")
    binding = value["binding_receipt"]
    if not isinstance(binding, dict) or set(binding) != {
        "signal_roster_row_sha256",
        "private_manifest_row_sha256",
        "source_row_number",
        "raw_description_included",
        "raw_path_included",
    }:
        raise ValueError("event binding receipt shape drifted")
    source_row_number = _integer(binding["source_row_number"], "source row", minimum=1)
    expected_event_id = f"PRIV-E{source_row_number:04d}"
    if event_id != expected_event_id:
        raise ValueError("event ID is inconsistent with the frozen source row")
    if binding["raw_description_included"] is not False or binding["raw_path_included"] is not False:
        raise ValueError("private annotation binding leaks raw content")
    return {
        "schema_version": PRIVATE_ANNOTATION_EVENT_SCHEMA,
        "event_id": event_id,
        "patient_id": patient_id,
        "source_signal_sha256": source_signal_sha256,
        "event_anchor_recording_seconds": anchor,
        "event_anchor_source": anchor_source,
        "markers": normalized_markers,
        "binding_receipt": {
            "signal_roster_row_sha256": _sha256(
                binding["signal_roster_row_sha256"], "signal roster row SHA-256"
            ),
            "private_manifest_row_sha256": _sha256(
                binding["private_manifest_row_sha256"], "private manifest row SHA-256"
            ),
            "source_row_number": source_row_number,
            "raw_description_included": False,
            "raw_path_included": False,
        },
    }


def _validate_pending(value: object, workbook_hashes: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _PENDING_KEYS:
        raise ValueError("pending Excel review row shape drifted")
    if value["schema_version"] != PRIVATE_EXCEL_PENDING_SCHEMA:
        raise ValueError("pending Excel review schema drifted")
    pending_id = value["pending_id"]
    if not isinstance(pending_id, str) or re.fullmatch(r"XLSREV-[0-9a-f]{24}", pending_id) is None:
        raise ValueError("pending Excel review ID is invalid")
    workbook_sha = _sha256(value["source_workbook_sha256"], "workbook SHA-256")
    if workbook_sha not in workbook_hashes:
        raise ValueError("pending Excel review source is not receipted")
    sheet_index = _integer(value["source_sheet_index"], "source sheet index")
    row_number = _integer(value["source_row_number"], "source row number", minimum=1)
    event_label = value["source_event_label"]
    if not isinstance(event_label, str) or _SZ_LABEL_RE.fullmatch(event_label) is None:
        raise ValueError("pending Excel source event label is invalid")
    fields = value["fields_present"]
    if not isinstance(fields, dict) or set(fields) != _FIELD_PRESENCE_KEYS:
        raise ValueError("pending Excel field-presence shape drifted")
    if not all(isinstance(fields[key], bool) for key in _FIELD_PRESENCE_KEYS):
        raise TypeError("pending Excel field-presence values must be boolean")
    if not any(fields.values()):
        raise ValueError("pending Excel row has no source annotation fields")
    if value["status"] != "pending_review":
        raise ValueError("Excel annotation must remain pending review")
    if value["raw_content_included"] is not False:
        raise ValueError("pending Excel row leaks raw content")
    if value["automatic_event_binding_allowed"] is not False:
        raise ValueError("pending Excel row permits unsafe automatic binding")
    if value["clinical_fact_eligible"] is not False or value["llm_eligible"] is not False:
        raise ValueError("pending Excel row cannot enter facts or the LLM")
    return {
        "schema_version": PRIVATE_EXCEL_PENDING_SCHEMA,
        "pending_id": pending_id,
        "status": "pending_review",
        "source_workbook_sha256": workbook_sha,
        "source_sheet_index": sheet_index,
        "source_row_number": row_number,
        "source_event_label": event_label.upper(),
        "source_row_sha256": _sha256(value["source_row_sha256"], "Excel row SHA-256"),
        "fields_present": {key: fields[key] for key in sorted(_FIELD_PRESENCE_KEYS)},
        "raw_content_included": False,
        "automatic_event_binding_allowed": False,
        "clinical_fact_eligible": False,
        "llm_eligible": False,
    }


def validate_private_annotation_ledger(
    value: object,
    *,
    expected_event_id: str | None = None,
    expected_patient_id: str | None = None,
    expected_source_signal_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and return a canonical deep copy of a private ledger.

    Optional expected values select exactly one report event and enforce the
    final current-record binding used by a caller.  The returned ledger still
    contains all events; callers should use :func:`select_private_annotation_event`
    when they need the selected record itself.
    """

    if not isinstance(value, dict) or set(value) != _LEDGER_KEYS:
        raise ValueError("private annotation ledger shape drifted")
    if value["schema_version"] != PRIVATE_ANNOTATION_LEDGER_SCHEMA:
        raise ValueError("private annotation ledger schema drifted")
    if value["status"] != "completed_deidentified_point_markers_excel_pending_review":
        raise ValueError("private annotation ledger is not complete")
    receipts = _source_receipts(value["source_receipts"])
    events = value["events"]
    if not isinstance(events, list) or not events:
        raise TypeError("private annotation ledger has no events")
    normalized_events = [_validate_event(item, receipts) for item in events]
    event_ids = [item["event_id"] for item in normalized_events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("private annotation ledger repeats an event ID")
    pending = value["pending_excel_review"]
    if not isinstance(pending, list):
        raise TypeError("pending Excel review queue must be a list")
    workbook_hashes = set(receipts["workbook_sha256s"])
    normalized_pending = [_validate_pending(item, workbook_hashes) for item in pending]
    pending_ids = [item["pending_id"] for item in normalized_pending]
    if len(pending_ids) != len(set(pending_ids)):
        raise ValueError("pending Excel review queue repeats an ID")
    exclusion = value["exclusion_summary"]
    required_exclusions = {
        "raw_annotation_rows",
        "recognized_point_markers",
        "positive_duration_rows_excluded",
        "unrecognized_descriptions_excluded",
        "markers_outside_selected_windows_excluded",
        "excel_rows_pending_review",
    }
    if not isinstance(exclusion, dict) or set(exclusion) != required_exclusions:
        raise ValueError("annotation exclusion summary shape drifted")
    normalized_exclusion = {
        key: _integer(exclusion[key], f"exclusion_summary.{key}")
        for key in sorted(required_exclusions)
    }
    marker_count = sum(len(event["markers"]) for event in normalized_events)
    if normalized_exclusion["recognized_point_markers"] != marker_count:
        raise ValueError("recognized marker count does not match event records")
    if normalized_exclusion["excel_rows_pending_review"] != len(normalized_pending):
        raise ValueError("pending Excel count does not match its queue")
    boundary = value["claim_boundary"]
    expected_boundary = {
        "point_marker_is_event_interval": False,
        "point_marker_is_electrographic_onset": False,
        "point_marker_is_seizure_diagnosis": False,
        "end_marker_is_verified_eeg_termination": False,
        "spreadsheet_row_automatically_bound": False,
        "physician_verification_inferred": False,
        "raw_description_or_path_released": False,
        "ledger_content_sent_to_llm": False,
    }
    if boundary != expected_boundary:
        raise ValueError("private annotation claim boundary drifted")

    normalized = {
        "schema_version": PRIVATE_ANNOTATION_LEDGER_SCHEMA,
        "status": "completed_deidentified_point_markers_excel_pending_review",
        "source_receipts": receipts,
        "events": normalized_events,
        "pending_excel_review": normalized_pending,
        "exclusion_summary": normalized_exclusion,
        "claim_boundary": expected_boundary,
    }
    if expected_event_id is not None:
        event_id = _identifier(expected_event_id, "expected event_id", _EVENT_ID_RE)
        matches = [item for item in normalized_events if item["event_id"] == event_id]
        if len(matches) != 1:
            raise ValueError("ledger must contain exactly one expected event")
        selected = matches[0]
        if expected_patient_id is not None and selected["patient_id"] != _identifier(
            expected_patient_id, "expected patient_id", _PATIENT_ID_RE
        ):
            raise ValueError("private annotation patient binding mismatch")
        if expected_source_signal_sha256 is not None and selected[
            "source_signal_sha256"
        ] != _sha256(expected_source_signal_sha256, "expected source signal SHA-256"):
            raise ValueError("private annotation source-signal binding mismatch")
    elif expected_patient_id is not None or expected_source_signal_sha256 is not None:
        raise ValueError("patient/source expectations require expected_event_id")
    return deepcopy(normalized)


def select_private_annotation_event(
    ledger: object,
    *,
    event_id: str,
    patient_id: str,
    source_signal_sha256: str,
) -> dict[str, Any]:
    """Return one strictly bound event without exposing raw source content."""

    normalized = validate_private_annotation_ledger(
        ledger,
        expected_event_id=event_id,
        expected_patient_id=patient_id,
        expected_source_signal_sha256=source_signal_sha256,
    )
    return deepcopy(next(item for item in normalized["events"] if item["event_id"] == event_id))


def _pending_excel_row(
    *,
    workbook_sha256: str,
    sheet_index: int,
    row_number: int,
    event_label: str,
    raw_source: Mapping[str, object],
    fields_present: Mapping[str, bool],
) -> dict[str, Any]:
    row_hash = _canonical_sha256(
        {
            "workbook_sha256": workbook_sha256,
            "sheet_index": sheet_index,
            "row_number": row_number,
            "event_label": event_label,
            "raw_source": {
                str(key): "" if value is None else str(value)
                for key, value in raw_source.items()
            },
        }
    )
    pending_digest = _canonical_sha256(
        {
            "workbook_sha256": workbook_sha256,
            "sheet_index": sheet_index,
            "row_number": row_number,
            "event_label": event_label,
            "source_row_sha256": row_hash,
        }
    )
    return {
        "schema_version": PRIVATE_EXCEL_PENDING_SCHEMA,
        "pending_id": f"XLSREV-{pending_digest[:24]}",
        "status": "pending_review",
        "source_workbook_sha256": workbook_sha256,
        "source_sheet_index": sheet_index,
        "source_row_number": row_number,
        "source_event_label": event_label.upper(),
        "source_row_sha256": row_hash,
        "fields_present": dict(fields_present),
        "raw_content_included": False,
        "automatic_event_binding_allowed": False,
        "clinical_fact_eligible": False,
        "llm_eligible": False,
    }


def build_private_annotation_ledger(
    *,
    edf_annotation_rows: Sequence[Mapping[str, object]],
    private_manifest_rows: Sequence[Mapping[str, object]],
    signal_roster_rows: Sequence[Mapping[str, object]],
    source_file_hashes: Mapping[str, object],
    source_signal_hashes: Mapping[str, str],
    excel_pending_rows: Sequence[Mapping[str, object]] = (),
    selected_event_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the pure in-memory ledger from already loaded private sources.

    ``private_manifest_rows`` and ``signal_roster_rows`` are joined exclusively
    through the frozen one-based ``source_row`` receipt.  Raw patient identity
    and paths are used only for this validation and are never copied out.
    ``excel_pending_rows`` must already contain the de-identified queue rows
    produced by :func:`read_excel_pending_review`.
    """

    if not private_manifest_rows or not signal_roster_rows:
        raise ValueError("private manifest and signal roster must be nonempty")
    if len(private_manifest_rows) != len(signal_roster_rows):
        raise ValueError("private manifest and frozen signal roster have different sizes")
    receipts = _source_receipts(source_file_hashes)
    patient_map = _private_patient_map(private_manifest_rows)
    annotations_by_path: dict[
        str, list[tuple[Mapping[str, object], int]]
    ] = {}
    raw_annotation_count = 0
    positive_duration_count = 0
    unrecognized_count = 0
    annotation_indices: set[tuple[str, str]] = set()
    for row_number, row in enumerate(edf_annotation_rows, start=2):
        if not isinstance(row, Mapping):
            raise TypeError(f"EDF annotation row {row_number} is not an object")
        ann_idx = str(row.get("ann_idx", "") or "").strip()
        onset = str(row.get("onset_sec", "") or "").strip()
        if not ann_idx and not onset:
            continue
        raw_annotation_count += 1
        relative = _normalized_relative_edf(row.get("edf_path"), "EDF annotation path")
        annotation_key = (relative, ann_idx)
        if annotation_key in annotation_indices:
            raise ValueError("duplicate EDF annotation marker/index")
        annotation_indices.add(annotation_key)
        declared_source_row = row.get("source_row")
        source_row_number = (
            _integer(declared_source_row, "EDF annotation source_row", minimum=1)
            if declared_source_row not in (None, "")
            else row_number
        )
        annotations_by_path.setdefault(relative, []).append((row, source_row_number))
        try:
            annotation_duration = _finite_number(
                row.get("duration_ann_sec"), "annotation duration"
            )
        except (TypeError, ValueError):
            positive_duration_count += 1
            continue
        if not math.isclose(annotation_duration, 0.0, rel_tol=0.0, abs_tol=1e-9):
            positive_duration_count += 1
        elif _marker_kind(row.get("description")) is None:
            unrecognized_count += 1

    selected = None
    if selected_event_ids is not None:
        selected = {
            _identifier(item, "selected event_id", _EVENT_ID_RE)
            for item in selected_event_ids
        }
        if not selected:
            raise ValueError("selected_event_ids must not be empty")

    roster_by_source_row: dict[int, Mapping[str, object]] = {}
    for roster in signal_roster_rows:
        source_row = _integer(roster.get("source_row"), "signal roster source_row", minimum=1)
        if source_row in roster_by_source_row:
            raise ValueError("signal roster repeats a frozen source row")
        roster_by_source_row[source_row] = roster
    if set(roster_by_source_row) != set(range(1, len(private_manifest_rows) + 1)):
        raise ValueError("signal roster source rows are not a complete frozen permutation")

    events: list[dict[str, Any]] = []
    outside_window_count = 0
    marker_count = 0
    for source_row, raw_source in enumerate(private_manifest_rows, start=1):
        roster = roster_by_source_row[source_row]
        event_id = _identifier(roster.get("event_id"), "signal roster event_id", _EVENT_ID_RE)
        if event_id != f"PRIV-E{source_row:04d}":
            raise ValueError("signal roster event ID/source-row binding drifted")
        patient_id = _identifier(roster.get("patient_id"), "signal roster patient_id", _PATIENT_ID_RE)
        raw_patient = str(raw_source.get("base_patient_id", "")).strip()
        if patient_map.get(raw_patient) != patient_id:
            raise ValueError(f"signal roster patient binding drifted for {event_id}")
        source_relative = _normalized_relative_edf(raw_source.get("edf_path"), "source manifest EDF path")
        roster_relative = _normalized_relative_edf(roster.get("relative_edf_path"), "signal roster EDF path")
        if source_relative != roster_relative:
            raise ValueError(f"signal roster source path binding drifted for {event_id}")
        if selected is not None and event_id not in selected:
            continue
        anchor = _finite_number(roster.get("global_event_t0_sec"), "event anchor")
        source_anchor = _finite_number(raw_source.get("sz_start"), "source event anchor")
        if not math.isclose(anchor, source_anchor, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"signal roster event anchor drifted for {event_id}")
        anchor_source = str(roster.get("time_source", ""))
        if anchor_source not in {"exact_sz_marker", "first_sz_marker"}:
            raise ValueError(f"unsupported event anchor source for {event_id}")
        source_signal_sha256 = _sha256(
            source_signal_hashes.get(event_id),
            f"source signal SHA-256 for {event_id}",
        )
        markers: list[dict[str, Any]] = []
        for annotation, annotation_source_row in annotations_by_path.get(
            source_relative, []
        ):
            try:
                duration = _finite_number(annotation.get("duration_ann_sec"), "annotation duration")
                recording_offset = _finite_number(annotation.get("onset_sec"), "annotation onset")
            except (TypeError, ValueError):
                continue
            if not math.isclose(duration, 0.0, rel_tol=0.0, abs_tol=1e-9):
                continue
            kind = _marker_kind(annotation.get("description"))
            if kind is None:
                continue
            relative_offset = recording_offset - anchor
            segment_offset = relative_offset + SEGMENT_ANCHOR_OFFSET_SECONDS
            if (
                relative_offset < EVENT_WINDOW_SECONDS[0] - 1e-6
                or relative_offset > EVENT_WINDOW_SECONDS[1] + 1e-6
            ):
                outside_window_count += 1
                continue
            row_hash = _source_row_hash(annotation)
            marker_digest = _canonical_sha256(
                {
                    "event_id": event_id,
                    "marker_kind": kind,
                    "recording_offset_seconds": recording_offset,
                    "source_row": annotation_source_row,
                    "source_row_sha256": row_hash,
                    "source_file_sha256": receipts["edf_annotations_sha256"],
                    "source_signal_sha256": source_signal_sha256,
                }
            )
            markers.append(
                {
                    "marker_id": f"EDFANN-{marker_digest[:24]}",
                    "marker_kind": kind,
                    "marker_scope": MARKER_SCOPES[kind],
                    "recording_offset_seconds": recording_offset,
                    "annotation_duration_seconds": 0.0,
                    "is_point_marker": True,
                    "event_relative_offset_seconds": relative_offset,
                    "segment_offset_seconds": segment_offset,
                    "semantics": "point_marker_only_not_event_interval",
                    "source_row": annotation_source_row,
                    "source_row_sha256": row_hash,
                    "source_file_sha256": receipts["edf_annotations_sha256"],
                    "source_signal_sha256": source_signal_sha256,
                    "requires_human_review": True,
                    "clinical_fact_eligible": False,
                    "deterministic_report_receipt_eligible": (
                        kind in DETERMINISTIC_REPORT_MARKER_KINDS
                    ),
                    "llm_eligible": False,
                }
            )
        markers.sort(key=lambda item: (float(item["recording_offset_seconds"]), str(item["marker_id"])))
        if len({item["marker_id"] for item in markers}) != len(markers):
            raise ValueError(f"duplicate EDF annotation marker for {event_id}")
        marker_count += len(markers)
        events.append(
            {
                "schema_version": PRIVATE_ANNOTATION_EVENT_SCHEMA,
                "event_id": event_id,
                "patient_id": patient_id,
                "source_signal_sha256": source_signal_sha256,
                "event_anchor_recording_seconds": anchor,
                "event_anchor_source": anchor_source,
                "markers": markers,
                "binding_receipt": {
                    "signal_roster_row_sha256": _source_row_hash(roster),
                    "private_manifest_row_sha256": _source_row_hash(raw_source),
                    "source_row_number": source_row,
                    "raw_description_included": False,
                    "raw_path_included": False,
                },
            }
        )
    if selected is not None and {item["event_id"] for item in events} != selected:
        raise ValueError("one or more selected private events were not found")

    ledger = {
        "schema_version": PRIVATE_ANNOTATION_LEDGER_SCHEMA,
        "status": "completed_deidentified_point_markers_excel_pending_review",
        "source_receipts": receipts,
        "events": events,
        "pending_excel_review": [dict(item) for item in excel_pending_rows],
        "exclusion_summary": {
            "raw_annotation_rows": raw_annotation_count,
            "recognized_point_markers": marker_count,
            "positive_duration_rows_excluded": positive_duration_count,
            "unrecognized_descriptions_excluded": unrecognized_count,
            "markers_outside_selected_windows_excluded": outside_window_count,
            "excel_rows_pending_review": len(excel_pending_rows),
        },
        "claim_boundary": {
            "point_marker_is_event_interval": False,
            "point_marker_is_electrographic_onset": False,
            "point_marker_is_seizure_diagnosis": False,
            "end_marker_is_verified_eeg_termination": False,
            "spreadsheet_row_automatically_bound": False,
            "physician_verification_inferred": False,
            "raw_description_or_path_released": False,
            "ledger_content_sent_to_llm": False,
        },
    }
    return validate_private_annotation_ledger(ledger)


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _spreadsheet_read_path(path: Path, temporary_root: Path) -> Path:
    if path.suffix.lower() != ".xls":
        return path
    config_root = temporary_root / "config"
    cache_root = temporary_root / "cache"
    runtime_root = temporary_root / "runtime"
    config_root.mkdir()
    cache_root.mkdir()
    runtime_root.mkdir()
    runtime_root.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "XDG_CONFIG_HOME": str(config_root),
            "XDG_CACHE_HOME": str(cache_root),
            "XDG_RUNTIME_DIR": str(runtime_root),
            "SAL_USE_VCLPLUGIN": "svp",
        }
    )
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--invisible",
            "--nodefault",
            "--nolockcheck",
            "--nologo",
            "--nofirststartwizard",
            f"-env:UserInstallation={(temporary_root / 'lo-profile').resolve().as_uri()}",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(temporary_root),
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    converted = temporary_root / f"{path.stem}.xlsx"
    if result.returncode != 0 or not converted.is_file():
        raise RuntimeError("legacy Excel conversion failed")
    return converted


def read_excel_pending_review(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return PHI-free pending rows and the corresponding workbook hashes."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pandas is required to inspect private annotation workbooks") from exc

    pending: list[dict[str, Any]] = []
    workbook_hashes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="private-eeg-excel-") as temporary:
        temporary_root = Path(temporary)
        for workbook_path in paths:
            workbook = workbook_path.resolve(strict=True)
            if workbook.suffix.lower() not in {".xls", ".xlsx", ".xlsm"}:
                raise ValueError("private annotation workbook format is unsupported")
            workbook_sha = _sha256_file(workbook)
            if workbook_sha in workbook_hashes:
                raise ValueError("private annotation workbook is repeated")
            workbook_hashes.append(workbook_sha)
            read_path = _spreadsheet_read_path(workbook, temporary_root)
            excel = pd.ExcelFile(read_path)
            for sheet_index, sheet_name in enumerate(excel.sheet_names):
                frame = pd.read_excel(read_path, sheet_name=sheet_name, header=None)
                if frame.shape[0] < 3:
                    continue
                header_top = [_cell(value) for value in frame.iloc[0].tolist()]
                header_sub = [_cell(value) for value in frame.iloc[1].tolist()]
                groups: dict[str, dict[str, int]] = {}
                current: str | None = None
                for column, (top, sub) in enumerate(zip(header_top, header_sub)):
                    if _SZ_LABEL_RE.fullmatch(top):
                        current = top.upper()
                        groups.setdefault(current, {})
                    if current is None:
                        continue
                    if "起始" in sub:
                        groups[current]["onset_description"] = column
                    elif "显著" in sub:
                        groups[current]["significant_electrodes"] = column
                    elif "扩散" in sub:
                        groups[current]["early_spread"] = column
                    elif "覆盖" in sub:
                        groups[current]["all_channel_coverage"] = column
                for row_index in range(2, len(frame)):
                    raw_row = {
                        f"column_{column}": _cell(frame.iat[row_index, column])
                        for column in range(frame.shape[1])
                    }
                    for event_label, columns in groups.items():
                        fields_present = {
                            field: bool(_cell(frame.iat[row_index, column]))
                            for field, column in columns.items()
                        }
                        fields_present = {
                            field: bool(fields_present.get(field, False))
                            for field in _FIELD_PRESENCE_KEYS
                        }
                        if not any(fields_present.values()):
                            continue
                        pending.append(
                            _pending_excel_row(
                                workbook_sha256=workbook_sha,
                                sheet_index=sheet_index,
                                row_number=row_index + 1,
                                event_label=event_label,
                                raw_source=raw_row,
                                fields_present=fields_present,
                            )
                        )
    return pending, workbook_hashes


def materialize_private_annotation_ledger(
    *,
    private_bundle_directory: Path,
    edf_annotations_path: Path,
    workbook_paths: Sequence[Path],
    output_path: Path,
    source_manifest_path: Path | None = None,
    eeg_root: Path | None = None,
    selected_event_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build and atomically write one PHI-free private annotation ledger."""

    if selected_event_ids is None:
        raise ValueError(
            "report-time annotation materialization requires explicit selected_event_ids"
        )

    bundle_root = private_bundle_directory.resolve(strict=True)
    bundle_manifest_path = bundle_root / "manifest.json"
    bundle_manifest = _read_json(bundle_manifest_path)
    if bundle_manifest.get("schema_version") != PRIVATE_BUNDLE_SCHEMA:
        raise ValueError("private frozen bundle schema drifted")
    files = bundle_manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("private frozen bundle has no file receipt")
    roster_name = files.get("target_free_signal_roster")
    if not isinstance(roster_name, str) or Path(roster_name).name != roster_name:
        raise ValueError("private signal roster basename is unsafe")
    roster_path = (bundle_root / roster_name).resolve(strict=True)
    if source_manifest_path is None:
        declared_source = bundle_manifest.get("source_manifest")
        if not isinstance(declared_source, str) or not declared_source:
            raise ValueError("private bundle has no source manifest")
        source_manifest = Path(declared_source).resolve(strict=True)
    else:
        source_manifest = source_manifest_path.resolve(strict=True)
    declared_root = bundle_manifest.get("eeg_root")
    if eeg_root is None:
        if not isinstance(declared_root, str) or not declared_root:
            raise ValueError("private bundle has no EEG root")
        source_root = Path(declared_root).resolve(strict=True)
    else:
        source_root = eeg_root.resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("private EEG root is not a directory")

    annotations_file = edf_annotations_path.resolve(strict=True)
    annotation_rows = _read_csv(annotations_file)
    private_rows = _read_csv(source_manifest)
    roster_rows = _read_csv(roster_path)
    pending, workbook_hashes = read_excel_pending_review(workbook_paths)

    requested = set(selected_event_ids)
    if not requested:
        raise ValueError("selected_event_ids must not be empty")
    source_signal_hashes: dict[str, str] = {}
    for roster in roster_rows:
        event_id = str(roster.get("event_id", ""))
        if requested is not None and event_id not in requested:
            continue
        source = _safe_source_edf(source_root, roster.get("relative_edf_path"), event_id)
        source_signal_hashes[event_id] = _sha256_file(source)

    ledger = build_private_annotation_ledger(
        edf_annotation_rows=annotation_rows,
        private_manifest_rows=private_rows,
        signal_roster_rows=roster_rows,
        source_file_hashes={
            "edf_annotations_sha256": _sha256_file(annotations_file),
            "private_manifest_sha256": _sha256_file(source_manifest),
            "signal_roster_sha256": _sha256_file(roster_path),
            "workbook_sha256s": workbook_hashes,
        },
        source_signal_hashes=source_signal_hashes,
        excel_pending_rows=pending,
        selected_event_ids=requested,
    )

    target = output_path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    published = False
    try:
        temporary_path.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Re-open through the public validator before publication.
        validate_private_annotation_ledger(_read_json(temporary_path))
        os.replace(temporary_path, target)
        published = True
    finally:
        if not published and temporary_path.exists():
            temporary_path.unlink()
    return deepcopy(ledger)


__all__ = [
    "EVENT_WINDOW_SECONDS",
    "MARKER_KINDS",
    "PRIVATE_ANNOTATION_EVENT_SCHEMA",
    "PRIVATE_ANNOTATION_LEDGER_SCHEMA",
    "PRIVATE_EXCEL_PENDING_SCHEMA",
    "build_private_annotation_ledger",
    "materialize_private_annotation_ledger",
    "read_excel_pending_review",
    "select_private_annotation_event",
    "validate_private_annotation_ledger",
]
