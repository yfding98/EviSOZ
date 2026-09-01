#!/usr/bin/env python3
"""Inventory and batch materialization for private long-recording EEG reports.

The command has two deliberately separate trust boundaries:

``build-inventory`` projects only ``edf_path`` and ``patient_id`` from the
historical rows119 CSV (or accepts an equally small explicit source JSON).
``run`` accepts only the resulting strict inventory.  Seizure starts, labels,
EDF annotations, spreadsheets and physician ground truth therefore have no
input route into detection, ranking, findings, language generation or report
rendering.

The full-filesystem route defines one recording/report unit per unique complete
EDF signal SHA-256 (the rows119 compatibility route records its path-based
policy explicitly).  Per-record stages are atomic and resumable, an already
completed report is skipped, zero-candidate recordings still enter the
deterministic report materializer, and optional Qwen wording is never required
for publication.  This command does not launch a run unless the ``run``
subcommand is explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_long_term_clinical_context_v1 import (  # noqa: E402
    build_detector_aligned_frozen_event_registry,
    validate_detector_aligned_frozen_event_registry,
)
from src.clinical_eeg_long_recording.schema import (  # noqa: E402
    validate_long_term_seizure_detection_manifest,
)
from src.clinical_eeg_long_recording.analysis_selection import (  # noqa: E402
    bind_long_term_eeg_analysis_selection,
)
from src.clinical_eeg_long_recording.adaptive_event_analysis_profile import (  # noqa: E402
    ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID,
    validate_materialized_adaptive_event_analysis_profile,
)


INVENTORY_SCHEMA_VERSION = "private_long_recording_inventory_v1"
EXPLICIT_SOURCE_SCHEMA_VERSION = "private_long_recording_inventory_source_v1"
COVERAGE_SCHEMA_VERSION = "private_long_recording_report_coverage_v1"
ADAPTIVE_COVERAGE_SCHEMA_VERSION = "private_long_recording_report_coverage_v2"
STATE_SCHEMA_VERSION = "private_long_recording_report_state_v1"
FAILURE_SCHEMA_VERSION = "private_long_recording_technical_failure_v1"
LOCATOR_LEDGER_SCHEMA_VERSION = "private_edf_subject_locator_ledger_v1"
TECHNICAL_REPORT_SCHEMA_VERSION = "private_long_recording_technical_report_v1"

LEGACY_ANALYSIS_PROFILE_ID = "fixed_v29_report_v1"
ANALYSIS_PROFILE_IDS = {
    LEGACY_ANALYSIS_PROFILE_ID,
    ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID,
}
ADAPTIVE_REPORT_ROUTE_CONNECTED = False

READY = "ready"
UNRESOLVED_EDF = "unresolved_edf"
INVALID_SUBJECT = "invalid_subject_identity"
SUBJECT_CONFLICT = "subject_identity_conflict"
INVENTORY_STATUSES = {READY, UNRESOLVED_EDF, INVALID_SUBJECT, SUBJECT_CONFLICT}
COMPLETED_DIAGNOSTIC_STATUSES = {
    "completed_localizable",
    "completed_nonlocalizable",
    "completed_insufficient_evidence",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORDING_ID_RE = re.compile(r"^PRIV-R[A-Z0-9._-]{1,55}$")
_PATIENT_ID_RE = re.compile(r"^PRIV-P[A-Z0-9._-]{1,55}$")
_SUBJECT_SUFFIX_RE = re.compile(r"(?:[_-]SZ\d+)$", re.IGNORECASE)

_INVENTORY_KEYS = {
    "schema_version",
    "inventory_id",
    "source_kind",
    "source_manifest_sha256",
    "recording_unit_policy",
    "record_count",
    "subject_count",
    "records",
    "source_rejections",
    "field_access_receipt",
}
_RECORD_KEYS = {
    "recording_id",
    "patient_pseudonym",
    "edf_relative_path",
    "source_signal_sha256",
    "source_size_bytes",
    "inventory_validation_status",
    "inventory_error_code",
    "source_row_count",
}
_REJECTION_KEYS = {"rejection_id", "source_row_number", "error_code"}
_LOCATOR_LEDGER_KEYS = {
    "schema_version",
    "ledger_id",
    "frozen_roster_sha256",
    "entries",
    "scope_receipt",
}
_LOCATOR_ENTRY_KEYS = {
    "subject_locator_sha256",
    "patient_pseudonym",
    "mapping_source",
}


def _field_access_receipt(source_kind: str) -> dict[str, Any]:
    projected_by_source = {
        "rows119_path_subject_projection_v1": ["edf_path", "patient_id"],
        "frozen_signal_roster_projection_v1": [
            "relative_edf_path",
            "patient_id",
        ],
        "filesystem_edf_frozen_roster_projection_v1": [
            "filesystem_relative_edf_path",
            "filesystem_signal_sha256",
            "frozen_roster.relative_edf_path",
            "frozen_roster.patient_id",
            "subject_locator_ledger.patient_pseudonym",
        ],
        "explicit_path_subject_projection_v1": [
            "edf_relative_path",
            "patient_pseudonym",
        ],
    }
    if source_kind not in projected_by_source:
        raise ValueError("inventory source_kind is unsupported")
    return {
        "projected_columns": projected_by_source[source_kind],
        "forbidden_columns_projected": [],
        "edf_annotations_loaded": False,
        "excel_or_workbook_loaded": False,
        "onset_or_label_fields_projected": False,
        "ground_truth_loaded": False,
        "raw_patient_id_field_persisted": False,
        "edf_relative_path_persisted_for_io_only": True,
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> object:
    if path.is_symlink():
        raise ValueError(f"JSON input must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"JSON input must be a regular file: {path}")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"JSON contains invalid constant {value!r}")

    return json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )


def _atomic_json(path: Path, value: object, *, replace: bool) -> None:
    target = path.resolve()
    if not replace and (target.exists() or target.is_symlink()):
        raise FileExistsError(target)
    if target.is_symlink():
        raise ValueError(f"refusing to replace symlink: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, target)
        os.chmod(target, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _regular_file(path: Path, *, suffix: str | None = None) -> Path:
    if path.is_symlink():
        raise ValueError(f"input must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"input must be a regular file: {path}")
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise ValueError(f"input must have {suffix} suffix: {path}")
    return resolved


def _dataset_root(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("dataset root must be a regular directory")
    return resolved


def _normalized_edf_relative(value: object, *, allow_suffix_projection: bool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("EDF path is empty")
    text = value.strip().replace("\\", "/")
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("EDF path must be a safe relative path")
    if relative.suffix.lower() != ".edf":
        if not allow_suffix_projection:
            raise ValueError("inventory path must end in .edf")
        relative = relative.with_suffix(".edf")
    return relative.as_posix()


def _resolve_edf(root: Path, relative_text: str) -> Path:
    relative = PurePosixPath(relative_text)
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("EDF path must not traverse a symlink")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("EDF source must be a regular file")
    if resolved.suffix.lower() != ".edf":
        raise ValueError("EDF source suffix drifted")
    return resolved


class EDFDiscontinuousTimelineUnsupportedError(ValueError):
    """The current causal v29 reader cannot map an EDF+D discontinuous clock."""


class EDFFixedHeaderMalformedError(ValueError):
    """Required non-identity EDF fixed-header fields are structurally invalid."""


def _edf_ascii_integer(raw: bytes, *, field: str) -> int:
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise EDFFixedHeaderMalformedError(
            f"EDF fixed-header {field} is not ASCII"
        ) from exc
    if not text or re.fullmatch(r"[+-]?\d+", text) is None:
        raise EDFFixedHeaderMalformedError(
            f"EDF fixed-header {field} is not an integer"
        )
    return int(text)


def _validate_edf_fixed_header_structure(path: Path) -> None:
    """Validate only non-identity EDF/BDF structural fixed-header slots.

    Bytes 8--183 contain patient/recording identifiers and timestamps and are
    deliberately not read.  EDF, EDF+ and BDF share the checked offsets; the
    version field is deliberately not constrained here.
    """

    if path.stat().st_size < 256:
        raise EDFFixedHeaderMalformedError(
            "EDF fixed header is shorter than 256 bytes"
        )
    with path.open("rb") as stream:
        stream.seek(184)
        header_bytes_raw = stream.read(8)
        stream.seek(236)
        structural_tail = stream.read(20)
    if len(header_bytes_raw) != 8 or len(structural_tail) != 20:
        raise EDFFixedHeaderMalformedError(
            "EDF fixed-header structural slots are short"
        )

    header_bytes = _edf_ascii_integer(header_bytes_raw, field="header_bytes")
    data_records = _edf_ascii_integer(
        structural_tail[0:8], field="data_record_count"
    )
    try:
        record_duration = Decimal(structural_tail[8:16].decode("ascii").strip())
    except (UnicodeDecodeError, InvalidOperation) as exc:
        raise EDFFixedHeaderMalformedError(
            "EDF fixed-header data_record_duration is not a finite decimal"
        ) from exc
    signal_count = _edf_ascii_integer(structural_tail[16:20], field="signal_count")

    if data_records < -1:
        raise EDFFixedHeaderMalformedError(
            "EDF fixed-header data_record_count is below the allowed unknown value"
        )
    if not record_duration.is_finite() or record_duration <= 0:
        raise EDFFixedHeaderMalformedError(
            "EDF fixed-header data_record_duration must be positive and finite"
        )
    if signal_count <= 0:
        raise EDFFixedHeaderMalformedError(
            "EDF fixed-header signal_count must be positive"
        )
    expected_header_bytes = 256 * (signal_count + 1)
    if header_bytes != expected_header_bytes or header_bytes > path.stat().st_size:
        raise EDFFixedHeaderMalformedError(
            "EDF fixed-header byte count is inconsistent with signal_count"
        )


def _edf_continuity_mode(path: Path) -> str:
    """Read only the fixed EDF header's reserved continuity marker."""

    with path.open("rb") as stream:
        stream.seek(192)
        reserved = stream.read(44)
    if len(reserved) != 44:
        return "unknown_or_short_header"
    try:
        marker = reserved.decode("ascii").strip()
    except UnicodeDecodeError:
        return "unknown_or_non_ascii_header"
    if marker in {"EDF+C", "EDF+D"}:
        return marker
    return "unspecified_or_nonstandard"


def _subject_base(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    return _SUBJECT_SUFFIX_RE.sub("", normalized).strip()


def _pseudonym(prefix: str, value: str) -> str:
    digest = hashlib.sha256(
        f"private-long-recording-v1\0{prefix}\0{value}".encode("utf-8")
    ).hexdigest()[:20].upper()
    return f"PRIV-{prefix}{digest}"


def _source_rows_from_csv(path: Path) -> tuple[list[tuple[int, str, str]], str]:
    source = _regular_file(path, suffix=".csv")
    rows: list[tuple[int, str, str]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("rows119 source CSV is empty") from exc
        if len(set(header)) != len(header):
            raise ValueError("rows119 source CSV has duplicate column names")
        missing = {"edf_path", "patient_id"}.difference(header)
        if missing:
            raise ValueError(f"rows119 source CSV misses columns: {sorted(missing)}")
        path_index = header.index("edf_path")
        patient_index = header.index("patient_id")
        needed_index = max(path_index, patient_index)
        for source_row_number, raw in enumerate(reader, start=2):
            if len(raw) <= needed_index:
                rows.append((source_row_number, "", ""))
                continue
            # This is the only projection from the historical event manifest.
            # In particular, sz_start, onset_channels, soz_bipolar and all label
            # columns are neither selected nor returned from this boundary.
            rows.append((source_row_number, raw[path_index], raw[patient_index]))
    return rows, _sha256_file(source)


def _source_rows_from_explicit_json(
    path: Path,
) -> tuple[list[tuple[int, str, str]], str]:
    source = _regular_file(path, suffix=".json")
    payload = _strict_json(source)
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "records",
    }:
        raise ValueError("explicit inventory source has missing or unknown keys")
    if payload["schema_version"] != EXPLICIT_SOURCE_SCHEMA_VERSION:
        raise ValueError("explicit inventory source schema drifted")
    raw_records = payload["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("explicit inventory source records must be a non-empty list")
    rows: list[tuple[int, str, str]] = []
    for index, item in enumerate(raw_records, start=1):
        if not isinstance(item, Mapping) or set(item) != {
            "edf_relative_path",
            "patient_pseudonym",
        }:
            raise ValueError(
                f"explicit inventory record {index} has missing or unknown keys"
            )
        rows.append(
            (index, str(item["edf_relative_path"]), str(item["patient_pseudonym"]))
        )
    return rows, _sha256_file(source)


def _source_rows_from_signal_roster(
    path: Path,
) -> tuple[list[tuple[int, str, str]], str]:
    """Project only the frozen pseudonym and EDF locator from a v18 roster."""

    source = _regular_file(path, suffix=".csv")
    rows: list[tuple[int, str, str]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("signal roster is empty") from exc
        if len(set(header)) != len(header):
            raise ValueError("signal roster has duplicate column names")
        missing = {"relative_edf_path", "patient_id"}.difference(header)
        if missing:
            raise ValueError(f"signal roster misses columns: {sorted(missing)}")
        path_index = header.index("relative_edf_path")
        patient_index = header.index("patient_id")
        needed_index = max(path_index, patient_index)
        for source_row_number, raw in enumerate(reader, start=2):
            if len(raw) <= needed_index:
                rows.append((source_row_number, "", ""))
                continue
            rows.append((source_row_number, raw[path_index], raw[patient_index]))
    return rows, _sha256_file(source)


def _locator_sha256(relative_parent: str) -> str:
    return hashlib.sha256(
        f"private-edf-subject-locator-v1\0{relative_parent}".encode("utf-8")
    ).hexdigest()


def _validate_locator_ledger(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _LOCATOR_LEDGER_KEYS:
        raise ValueError("subject locator ledger has missing or unknown keys")
    if value["schema_version"] != LOCATOR_LEDGER_SCHEMA_VERSION:
        raise ValueError("subject locator ledger schema drifted")
    roster_sha = value["frozen_roster_sha256"]
    if not isinstance(roster_sha, str) or _SHA256_RE.fullmatch(roster_sha) is None:
        raise ValueError("subject locator ledger roster SHA-256 is invalid")
    expected_scope = {
        "append_only_patient_assignment": True,
        "raw_subject_names_or_paths_persisted": False,
        "edf_annotations_loaded": False,
        "excel_onset_or_ground_truth_loaded": False,
        "mapping_inputs": [
            "filesystem_subject_locator_hash",
            "frozen_roster_patient_pseudonym",
            "complete_edf_signal_sha256",
        ],
    }
    if value["scope_receipt"] != expected_scope:
        raise ValueError("subject locator ledger scope receipt drifted")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("subject locator ledger entries must be a list")
    entries: list[dict[str, str]] = []
    locator_hashes: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or set(raw) != _LOCATOR_ENTRY_KEYS:
            raise ValueError("subject locator ledger entry has missing or unknown keys")
        locator_hash = raw["subject_locator_sha256"]
        patient = raw["patient_pseudonym"]
        source = raw["mapping_source"]
        if not isinstance(locator_hash, str) or _SHA256_RE.fullmatch(locator_hash) is None:
            raise ValueError("subject locator hash is invalid")
        if locator_hash in locator_hashes:
            raise ValueError("subject locator ledger repeats a locator")
        locator_hashes.add(locator_hash)
        if not isinstance(patient, str) or _PATIENT_ID_RE.fullmatch(patient) is None:
            raise ValueError("subject locator patient pseudonym is invalid")
        if source not in {
            "frozen_roster_parent",
            "signal_sha_match_to_frozen_roster",
            "append_only_new_subject",
        }:
            raise ValueError("subject locator mapping source is unsupported")
        entries.append(
            {
                "subject_locator_sha256": locator_hash,
                "patient_pseudonym": patient,
                "mapping_source": source,
            }
        )
    entries.sort(key=lambda item: item["subject_locator_sha256"])
    body = {
        "schema_version": LOCATOR_LEDGER_SCHEMA_VERSION,
        "frozen_roster_sha256": roster_sha,
        "entries": entries,
        "scope_receipt": expected_scope,
    }
    expected_id = "PLEDGER-" + _canonical_sha256(body)[:24]
    if value["ledger_id"] != expected_id:
        raise ValueError("subject locator ledger ID does not bind content")
    return {
        "schema_version": LOCATOR_LEDGER_SCHEMA_VERSION,
        "ledger_id": expected_id,
        **{key: item for key, item in body.items() if key != "schema_version"},
    }


def _build_locator_ledger(
    *, frozen_roster_sha256: str, entries: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    scope = {
        "append_only_patient_assignment": True,
        "raw_subject_names_or_paths_persisted": False,
        "edf_annotations_loaded": False,
        "excel_onset_or_ground_truth_loaded": False,
        "mapping_inputs": [
            "filesystem_subject_locator_hash",
            "frozen_roster_patient_pseudonym",
            "complete_edf_signal_sha256",
        ],
    }
    normalized_entries = sorted(
        [dict(item) for item in entries],
        key=lambda item: item["subject_locator_sha256"],
    )
    body = {
        "schema_version": LOCATOR_LEDGER_SCHEMA_VERSION,
        "frozen_roster_sha256": frozen_roster_sha256,
        "entries": normalized_entries,
        "scope_receipt": scope,
    }
    ledger = {
        "schema_version": LOCATOR_LEDGER_SCHEMA_VERSION,
        "ledger_id": "PLEDGER-" + _canonical_sha256(body)[:24],
        **{key: item for key, item in body.items() if key != "schema_version"},
    }
    return _validate_locator_ledger(ledger)


def _next_numeric_patient_id(used: set[str]) -> str:
    numbers: list[int] = []
    widths: list[int] = []
    for patient in used:
        match = re.fullmatch(r"PRIV-P(\d+)", patient)
        if match is not None:
            numbers.append(int(match.group(1)))
            widths.append(len(match.group(1)))
    width = max([3, *widths])
    number = max(numbers, default=0) + 1
    candidate = f"PRIV-P{number:0{width}d}"
    while candidate in used:
        number += 1
        candidate = f"PRIV-P{number:0{width}d}"
    return candidate


def build_filesystem_inventory_source(
    *,
    signal_roster_path: Path,
    dataset_root: Path,
    locator_ledger_path: Path,
) -> tuple[list[tuple[int, str, str]], str, dict[str, Any]]:
    """Map every filesystem EDF using only paths, hashes and frozen pseudonyms.

    Known parent locators inherit their roster pseudonym.  An unseen parent may
    inherit an existing pseudonym only when one of its complete EDF SHA-256
    values exactly matches a frozen-roster signal.  Remaining unseen parents
    receive the next numeric pseudonym in an append-only private ledger.
    """

    root = _dataset_root(dataset_root)
    roster_rows, roster_sha = _source_rows_from_signal_roster(signal_roster_path)
    roster_path_patient: dict[str, str] = {}
    roster_parent_patients: dict[str, set[str]] = {}
    for _row_number, raw_path, patient in roster_rows:
        relative = _normalized_edf_relative(raw_path, allow_suffix_projection=False)
        if _PATIENT_ID_RE.fullmatch(patient) is None:
            raise ValueError("frozen signal roster contains a non-pseudonymous patient")
        existing = roster_path_patient.setdefault(relative, patient)
        if existing != patient:
            raise ValueError("frozen signal roster path has conflicting patients")
        parent = PurePosixPath(relative).parent.as_posix()
        roster_parent_patients.setdefault(parent, set()).add(patient)
    if any(len(patients) != 1 for patients in roster_parent_patients.values()):
        raise ValueError("frozen signal roster parent has conflicting patients")

    filesystem_paths: list[str] = []
    signal_sha_by_path: dict[str, str] = {}
    paths_by_parent: dict[str, list[str]] = {}
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if candidate.suffix.lower() != ".edf":
            continue
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        relative = resolved.relative_to(root).as_posix()
        # Reuse the strict no-symlink path walk before hashing.
        verified = _resolve_edf(root, relative)
        signal_sha_by_path[relative] = _sha256_file(verified)
        filesystem_paths.append(relative)
        paths_by_parent.setdefault(PurePosixPath(relative).parent.as_posix(), []).append(
            relative
        )
    if not filesystem_paths:
        raise ValueError("dataset root contains no regular EDF files")
    missing_roster_paths = sorted(set(roster_path_patient).difference(filesystem_paths))
    if missing_roster_paths:
        raise ValueError("frozen signal roster references absent filesystem EDFs")

    frozen_patients_by_signal: dict[str, set[str]] = {}
    for relative, patient in roster_path_patient.items():
        frozen_patients_by_signal.setdefault(signal_sha_by_path[relative], set()).add(
            patient
        )
    if any(len(patients) != 1 for patients in frozen_patients_by_signal.values()):
        raise ValueError("one frozen signal SHA-256 maps to conflicting patients")

    existing_ledger: dict[str, Any] | None = None
    if locator_ledger_path.exists() or locator_ledger_path.is_symlink():
        existing_ledger = _validate_locator_ledger(
            _strict_json(locator_ledger_path)
        )
        if existing_ledger["frozen_roster_sha256"] != roster_sha:
            raise ValueError("locator ledger was built from a different frozen roster")
    existing_by_locator = {
        item["subject_locator_sha256"]: item
        for item in (existing_ledger or {}).get("entries", [])
    }
    entries: dict[str, dict[str, str]] = {
        key: dict(item) for key, item in existing_by_locator.items()
    }
    used_patients = set(roster_path_patient.values()) | {
        str(item["patient_pseudonym"]) for item in entries.values()
    }
    parent_assignment: dict[str, str] = {}
    unresolved_parents: list[tuple[str, str]] = []
    for parent, parent_paths in paths_by_parent.items():
        locator_hash = _locator_sha256(parent)
        expected_patient: str | None = None
        expected_source: str | None = None
        roster_patients = roster_parent_patients.get(parent, set())
        if len(roster_patients) == 1:
            expected_patient = next(iter(roster_patients))
            expected_source = "frozen_roster_parent"
        else:
            matched_patients: set[str] = set()
            for relative in parent_paths:
                matched_patients.update(
                    frozen_patients_by_signal.get(signal_sha_by_path[relative], set())
                )
            if len(matched_patients) > 1:
                raise ValueError("unseen filesystem parent matches multiple patients")
            if len(matched_patients) == 1:
                expected_patient = next(iter(matched_patients))
                expected_source = "signal_sha_match_to_frozen_roster"
        previous = entries.get(locator_hash)
        if previous is not None:
            if expected_patient is not None and previous["patient_pseudonym"] != expected_patient:
                raise ValueError("append-only locator mapping conflicts with frozen evidence")
            patient = str(previous["patient_pseudonym"])
            parent_assignment[parent] = patient
            continue
        if expected_patient is None:
            unresolved_parents.append((locator_hash, parent))
            continue
        entries[locator_hash] = {
            "subject_locator_sha256": locator_hash,
            "patient_pseudonym": expected_patient,
            "mapping_source": str(expected_source),
        }
        parent_assignment[parent] = expected_patient

    # Hash ordering is deterministic without persisting or publishing names;
    # once assigned, the ledger freezes all prior mappings append-only.
    for locator_hash, parent in sorted(unresolved_parents):
        patient = _next_numeric_patient_id(used_patients)
        used_patients.add(patient)
        entries[locator_hash] = {
            "subject_locator_sha256": locator_hash,
            "patient_pseudonym": patient,
            "mapping_source": "append_only_new_subject",
        }
        parent_assignment[parent] = patient

    ledger = _build_locator_ledger(
        frozen_roster_sha256=roster_sha,
        entries=list(entries.values()),
    )
    _atomic_json(locator_ledger_path, ledger, replace=existing_ledger is not None)
    source_rows = [
        (
            index,
            relative,
            parent_assignment[PurePosixPath(relative).parent.as_posix()],
        )
        for index, relative in enumerate(sorted(filesystem_paths), start=1)
    ]
    filesystem_binding = [
        {
            "relative_path_sha256": hashlib.sha256(relative.encode("utf-8")).hexdigest(),
            "signal_sha256": signal_sha_by_path[relative],
        }
        for relative in sorted(filesystem_paths)
    ]
    composite_source_sha = _canonical_sha256(
        {
            "frozen_roster_sha256": roster_sha,
            "locator_ledger_id": ledger["ledger_id"],
            "filesystem_bindings": filesystem_binding,
        }
    )
    return source_rows, composite_source_sha, ledger


def build_inventory(
    *,
    source_rows: Sequence[tuple[int, str, str]],
    source_kind: str,
    source_manifest_sha256: str,
    dataset_root: Path,
    allow_suffix_projection: bool,
    subject_values_are_pseudonyms: bool = False,
    recording_unit_policy: str = "unique_edf_path_v1",
) -> dict[str, Any]:
    """Build a pseudonymous, unique-EDF inventory from two projected fields."""

    if recording_unit_policy not in {
        "unique_edf_path_v1",
        "unique_signal_sha256_v1",
    }:
        raise ValueError("recording_unit_policy is unsupported")
    root = _dataset_root(dataset_root)
    grouped: dict[str, dict[str, Any]] = {}
    rejections: list[dict[str, Any]] = []
    for source_row_number, raw_path, raw_subject in source_rows:
        try:
            relative = _normalized_edf_relative(
                raw_path, allow_suffix_projection=allow_suffix_projection
            )
        except (TypeError, ValueError):
            rejection = {
                "rejection_id": "INVREJ-"
                + _canonical_sha256(
                    {"source_row_number": source_row_number, "error": "invalid_edf_path"}
                )[:20],
                "source_row_number": int(source_row_number),
                "error_code": "invalid_edf_path",
            }
            rejections.append(rejection)
            continue
        bucket = grouped.setdefault(relative, {"subjects": set(), "rows": 0})
        subject = _subject_base(raw_subject)
        if subject:
            bucket["subjects"].add(subject)
        bucket["rows"] += 1

    records: list[dict[str, Any]] = []
    for relative, bucket in sorted(grouped.items()):
        subjects = sorted(bucket["subjects"])
        valid_single_pseudonym = (
            len(subjects) == 1
            and isinstance(subjects[0], str)
            and _PATIENT_ID_RE.fullmatch(subjects[0]) is not None
        )
        if len(subjects) == 1 and (
            not subject_values_are_pseudonyms or valid_single_pseudonym
        ):
            patient = (
                subjects[0]
                if subject_values_are_pseudonyms
                else _pseudonym("P", subjects[0])
            )
            identity_status = READY
            identity_error = None
        elif not subjects or (subject_values_are_pseudonyms and len(subjects) == 1):
            patient = _pseudonym("P", f"unresolved\0{relative}")
            identity_status = INVALID_SUBJECT
            identity_error = INVALID_SUBJECT
        else:
            patient = _pseudonym("P", f"conflict\0{relative}")
            identity_status = SUBJECT_CONFLICT
            identity_error = SUBJECT_CONFLICT

        source_hash: str | None = None
        source_size: int | None = None
        status = identity_status
        error_code = identity_error
        try:
            edf = _resolve_edf(root, relative)
            source_hash = _sha256_file(edf)
            source_size = int(edf.stat().st_size)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            if status == READY:
                status = UNRESOLVED_EDF
                error_code = UNRESOLVED_EDF
        recording_id = (
            "PRIV-RH" + str(source_hash)[:16].upper()
            if recording_unit_policy == "unique_signal_sha256_v1"
            and source_hash is not None
            else _pseudonym("R", relative)
        )
        records.append(
            {
                "recording_id": recording_id,
                "patient_pseudonym": patient,
                "edf_relative_path": relative,
                "source_signal_sha256": source_hash,
                "source_size_bytes": source_size,
                "inventory_validation_status": status,
                "inventory_error_code": error_code,
                "source_row_count": int(bucket["rows"]),
            }
        )

    if recording_unit_policy == "unique_signal_sha256_v1":
        by_signal: dict[str, list[dict[str, Any]]] = {}
        unresolved: list[dict[str, Any]] = []
        for record in records:
            signal_sha = record["source_signal_sha256"]
            if signal_sha is None:
                unresolved.append(record)
            else:
                by_signal.setdefault(str(signal_sha), []).append(record)
        collapsed: list[dict[str, Any]] = []
        for signal_sha, duplicates in sorted(by_signal.items()):
            duplicates.sort(key=lambda item: str(item["edf_relative_path"]))
            representative = dict(duplicates[0])
            patients = {str(item["patient_pseudonym"]) for item in duplicates}
            statuses = {str(item["inventory_validation_status"]) for item in duplicates}
            representative["source_row_count"] = sum(
                int(item["source_row_count"]) for item in duplicates
            )
            representative["recording_id"] = "PRIV-RH" + signal_sha[:16].upper()
            if len(patients) != 1 or statuses != {READY}:
                representative["patient_pseudonym"] = _pseudonym(
                    "P", f"signal-conflict\0{signal_sha}"
                )
                representative["inventory_validation_status"] = SUBJECT_CONFLICT
                representative["inventory_error_code"] = SUBJECT_CONFLICT
            collapsed.append(representative)
        records = sorted(
            [*collapsed, *unresolved], key=lambda item: str(item["recording_id"])
        )

    body = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_manifest_sha256": source_manifest_sha256,
        "recording_unit_policy": recording_unit_policy,
        "record_count": len(records),
        "subject_count": len({item["patient_pseudonym"] for item in records}),
        "records": records,
        "source_rejections": rejections,
        "field_access_receipt": _field_access_receipt(source_kind),
    }
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "inventory_id": "PLINV-" + _canonical_sha256(body)[:24],
        **{key: value for key, value in body.items() if key != "schema_version"},
    }
    return validate_inventory(inventory)


def validate_inventory(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _INVENTORY_KEYS:
        raise ValueError("private inventory has missing or unknown keys")
    if value["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise ValueError("private inventory schema drifted")
    if value["source_kind"] not in {
        "rows119_path_subject_projection_v1",
        "frozen_signal_roster_projection_v1",
        "filesystem_edf_frozen_roster_projection_v1",
        "explicit_path_subject_projection_v1",
    }:
        raise ValueError("private inventory source_kind is unsupported")
    source_sha = value["source_manifest_sha256"]
    if not isinstance(source_sha, str) or _SHA256_RE.fullmatch(source_sha) is None:
        raise ValueError("inventory source manifest SHA-256 is invalid")
    recording_unit_policy = value["recording_unit_policy"]
    if recording_unit_policy not in {
        "unique_edf_path_v1",
        "unique_signal_sha256_v1",
    }:
        raise ValueError("inventory recording unit policy is unsupported")
    expected_field_receipt = _field_access_receipt(str(value["source_kind"]))
    if value["field_access_receipt"] != expected_field_receipt:
        raise ValueError("inventory field-access firewall drifted")
    raw_records = value["records"]
    if not isinstance(raw_records, list):
        raise TypeError("inventory records must be a list")
    records: list[dict[str, Any]] = []
    recording_ids: set[str] = set()
    relative_paths: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping) or set(raw) != _RECORD_KEYS:
            raise ValueError(f"inventory record {index} has missing or unknown keys")
        recording_id = raw["recording_id"]
        patient = raw["patient_pseudonym"]
        if not isinstance(recording_id, str) or _RECORDING_ID_RE.fullmatch(
            recording_id
        ) is None:
            raise ValueError("inventory recording_id is invalid")
        if not isinstance(patient, str) or _PATIENT_ID_RE.fullmatch(patient) is None:
            raise ValueError("inventory patient_pseudonym is invalid")
        relative = _normalized_edf_relative(
            raw["edf_relative_path"], allow_suffix_projection=False
        )
        if recording_id in recording_ids or relative in relative_paths:
            raise ValueError("inventory repeats a recording ID or EDF path")
        recording_ids.add(recording_id)
        relative_paths.add(relative)
        status = raw["inventory_validation_status"]
        if status not in INVENTORY_STATUSES:
            raise ValueError("inventory record status is unsupported")
        signal_sha = raw["source_signal_sha256"]
        size = raw["source_size_bytes"]
        error_code = raw["inventory_error_code"]
        if status == READY:
            if not isinstance(signal_sha, str) or _SHA256_RE.fullmatch(signal_sha) is None:
                raise ValueError("ready inventory record lacks a signal SHA-256")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("ready inventory record lacks a positive file size")
            if error_code is not None:
                raise ValueError("ready inventory record carries an error code")
        else:
            if error_code != status:
                raise ValueError("invalid inventory record error/status mismatch")
            if signal_sha is not None and (
                not isinstance(signal_sha, str) or _SHA256_RE.fullmatch(signal_sha) is None
            ):
                raise ValueError("inventory signal SHA-256 is invalid")
            if size is not None and (
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
            ):
                raise ValueError("inventory source size is invalid")
        row_count = raw["source_row_count"]
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
            raise ValueError("inventory source_row_count must be positive")
        records.append(dict(raw))
    if value["record_count"] != len(records):
        raise ValueError("inventory record_count does not match records")
    subjects = {item["patient_pseudonym"] for item in records}
    if value["subject_count"] != len(subjects):
        raise ValueError("inventory subject_count does not match records")
    rejections = value["source_rejections"]
    if not isinstance(rejections, list):
        raise TypeError("inventory source_rejections must be a list")
    for raw in rejections:
        if not isinstance(raw, Mapping) or set(raw) != _REJECTION_KEYS:
            raise ValueError("inventory rejection has missing or unknown keys")
        if not isinstance(raw["rejection_id"], str):
            raise TypeError("inventory rejection_id must be text")
        if (
            isinstance(raw["source_row_number"], bool)
            or not isinstance(raw["source_row_number"], int)
            or raw["source_row_number"] < 1
        ):
            raise ValueError("inventory rejection row number is invalid")
        if raw["error_code"] != "invalid_edf_path":
            raise ValueError("inventory rejection error code is unsupported")
    normalized = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "source_kind": value["source_kind"],
        "source_manifest_sha256": source_sha,
        "recording_unit_policy": recording_unit_policy,
        "record_count": len(records),
        "subject_count": len(subjects),
        "records": records,
        "source_rejections": [dict(item) for item in rejections],
        "field_access_receipt": expected_field_receipt,
    }
    expected_id = "PLINV-" + _canonical_sha256(normalized)[:24]
    if value["inventory_id"] != expected_id:
        raise ValueError("inventory_id does not bind inventory content")
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "inventory_id": expected_id,
        **{key: item for key, item in normalized.items() if key != "schema_version"},
    }


class StageExecutionError(RuntimeError):
    def __init__(
        self,
        stage: str,
        *,
        returncode: int,
        stdout_sha256: str,
        stderr_sha256: str,
    ) -> None:
        super().__init__(f"{stage} subprocess failed with return code {returncode}")
        self.stage = stage
        self.returncode = int(returncode)
        self.stdout_sha256 = stdout_sha256
        self.stderr_sha256 = stderr_sha256


class CliStageExecutor:
    """Invoke the already-audited single-record CLIs through ``rtk``."""

    def __init__(
        self,
        *,
        python_executable: Path,
        device: str,
        base_url: str,
        use_qwen: bool,
    ) -> None:
        rtk = shutil.which("rtk")
        if rtk is None:
            raise FileNotFoundError("rtk is required for private batch stages")
        self.rtk = rtk
        self.python = str(python_executable.resolve(strict=True))
        self.device = device
        self.base_url = base_url
        self.use_qwen = use_qwen
        cache_dir = Path(tempfile.gettempdir()) / "eeg_seizure_numba_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(cache_dir, 0o700)
        matplotlib_cache_dir = (
            Path(tempfile.gettempdir()) / "eeg_seizure_matplotlib_cache"
        )
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(matplotlib_cache_dir, 0o700)
        self.environment = dict(os.environ)
        self.environment.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
        self.environment.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    def _run(self, stage: str, script: str, arguments: Sequence[str]) -> None:
        command = [self.rtk, self.python, str(ROOT / "scripts" / script), *arguments]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=self.environment,
        )
        if completed.returncode != 0:
            raise StageExecutionError(
                stage,
                returncode=completed.returncode,
                stdout_sha256=hashlib.sha256(completed.stdout).hexdigest(),
                stderr_sha256=hashlib.sha256(completed.stderr).hexdigest(),
            )

    def scan(self, *, record: Mapping[str, Any], edf: Path, output: Path) -> None:
        self._run(
            "coarse_scan",
            "scan_long_recording_transition_review_v1.py",
            [
                "--edf",
                str(edf),
                "--recording-id",
                str(record["recording_id"]),
                "--patient-pseudonym",
                str(record["patient_pseudonym"]),
                "--source-signal-sha256",
                str(record["source_signal_sha256"]),
                "--output",
                str(output),
            ],
        )

    def adaptive_event_analysis(
        self,
        *,
        edf: Path,
        detection: Path,
        output: Path,
    ) -> None:
        self._run(
            "adaptive_event_analysis_v2",
            "materialize_adaptive_event_analysis_v2.py",
            [
                "--edf",
                str(edf),
                "--detection-manifest",
                str(detection),
                "--output",
                str(output),
            ],
        )

    def rank(
        self,
        *,
        edf: Path,
        detection: Path,
        registry: Path,
        output: Path,
    ) -> None:
        self._run(
            "v29_ranking",
            "materialize_v29_long_recording_rankings.py",
            [
                "--recording-edf",
                str(edf),
                "--detection-manifest",
                str(detection),
                "--event-registry",
                str(registry),
                "--output",
                str(output),
                "--device",
                self.device,
            ],
        )

    def materialize_events(
        self,
        *,
        edf: Path,
        detection: Path,
        registry: Path,
        ranking_manifest: Path,
        output: Path,
        analysis_selection: Path | None = None,
    ) -> None:
        arguments = [
            "--recording-edf",
            str(edf),
            "--detection-manifest",
            str(detection),
            "--event-registry",
            str(registry),
            "--ranking-manifest",
            str(ranking_manifest),
            "--output",
            str(output),
        ]
        if analysis_selection is not None:
            arguments.extend(
                ["--analysis-selection", str(analysis_selection)]
            )
        self._run(
            "event_materialization",
            "materialize_long_term_event_segments_v1.py",
            arguments,
        )

    def report(
        self,
        *,
        detection: Path,
        segment_receipts: Sequence[Path],
        waveform_root: Path,
        bundle_id: str,
        output: Path,
        analysis_selection: Path | None = None,
    ) -> None:
        arguments = [
            "--detection-manifest",
            str(detection),
            "--waveform-root",
            str(waveform_root),
            "--bundle-id",
            bundle_id,
            "--output",
            str(output),
            "--base-url",
            self.base_url,
        ]
        for receipt in segment_receipts:
            arguments.extend(["--segment-receipt", str(receipt)])
        if analysis_selection is not None:
            arguments.extend(
                ["--analysis-selection", str(analysis_selection)]
            )
        if self.use_qwen:
            arguments.append("--use-qwen")
        self._run(
            "report_materialization",
            "materialize_trustworthy_long_term_clinical_eeg_report_v1.py",
            arguments,
        )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _strict_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(value)


def _validate_detection_for_record(
    path: Path, record: Mapping[str, Any]
) -> dict[str, Any]:
    value = validate_long_term_seizure_detection_manifest(_read_json_object(path))
    exact = {
        "recording_id": record["recording_id"],
        "patient_pseudonym": record["patient_pseudonym"],
        "source_signal_sha256": record["source_signal_sha256"],
    }
    if any(value[key] != expected for key, expected in exact.items()):
        raise ValueError("detection manifest identity differs from inventory")
    receipt = value["detector_receipt"]
    if receipt.get("annotations_used") is not False or receipt.get("labels_used") is not False:
        raise ValueError("detector receipt did not preserve the EEG-only boundary")
    return value


def _validate_adaptive_profile_for_record(
    path: Path, record: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = validate_materialized_adaptive_event_analysis_profile(path)
    exact = {
        "recording_id": record["recording_id"],
        "patient_pseudonym": record["patient_pseudonym"],
        "source_signal_sha256": record["source_signal_sha256"],
    }
    if any(manifest[key] != expected for key, expected in exact.items()):
        raise ValueError("adaptive event analysis identity differs from inventory")
    scope = manifest["scope_receipt"]
    if (
        scope["eeg_signal_only"] is not True
        or scope["edf_annotation_api_called"] is not False
        or scope["excel_used"] is not False
        or scope["labels_or_ground_truth_used"] is not False
        or scope["primary_findings_window_is_adaptive"] is not True
        or scope["fixed_crop_role"] != "compatibility_core_only"
    ):
        raise ValueError("adaptive event analysis scope/profile drifted")
    return manifest


def _selected_count(detection: Mapping[str, Any]) -> int:
    return sum(
        item.get("decision") == "selected_for_event_analysis"
        for item in detection["merge_candidates"]
    )


def _event_receipts(events_dir: Path, record: Mapping[str, Any]) -> list[Path]:
    manifest = _read_json_object(events_dir / "manifest.json")
    if manifest.get("recording_id") != record["recording_id"]:
        raise ValueError("event materialization recording identity drifted")
    if manifest.get("patient_pseudonym") != record["patient_pseudonym"]:
        raise ValueError("event materialization patient identity drifted")
    raw = manifest.get("segment_receipts")
    if not isinstance(raw, list) or manifest.get("event_count") != len(raw):
        raise ValueError("event materialization receipt count drifted")
    receipts: list[Path] = []
    root = events_dir.resolve(strict=True)
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("event segment receipt entry must be an object")
        relative_text = item.get("segment_receipt_file")
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError("event segment receipt path is invalid")
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("event segment receipt path is unsafe")
        candidate = root.joinpath(*relative.parts).resolve(strict=True)
        candidate.relative_to(root)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("event segment receipt is not a regular file")
        if _sha256_file(candidate) != item.get("segment_receipt_sha256"):
            raise ValueError("event segment receipt SHA-256 differs")
        receipts.append(candidate)
    return receipts


def _completed_report(
    report_dir: Path, record: Mapping[str, Any]
) -> dict[str, Any] | None:
    manifest_path = report_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return None
    value = _read_json_object(manifest_path)
    if value.get("status") != "completed_unsigned_ai_draft":
        return None
    if value.get("recording_id") != record["recording_id"]:
        raise ValueError("existing report recording identity drifted")
    if value.get("patient_pseudonym") != record["patient_pseudonym"]:
        raise ValueError("existing report patient identity drifted")
    diagnostic = value.get("diagnostic_status")
    if diagnostic not in COMPLETED_DIAGNOSTIC_STATUSES:
        raise ValueError("existing report diagnostic status is not completed")
    scope = value.get("scope_receipt")
    if not isinstance(scope, Mapping) or any(
        scope.get(key) is not False
        for key in (
            "external_edf_annotations_loaded",
            "excel_observations_loaded",
            "research_soz_used_in_clinical_facts_or_llm",
        )
    ):
        raise ValueError("existing report violates the EEG-only boundary")
    return value


def _state(
    record: Mapping[str, Any],
    *,
    status: str,
    last_completed_stage: str | None,
    diagnostic_status: str | None,
    event_count: int | None,
    attempt: int,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "recording_id": record["recording_id"],
        "patient_pseudonym": record["patient_pseudonym"],
        "status": status,
        "last_completed_stage": last_completed_stage,
        "diagnostic_status": diagnostic_status,
        "event_count": event_count,
        "attempt": attempt,
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotations_loaded": False,
            "excel_loaded": False,
            "onset_or_ground_truth_loaded": False,
        },
    }


def _attempt_number(path: Path) -> int:
    if not path.is_file() or path.is_symlink():
        return 1
    try:
        value = _read_json_object(path)
        previous = value.get("attempt")
        if isinstance(previous, int) and not isinstance(previous, bool) and previous >= 1:
            return previous + 1
    except (OSError, TypeError, ValueError):
        pass
    return 1


def _failure_receipt(
    record: Mapping[str, Any],
    *,
    failed_stage: str,
    error_code: str,
    error: BaseException,
    attempt: int,
) -> dict[str, Any]:
    exception_class = type(error).__name__
    fingerprint_source: dict[str, Any] = {
        "stage": failed_stage,
        "error_code": error_code,
        "exception_class": exception_class,
    }
    subprocess_receipt: dict[str, Any] | None = None
    if isinstance(error, StageExecutionError):
        subprocess_receipt = {
            "returncode": error.returncode,
            "stdout_sha256": error.stdout_sha256,
            "stderr_sha256": error.stderr_sha256,
            "stdout_or_stderr_persisted": False,
        }
        fingerprint_source["subprocess"] = subprocess_receipt
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "technical_failure_receipt",
        "recording_id": record["recording_id"],
        "patient_pseudonym": record["patient_pseudonym"],
        "failed_stage": failed_stage,
        "error_code": error_code,
        "exception_class": exception_class,
        "error_fingerprint": _canonical_sha256(fingerprint_source),
        "attempt": attempt,
        "retryable": True,
        "report_generated": False,
        "subprocess_receipt": subprocess_receipt,
        "privacy_receipt": {
            "exception_message_persisted": False,
            "raw_edf_path_persisted": False,
            "raw_patient_identity_persisted": False,
            "annotation_excel_onset_or_gt_persisted": False,
        },
    }


def _materialize_technical_unassessable_report(
    *,
    case_dir: Path,
    record: Mapping[str, Any],
    failure_receipt: Mapping[str, Any],
    attempt: int,
) -> Path:
    """Atomically publish a non-clinical report shell for a technical failure."""

    parent = case_dir / "technical_reports"
    parent.mkdir(parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    target = parent / f"attempt_{attempt:04d}"
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    published = False
    try:
        conclusion = (
            "本次流程未成功读取或完成 EEG 信号分析，因而没有形成可用于 SOZ "
            "判断的 EEG 证据；SOZ 无法判断。本结果仅表示技术不可评估，不能据此"
            "作任何脑电诊断或定位推断。"
        )
        report = {
            "schema_version": TECHNICAL_REPORT_SCHEMA_VERSION,
            "status": "completed_technical_unassessable",
            "recording_id": record["recording_id"],
            "patient_pseudonym": record["patient_pseudonym"],
            "failure_stage": failure_receipt["failed_stage"],
            "technical_failure_receipt_fingerprint": failure_receipt[
                "error_fingerprint"
            ],
            "eeg_analysis_completed": False,
            "diagnostic_status": "completed_technical_unassessable",
            "soz_conclusion_code": "soz_unassessable_technical_failure",
            "conclusion_zh": conclusion,
            "claim_boundary": {
                "normal_or_negative_eeg_claimed": False,
                "diffuse_or_bilateral_onset_claimed": False,
                "focal_soz_claimed": False,
                "clinical_diagnosis_generated": False,
                "physician_signed": False,
            },
            "scope_receipt": {
                "edf_annotations_loaded": False,
                "excel_or_workbook_loaded": False,
                "onset_label_or_ground_truth_loaded": False,
                "exception_message_or_raw_path_persisted": False,
            },
        }
        report_path = staging / "report.json"
        _atomic_json(report_path, report, replace=False)
        html_text = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>长程 EEG 技术不可评估报告</title></head><body>
<h1>长程 EEG 实验报告（技术不可评估）</h1>
<p>记录编号：{recording}</p><p>被试伪名：{patient}</p>
<h2>脑电图印象</h2><p>{conclusion}</p>
<h2>范围说明</h2><p>未使用 EDF annotation、Excel、onset、标签或医生 GT。</p>
<p>本报告为未签名研究用 AI 技术状态记录，不能替代医生审核。</p>
</body></html>
""".format(
            recording=html.escape(str(record["recording_id"])),
            patient=html.escape(str(record["patient_pseudonym"])),
            conclusion=html.escape(conclusion),
        )
        html_path = staging / "report.html"
        html_path.write_text(html_text, encoding="utf-8")
        os.chmod(html_path, 0o600)
        manifest = {
            "schema_version": TECHNICAL_REPORT_SCHEMA_VERSION,
            "status": "completed_technical_unassessable",
            "recording_id": record["recording_id"],
            "patient_pseudonym": record["patient_pseudonym"],
            "diagnostic_status": "completed_technical_unassessable",
            "failure_stage": failure_receipt["failed_stage"],
            "event_count": 0,
            "artifacts": {
                "report.json": _sha256_file(report_path),
                "report.html": _sha256_file(html_path),
            },
            "technical_failure_receipt_fingerprint": failure_receipt[
                "error_fingerprint"
            ],
        }
        _atomic_json(staging / "manifest.json", manifest, replace=False)
        for path in staging.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        os.chmod(staging, 0o700)
        os.replace(staging, target)
        os.chmod(target, 0o700)
        published = True
        return target
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _case_result(
    record: Mapping[str, Any],
    *,
    run_status: str,
    diagnostic_status: str | None = None,
    event_count: int | None = None,
    failure_stage: str | None = None,
    reused: bool = False,
    technical_artifact_relative_dir: str | None = None,
) -> dict[str, Any]:
    return {
        "recording_id": record["recording_id"],
        "patient_pseudonym": record["patient_pseudonym"],
        "inventory_validation_status": record["inventory_validation_status"],
        "run_status": run_status,
        "diagnostic_status": diagnostic_status,
        "event_count": event_count,
        "failure_stage": failure_stage,
        "existing_success_reused": reused,
        "technical_artifact_relative_dir": technical_artifact_relative_dir,
    }


def _write_coverage(
    *,
    output: Path,
    inventory: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    mode: str,
    qwen_requested: bool,
    analysis_profile: str = LEGACY_ANALYSIS_PROFILE_ID,
) -> dict[str, Any]:
    by_subject: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(str(row["patient_pseudonym"]), []).append(row)
    subjects: list[dict[str, Any]] = []
    for patient, patient_rows in sorted(by_subject.items()):
        completed_eeg = sum(
            row["diagnostic_status"] in COMPLETED_DIAGNOSTIC_STATUSES
            for row in patient_rows
        )
        technical_reports = sum(
            row["diagnostic_status"] == "completed_technical_unassessable"
            for row in patient_rows
        )
        failures = sum(row["failure_stage"] is not None for row in patient_rows)
        completed_artifacts = completed_eeg + technical_reports
        subjects.append(
            {
                "patient_pseudonym": patient,
                "expected_record_count": len(patient_rows),
                "completed_report_artifact_count": completed_artifacts,
                "completed_eeg_report_count": completed_eeg,
                "technical_unassessable_report_count": technical_reports,
                "technical_failure_count": failures,
                "has_at_least_one_report_artifact": completed_artifacts > 0,
                "has_at_least_one_completed_eeg_report": completed_eeg > 0,
                "coverage_complete": completed_artifacts == len(patient_rows),
                "eeg_coverage_complete": completed_eeg == len(patient_rows),
            }
        )
    completed_eeg = sum(
        row["diagnostic_status"] in COMPLETED_DIAGNOSTIC_STATUSES for row in rows
    )
    technical_reports = sum(
        row["diagnostic_status"] == "completed_technical_unassessable" for row in rows
    )
    failures = sum(row["failure_stage"] is not None for row in rows)
    completed_artifacts = completed_eeg + technical_reports
    coverage = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "inventory_id": inventory["inventory_id"],
        "recording_unit_policy": inventory["recording_unit_policy"],
        "mode": mode,
        "expected_record_count": inventory["record_count"],
        "expected_subject_count": inventory["subject_count"],
        "inventory_rejection_count": len(inventory["source_rejections"]),
        "completed_report_count": completed_artifacts,
        "completed_report_artifact_count": completed_artifacts,
        "completed_eeg_report_count": completed_eeg,
        "technical_unassessable_report_count": technical_reports,
        "technical_failure_count": failures,
        "pending_or_not_run_count": len(rows) - completed_artifacts,
        "dataset_coverage_complete": (
            completed_artifacts == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
        "dataset_artifact_coverage_complete": (
            completed_artifacts == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
        "dataset_eeg_coverage_complete": (
            completed_eeg == inventory["record_count"]
            and not inventory["source_rejections"]
        ),
        "diagnostic_status_counts": {
            status: sum(row["diagnostic_status"] == status for row in rows)
            for status in sorted(
                {*COMPLETED_DIAGNOSTIC_STATUSES, "completed_technical_unassessable"}
            )
        },
        "records": [dict(row) for row in rows],
        "subjects": subjects,
        "scope_receipt": {
            "one_report_unit_per_inventory_recording_unit": True,
            "recording_unit_policy": inventory["recording_unit_policy"],
            "source_event_rows_deduplicated_before_inference": True,
            "generation_uses_eeg_signal_only": True,
            "edf_annotations_loaded": False,
            "excel_or_workbook_loaded": False,
            "onset_or_label_fields_forwarded": False,
            "ground_truth_forwarded": False,
            "qwen_optional": True,
            "qwen_requested": qwen_requested,
            "qwen_failure_blocks_report": False,
            "zero_candidates_still_materialize_report": True,
            "findings_insufficiency_is_completed_abstention": True,
            "technical_failure_is_not_eeg_insufficiency": True,
            "technical_failure_gets_non_diagnostic_report_shell": True,
            "raw_edf_paths_or_patient_names_in_coverage": False,
        },
    }
    if analysis_profile == ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID:
        coverage["schema_version"] = ADAPTIVE_COVERAGE_SCHEMA_VERSION
        coverage["analysis_profile"] = ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID
        coverage["analysis_route_status"] = (
            "blocked_primary_findings_route_not_connected"
        )
        coverage["adaptive_event_analysis_relative_path"] = (
            "records/<recording_id>/adaptive_event_analysis_v2"
        )
        coverage["scope_receipt"]["requested_primary_findings_window"] = (
            "adaptive_variable_per_event"
        )
        coverage["scope_receipt"]["adaptive_primary_findings_connected_to_report"] = (
            False
        )
        coverage["scope_receipt"]["effective_report_findings_window"] = None
        coverage["scope_receipt"]["adaptive_artifact_fixed_crop_role"] = (
            "compatibility_core_only"
        )
        coverage["scope_receipt"]["execution_allowed"] = False
    elif analysis_profile != LEGACY_ANALYSIS_PROFILE_ID:
        raise ValueError("private batch analysis profile is unsupported")
    _atomic_json(output, coverage, replace=True)
    return coverage


def _run_one(
    *,
    record: Mapping[str, Any],
    dataset_root: Path,
    records_root: Path,
    executor: Any,
    analysis_profile: str,
) -> dict[str, Any]:
    if analysis_profile not in ANALYSIS_PROFILE_IDS:
        raise ValueError("private batch analysis profile is unsupported")
    case_dir = records_root / str(record["recording_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(case_dir, 0o700)
    state_path = case_dir / "state.json"
    attempt = _attempt_number(state_path)
    stage = "input_validation"
    try:
        existing = _completed_report(case_dir / "report", record)
        adaptive_existing = None
        adaptive_dir = case_dir / "adaptive_event_analysis_v2"
        if (
            analysis_profile == ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID
            and (adaptive_dir / "manifest.json").is_file()
        ):
            adaptive_existing = _validate_adaptive_profile_for_record(
                adaptive_dir, record
            )
        if existing is not None and (
            analysis_profile == LEGACY_ANALYSIS_PROFILE_ID
            or adaptive_existing is not None
        ):
            _atomic_json(
                state_path,
                _state(
                    record,
                    status="completed",
                    last_completed_stage="report_materialization",
                    diagnostic_status=str(existing["diagnostic_status"]),
                    event_count=int(existing["event_count"]),
                    attempt=attempt,
                ),
                replace=True,
            )
            return _case_result(
                record,
                run_status="completed_existing",
                diagnostic_status=str(existing["diagnostic_status"]),
                event_count=int(existing["event_count"]),
                reused=True,
            )
        if record["inventory_validation_status"] != READY:
            raise ValueError("inventory record is not ready for EEG inference")
        edf = _resolve_edf(dataset_root, str(record["edf_relative_path"]))
        if int(edf.stat().st_size) != record["source_size_bytes"]:
            raise ValueError("EDF size differs from inventory binding")
        if _sha256_file(edf) != record["source_signal_sha256"]:
            raise ValueError("EDF SHA-256 differs from inventory binding")
        _validate_edf_fixed_header_structure(edf)
        if _edf_continuity_mode(edf) == "EDF+D":
            raise EDFDiscontinuousTimelineUnsupportedError(
                "EDF+D discontinuous timeline is unsupported by the frozen "
                "causal v29 physical reader"
            )
        _atomic_json(
            state_path,
            _state(
                record,
                status="running",
                last_completed_stage="input_validation",
                diagnostic_status=None,
                event_count=None,
                attempt=attempt,
            ),
            replace=True,
        )

        stage = "coarse_scan"
        detection_path = case_dir / "detection_manifest.json"
        if not detection_path.exists():
            executor.scan(record=record, edf=edf, output=detection_path)
        detection = _validate_detection_for_record(detection_path, record)
        selected_count = _selected_count(detection)
        _atomic_json(
            state_path,
            _state(
                record,
                status="running",
                last_completed_stage=stage,
                diagnostic_status=None,
                event_count=selected_count,
                attempt=attempt,
            ),
            replace=True,
        )

        if analysis_profile == ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID:
            stage = "adaptive_event_analysis_v2"
            if not (adaptive_dir / "manifest.json").is_file():
                executor.adaptive_event_analysis(
                    edf=edf,
                    detection=detection_path,
                    output=adaptive_dir,
                )
            _validate_adaptive_profile_for_record(adaptive_dir, record)
            _atomic_json(
                state_path,
                _state(
                    record,
                    status="running",
                    last_completed_stage=stage,
                    diagnostic_status=None,
                    event_count=selected_count,
                    attempt=attempt,
                ),
                replace=True,
            )

        segment_receipts: list[Path] = []
        analysis_selection_path: Path | None = None
        if selected_count == 0:
            waveform_root = case_dir / "zero_candidate_waveforms"
            waveform_root.mkdir(parents=True, exist_ok=True)
            os.chmod(waveform_root, 0o700)
        else:
            stage = "event_registry"
            registry_path = case_dir / "event_registry.json"
            if registry_path.exists():
                registry = validate_detector_aligned_frozen_event_registry(
                    _read_json_object(registry_path)
                )
            else:
                registry = build_detector_aligned_frozen_event_registry(
                    detection,
                    source_transition_manifest_sha256=_sha256_file(detection_path),
                    expected_selected_count=selected_count,
                )
                _atomic_json(registry_path, registry, replace=False)
            if registry["recording_id"] != record["recording_id"]:
                raise ValueError("event registry identity differs from inventory")

            stage = "v29_ranking"
            ranking_dir = case_dir / "v29_ranking"
            ranking_manifest = ranking_dir / "manifest.json"
            candidate_selection_path = (
                ranking_dir / "analysis_selection_manifest.json"
            )
            if (
                not ranking_manifest.is_file()
                and not candidate_selection_path.is_file()
            ):
                executor.rank(
                    edf=edf,
                    detection=detection_path,
                    registry=registry_path,
                    output=ranking_dir,
                )
            if candidate_selection_path.is_file():
                selection = bind_long_term_eeg_analysis_selection(
                    _read_json_object(candidate_selection_path),
                    detection,
                )
                if selection["detector_selected_count"] != selected_count:
                    raise ValueError(
                        "analysis selection count differs from detector selection"
                    )
                analyzable_count = int(selection["analyzable_count"])
                analysis_selection_path = candidate_selection_path
            else:
                # Compatibility for an already-published pre-partition v1
                # ranking.  It remains valid only when it covers the complete
                # detector selection; partial coverage is never inferred.
                analyzable_count = selected_count

            if analyzable_count:
                ranking_value = _read_json_object(ranking_manifest)
                if ranking_value.get("recording_id") != record["recording_id"]:
                    raise ValueError("v29 ranking recording identity drifted")
                if ranking_value.get("event_count") != analyzable_count:
                    raise ValueError("v29 ranking analyzable event count drifted")

                stage = "event_materialization"
                events_dir = case_dir / "events"
                if not (events_dir / "manifest.json").is_file():
                    if analysis_selection_path is None:
                        executor.materialize_events(
                            edf=edf,
                            detection=detection_path,
                            registry=registry_path,
                            ranking_manifest=ranking_manifest,
                            output=events_dir,
                        )
                    else:
                        executor.materialize_events(
                            edf=edf,
                            detection=detection_path,
                            registry=registry_path,
                            ranking_manifest=ranking_manifest,
                            output=events_dir,
                            analysis_selection=analysis_selection_path,
                        )
                segment_receipts = _event_receipts(events_dir, record)
                if len(segment_receipts) != analyzable_count:
                    raise ValueError(
                        "event segment count differs from analyzable candidates"
                    )
                waveform_root = events_dir
            else:
                waveform_root = case_dir / "no_analyzable_candidate_waveforms"
                waveform_root.mkdir(parents=True, exist_ok=True)
                os.chmod(waveform_root, 0o700)

        stage = "report_materialization"
        report_dir = case_dir / "report"
        if not (report_dir / "manifest.json").is_file():
            if analysis_selection_path is None:
                executor.report(
                    detection=detection_path,
                    segment_receipts=segment_receipts,
                    waveform_root=waveform_root,
                    bundle_id="BUNDLE-" + str(record["recording_id"]),
                    output=report_dir,
                )
            else:
                executor.report(
                    detection=detection_path,
                    segment_receipts=segment_receipts,
                    waveform_root=waveform_root,
                    bundle_id="BUNDLE-" + str(record["recording_id"]),
                    output=report_dir,
                    analysis_selection=analysis_selection_path,
                )
        report = _completed_report(report_dir, record)
        if report is None:
            raise ValueError("report materializer did not publish a completed report")
        _atomic_json(
            state_path,
            _state(
                record,
                status="completed",
                last_completed_stage=stage,
                diagnostic_status=str(report["diagnostic_status"]),
                event_count=int(report["event_count"]),
                attempt=attempt,
            ),
            replace=True,
        )
        return _case_result(
            record,
            run_status="completed",
            diagnostic_status=str(report["diagnostic_status"]),
            event_count=int(report["event_count"]),
        )
    except Exception as error:  # isolate one record and continue the batch
        if isinstance(error, EDFFixedHeaderMalformedError):
            error_code = "edf_fixed_header_malformed"
        elif isinstance(error, EDFDiscontinuousTimelineUnsupportedError):
            error_code = "edf_discontinuous_timeline_unsupported"
        elif (
            stage == "input_validation"
            and record.get("inventory_error_code") is not None
        ):
            error_code = str(record["inventory_error_code"])
        elif stage == "input_validation":
            error_code = "input_edf_unreadable_or_binding_mismatch"
        else:
            error_code = "stage_execution_or_artifact_validation_failed"
        receipt = _failure_receipt(
            record,
            failed_stage=stage,
            error_code=error_code,
            error=error,
            attempt=attempt,
        )
        receipt_path = case_dir / "technical_failure_receipt.json"
        _atomic_json(receipt_path, receipt, replace=True)
        try:
            technical_report = _materialize_technical_unassessable_report(
                case_dir=case_dir,
                record=record,
                failure_receipt=receipt,
                attempt=attempt,
            )
        except Exception:
            _atomic_json(
                state_path,
                _state(
                    record,
                    status="technical_failure",
                    last_completed_stage=None,
                    diagnostic_status=None,
                    event_count=None,
                    attempt=attempt,
                ),
                replace=True,
            )
            return _case_result(
                record,
                run_status="technical_failure",
                failure_stage=stage,
            )
        technical_relative = technical_report.relative_to(case_dir).as_posix()
        completed_receipt = dict(receipt)
        completed_receipt["report_generated"] = True
        completed_receipt["technical_report_relative_dir"] = technical_relative
        _atomic_json(receipt_path, completed_receipt, replace=True)
        _atomic_json(
            state_path,
            _state(
                record,
                status="completed_technical_unassessable",
                last_completed_stage="technical_report_materialization",
                diagnostic_status="completed_technical_unassessable",
                event_count=0,
                attempt=attempt,
            ),
            replace=True,
        )
        return _case_result(
            record,
            run_status="completed_technical_unassessable",
            diagnostic_status="completed_technical_unassessable",
            event_count=0,
            failure_stage=stage,
            technical_artifact_relative_dir=technical_relative,
        )


def run_batch(
    *,
    inventory: Mapping[str, Any],
    dataset_root: Path,
    output_root: Path,
    executor: Any,
    dry_run: bool,
    max_recordings: int | None,
    requested_recording_ids: Sequence[str],
    qwen_requested: bool,
    analysis_profile: str = LEGACY_ANALYSIS_PROFILE_ID,
) -> dict[str, Any]:
    if analysis_profile not in ANALYSIS_PROFILE_IDS:
        raise ValueError("private batch analysis profile is unsupported")
    if (
        analysis_profile == ADAPTIVE_EVENT_ANALYSIS_PROFILE_ID
        and not dry_run
        and not ADAPTIVE_REPORT_ROUTE_CONNECTED
    ):
        raise RuntimeError(
            "adaptive_event_findings_v2 execution is fail-closed: the adaptive "
            "search/window release is not yet connected to event Findings, "
            "record aggregation, waveform rendering, or report materialization"
        )
    validated = validate_inventory(inventory)
    root = _dataset_root(dataset_root)
    output = output_root.resolve()
    if output.is_symlink():
        raise ValueError("output root must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    records_root = output / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    os.chmod(records_root, 0o700)

    all_records = list(validated["records"])
    requested = set(requested_recording_ids)
    unknown = requested.difference(item["recording_id"] for item in all_records)
    if unknown:
        raise ValueError(f"requested recording IDs are absent from inventory: {sorted(unknown)}")
    selected = [
        item for item in all_records if not requested or item["recording_id"] in requested
    ]
    if max_recordings is not None:
        if max_recordings < 1:
            raise ValueError("max_recordings must be positive")
        selected = selected[:max_recordings]
    selected_ids = {item["recording_id"] for item in selected}

    rows: list[dict[str, Any]] = []
    for record in all_records:
        if record["recording_id"] in selected_ids:
            status = "planned" if dry_run else "queued"
        else:
            status = "not_selected_in_this_run"
        rows.append(_case_result(record, run_status=status))
    coverage_path = output / "coverage_manifest.json"
    mode = "dry_run" if dry_run else "execution"
    coverage = _write_coverage(
        output=coverage_path,
        inventory=validated,
        rows=rows,
        mode=mode,
        qwen_requested=qwen_requested,
        analysis_profile=analysis_profile,
    )
    if dry_run:
        return coverage

    row_by_recording = {row["recording_id"]: row for row in rows}
    for record in selected:
        row_by_recording[record["recording_id"]] = _run_one(
            record=record,
            dataset_root=root,
            records_root=records_root,
            executor=executor,
            analysis_profile=analysis_profile,
        )
        rows = [row_by_recording[item["recording_id"]] for item in all_records]
        coverage = _write_coverage(
            output=coverage_path,
            inventory=validated,
            rows=rows,
            mode=mode,
            qwen_requested=qwen_requested,
            analysis_profile=analysis_profile,
        )
    return coverage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser(
        "build-inventory",
        allow_abbrev=False,
        help="Project a strict EEG-only unique-record inventory.",
    )
    source_group = inventory_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--rows119-manifest", type=Path)
    source_group.add_argument(
        "--signal-roster",
        type=Path,
        help=(
            "Frozen roster with relative_edf_path and stable PRIV-P patient_id; "
            "records are deduplicated by complete signal SHA-256."
        ),
    )
    source_group.add_argument(
        "--filesystem-edf-roster",
        type=Path,
        help=(
            "Frozen signal roster used to map every filesystem EDF through "
            "parent locators and exact signal hashes."
        ),
    )
    source_group.add_argument("--source-json", type=Path)
    inventory_parser.add_argument("--dataset-root", type=Path, required=True)
    inventory_parser.add_argument(
        "--locator-ledger",
        type=Path,
        help=(
            "Required with --filesystem-edf-roster; private append-only mapping "
            "ledger containing hashes and stable patient pseudonyms only."
        ),
    )
    inventory_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run", allow_abbrev=False, help="Run or plan the resumable report batch."
    )
    run_parser.add_argument("--inventory", type=Path, required=True)
    run_parser.add_argument("--dataset-root", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="Execute/plan only the first inventory record.",
    )
    run_parser.add_argument("--max-recordings", type=int)
    run_parser.add_argument("--recording-id", action="append", default=[])
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run_parser.add_argument("--use-qwen", action="store_true")
    run_parser.add_argument(
        "--analysis-profile",
        choices=tuple(sorted(ANALYSIS_PROFILE_IDS)),
        default=LEGACY_ANALYSIS_PROFILE_ID,
        help=(
            "The default preserves the frozen v1 report path. The reserved "
            "adaptive_event_findings_v2 profile is fail-closed until its "
            "variable event windows are connected to event Findings and the "
            "record-level report materializer; dry-run reports that state."
        ),
    )
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run_parser.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build-inventory":
        if args.filesystem_edf_roster is None and args.locator_ledger is not None:
            raise ValueError(
                "--locator-ledger is only valid with --filesystem-edf-roster"
            )
        if args.rows119_manifest is not None:
            source_rows, source_sha = _source_rows_from_csv(args.rows119_manifest)
            source_kind = "rows119_path_subject_projection_v1"
            allow_suffix_projection = True
            source_values_are_pseudonyms = False
            recording_unit_policy = "unique_edf_path_v1"
        elif args.signal_roster is not None:
            source_rows, source_sha = _source_rows_from_signal_roster(
                args.signal_roster
            )
            source_kind = "frozen_signal_roster_projection_v1"
            allow_suffix_projection = False
            source_values_are_pseudonyms = True
            recording_unit_policy = "unique_signal_sha256_v1"
        elif args.filesystem_edf_roster is not None:
            if args.locator_ledger is None:
                raise ValueError(
                    "--locator-ledger is required with --filesystem-edf-roster"
                )
            source_rows, source_sha, _ledger = build_filesystem_inventory_source(
                signal_roster_path=args.filesystem_edf_roster,
                dataset_root=args.dataset_root,
                locator_ledger_path=args.locator_ledger,
            )
            source_kind = "filesystem_edf_frozen_roster_projection_v1"
            allow_suffix_projection = False
            source_values_are_pseudonyms = True
            recording_unit_policy = "unique_signal_sha256_v1"
        else:
            source_rows, source_sha = _source_rows_from_explicit_json(args.source_json)
            source_kind = "explicit_path_subject_projection_v1"
            allow_suffix_projection = False
            source_values_are_pseudonyms = True
            recording_unit_policy = "unique_signal_sha256_v1"
        inventory = build_inventory(
            source_rows=source_rows,
            source_kind=source_kind,
            source_manifest_sha256=source_sha,
            dataset_root=args.dataset_root,
            allow_suffix_projection=allow_suffix_projection,
            subject_values_are_pseudonyms=source_values_are_pseudonyms,
            recording_unit_policy=recording_unit_policy,
        )
        _atomic_json(args.output, inventory, replace=False)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "inventory_id": inventory["inventory_id"],
                    "unique_record_count": inventory["record_count"],
                    "subject_count": inventory["subject_count"],
                    "ready_record_count": sum(
                        item["inventory_validation_status"] == READY
                        for item in inventory["records"]
                    ),
                    "source_rejection_count": len(inventory["source_rejections"]),
                    "onset_label_annotation_excel_or_gt_projected": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    inventory_value = validate_inventory(_strict_json(args.inventory))
    maximum = args.max_recordings
    if args.smoke_one:
        if maximum not in {None, 1}:
            raise ValueError("--smoke-one conflicts with --max-recordings other than 1")
        maximum = 1
    executor = None
    if not args.dry_run:
        executor = CliStageExecutor(
            python_executable=args.python_executable,
            device=args.device,
            base_url=args.base_url,
            use_qwen=args.use_qwen,
        )
    coverage = run_batch(
        inventory=inventory_value,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        executor=executor,
        dry_run=args.dry_run,
        max_recordings=maximum,
        requested_recording_ids=tuple(args.recording_id),
        qwen_requested=bool(args.use_qwen),
        analysis_profile=str(args.analysis_profile),
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "mode": coverage["mode"],
                "expected_record_count": coverage["expected_record_count"],
                "completed_report_count": coverage["completed_report_count"],
                "technical_failure_count": coverage["technical_failure_count"],
                "dataset_coverage_complete": coverage["dataset_coverage_complete"],
                "analysis_profile": str(args.analysis_profile),
                "annotation_excel_onset_label_or_gt_used": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if coverage["technical_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTIVE_REPORT_ROUTE_CONNECTED",
    "ADAPTIVE_COVERAGE_SCHEMA_VERSION",
    "ANALYSIS_PROFILE_IDS",
    "COVERAGE_SCHEMA_VERSION",
    "EXPLICIT_SOURCE_SCHEMA_VERSION",
    "INVENTORY_SCHEMA_VERSION",
    "LEGACY_ANALYSIS_PROFILE_ID",
    "CliStageExecutor",
    "build_inventory",
    "build_filesystem_inventory_source",
    "run_batch",
    "validate_inventory",
]
