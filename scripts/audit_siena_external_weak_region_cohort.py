#!/usr/bin/env python3
"""Audit Siena Scalp EEG without inventing channel-level SOZ labels.

The release supplies seizure intervals plus patient-level temporal/frontal and
laterality fields.  Those fields are retained as weak external region labels;
they are never expanded into C18 electrode targets.  Source timing text is
messy, so every extracted clock value and ambiguity is preserved and an event
is timing-ready only when a unique, in-record interpretation exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Iterable


DEFAULT_ROOT = Path("/mnt/hd1/dyf/dataset/SienaScalpEEG_v1.0.0")
DEFAULT_OUTPUT = Path("outputs/siena_external_weak_region_audit_v1_20260814.json")
SCHEMA = "siena_external_weak_region_audit_v1"
STANDARD_19 = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)
LEGACY_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
SOURCE_FILENAME_CORRECTIONS = {
    ("PN01", "PN01.edf"): "PN01-1.edf",
    ("PN06", "PNO6-1.edf"): "PN06-1.edf",
    ("PN06", "PNO6-2.edf"): "PN06-2.edf",
    ("PN06", "PNO6-4.edf"): "PN06-4.edf",
    ("PN11", "PN11-.edf"): "PN11-1.edf",
}
TIME_RE = re.compile(
    r"(?<!\d)(?<!\d\s)(?P<hour>[0-2]?\d)[.:](?P<minute>[0-5]\d)[.:](?P<second>[0-5]\d)(?!\d)"
)
SEIZURE_BLOCK_RE = re.compile(
    r"(?=^\s*Seizure\s+n\s*\d+\b)", re.I | re.M
)
SEIZURE_NUMBER_RE = re.compile(r"Seizure\s+n\s*(\d+)\b", re.I)


def _clock_candidates(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for match in TIME_RE.finditer(text):
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        second = int(match.group("second"))
        if hour > 23:
            continue
        rows.append(
            {
                "raw": match.group(0),
                "seconds_of_day": hour * 3600 + minute * 60 + second,
            }
        )
    return rows


def _field_lines(text: str, labels: Iterable[str]) -> list[str]:
    label_tuple = tuple(value.lower() for value in labels)
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith(label_tuple):
            rows.append(line)
    return rows


def _field_times(text: str, labels: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in _field_lines(text, labels):
        for value in _clock_candidates(line):
            rows.append({**value, "source_line": line})
    return rows


def _file_candidates(text: str) -> list[str]:
    values = re.findall(r"^\s*File\s+name\s*:\s*([^\s]+\.edf)\b", text, re.I | re.M)
    return list(dict.fromkeys(value.strip() for value in values))


def _resolve_source_files(
    patient_id: str, source_files: list[str]
) -> tuple[list[str], list[dict[str, str]]]:
    resolved: list[str] = []
    corrections: list[dict[str, str]] = []
    for source in source_files:
        value = SOURCE_FILENAME_CORRECTIONS.get((patient_id, source), source)
        resolved.append(value)
        if value != source:
            corrections.append({"source": source, "resolved": value})
    return list(dict.fromkeys(resolved)), corrections


def _canonical_channel(raw_name: str) -> str | None:
    value = str(raw_name).strip().upper()
    value = re.sub(r"^EEG\s+", "", value)
    value = re.sub(r"-(REF|LE|AR|AVG)$", "", value)
    value = re.sub(r"[^A-Z0-9]", "", value)
    value = LEGACY_ALIASES.get(value, value)
    return value if value in STANDARD_19 else None


def _edf_ascii_int(value: bytes, *, field: str) -> int:
    try:
        return int(value.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid EDF integer field {field}") from exc


def _edf_storage_contract(path: Path) -> dict[str, int]:
    """Prove local EDF byte completeness from its native record structure."""
    actual_bytes = int(path.stat().st_size)
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError(f"EDF fixed header is truncated: {path}")
        header_bytes = _edf_ascii_int(fixed[184:192], field="header_bytes")
        data_records = _edf_ascii_int(fixed[236:244], field="data_records")
        signal_count = _edf_ascii_int(fixed[252:256], field="signal_count")
        if signal_count <= 0 or header_bytes != 256 * (signal_count + 1):
            raise ValueError(f"invalid EDF header geometry: {path}")
        if data_records < 0:
            raise ValueError(f"EDF data-record count is unknown: {path}")
        samples_offset = 256 + 216 * signal_count
        stream.seek(samples_offset)
        sample_fields = stream.read(8 * signal_count)
        if len(sample_fields) != 8 * signal_count:
            raise ValueError(f"EDF signal header is truncated: {path}")
    samples_per_record = [
        _edf_ascii_int(sample_fields[index * 8 : (index + 1) * 8], field="samples_per_record")
        for index in range(signal_count)
    ]
    if any(value < 0 for value in samples_per_record):
        raise ValueError(f"negative EDF samples-per-record value: {path}")
    samples_per_record_total = sum(samples_per_record)
    expected_bytes = header_bytes + 2 * data_records * samples_per_record_total
    if actual_bytes != expected_bytes:
        state = "truncated" if actual_bytes < expected_bytes else "has trailing bytes"
        raise ValueError(
            f"EDF storage size mismatch ({state}): {path}; "
            f"actual={actual_bytes}, expected={expected_bytes}"
        )
    return {
        "file_size_bytes": actual_bytes,
        "expected_file_size_bytes": expected_bytes,
        "edf_data_records": data_records,
        "edf_signal_count": signal_count,
        "edf_samples_per_record_total": samples_per_record_total,
    }


def _read_edf_header(path: Path) -> dict[str, object]:
    import mne

    storage = _edf_storage_contract(path)
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    canonical = [_canonical_channel(value) for value in raw.ch_names]
    counts = {name: canonical.count(name) for name in STANDARD_19}
    missing = [name for name, count in counts.items() if count == 0]
    duplicate = [name for name, count in counts.items() if count > 1]
    duration_sec = float(raw.n_times) / float(raw.info["sfreq"])
    return {
        **storage,
        "relative_edf": str(path),
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "duration_sec": duration_sec,
        "raw_channel_names": list(raw.ch_names),
        "standard19_missing": missing,
        "standard19_duplicate": duplicate,
        "standard19_complete_unique": not missing and not duplicate,
    }


def _unique_seconds(rows: list[dict[str, object]]) -> list[int]:
    return sorted({int(row["seconds_of_day"]) for row in rows})


def _relative_time(value: int, record_start: int) -> float:
    return float((value - record_start) % 86400)


def _event_timing_status(
    *,
    start_rows: list[dict[str, object]],
    end_rows: list[dict[str, object]],
    registration_start_rows: list[dict[str, object]],
    duration_sec: float | None,
) -> dict[str, object]:
    reasons: list[str] = []
    starts = _unique_seconds(start_rows)
    ends = _unique_seconds(end_rows)
    registrations = _unique_seconds(registration_start_rows)
    if len(starts) != 1:
        reasons.append("event_start_not_unique")
    if len(ends) != 1:
        reasons.append("event_end_not_unique")
    if len(registrations) != 1:
        reasons.append("registration_start_not_unique")
    if duration_sec is None or not math.isfinite(duration_sec) or duration_sec <= 0:
        reasons.append("edf_duration_unavailable")
    onset = offset = None
    if not reasons:
        onset = _relative_time(starts[0], registrations[0])
        offset = _relative_time(ends[0], registrations[0])
        if offset < onset:
            offset += 86400.0
        tolerance = 1.0
        if onset < 0 or onset > float(duration_sec) + tolerance:
            reasons.append("event_start_outside_edf")
        if offset <= onset or offset > float(duration_sec) + tolerance:
            reasons.append("event_end_outside_edf")
    return {
        "timing_ready": not reasons,
        "onset_sec_from_record_start": onset,
        "offset_sec_from_record_start": offset,
        "reason_codes": reasons,
    }


def _strict_external_status(
    timing_status: dict[str, object], header: dict[str, object] | None
) -> dict[str, object]:
    reasons = list(str(value) for value in timing_status["reason_codes"])
    if header is not None and not bool(header["standard19_complete_unique"]):
        reasons.append("standard19_not_complete_unique")
    return {
        "strict_external_ready": not reasons,
        "strict_external_reason_codes": list(dict.fromkeys(reasons)),
        "standard19_complete_unique_for_event": (
            bool(header["standard19_complete_unique"]) if header is not None else None
        ),
    }


def _load_subjects(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    output = []
    for row in rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError("subject_info contains an empty patient_id")
        output.append(
            {
                "patient_id": patient_id,
                "age_years": int(str(row["age_years"]).strip()),
                "gender": str(row["gender"]).strip(),
                "seizure_class": str(row["seizure"]).strip(),
                "weak_localization": str(row["localization"]).strip(),
                "weak_lateralization": str(row["lateralization"]).strip(),
                "declared_eeg_channel_count": int(str(row["eeg_channel"]).strip()),
                "declared_seizure_count": int(str(row["number_seizures"]).strip()),
                "declared_recording_minutes": int(str(row["rec_time_minutes"]).strip()),
            }
        )
    if len(output) != 14 or len({row["patient_id"] for row in output}) != 14:
        raise ValueError("expected exactly 14 unique Siena patients")
    return output


def audit(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    subject_path = root / "subject_info.csv"
    records_path = root / "RECORDS"
    subjects = _load_subjects(subject_path)
    expected_records = [
        line.strip()
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(expected_records) != len(set(expected_records)):
        raise ValueError("RECORDS contains duplicate paths")
    missing_files = [value for value in expected_records if not (root / value).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Siena download incomplete: {len(missing_files)} expected EDF files missing"
        )

    header_by_subject_name: dict[tuple[str, str], dict[str, object]] = {}
    edf_rows: list[dict[str, object]] = []
    for relative in expected_records:
        path = root / relative
        header = _read_edf_header(path)
        header["relative_edf"] = relative
        edf_rows.append(header)
        key = (path.parent.name, path.name)
        if key in header_by_subject_name:
            raise ValueError(f"duplicate subject/EDF key in RECORDS: {key}")
        header_by_subject_name[key] = header

    subject_events: dict[str, list[dict[str, object]]] = {}
    for subject in subjects:
        patient_id = str(subject["patient_id"])
        source_path = root / patient_id / f"Seizures-list-{patient_id}.txt"
        if not source_path.is_file():
            raise FileNotFoundError(f"missing seizure list: {source_path}")
        text = source_path.read_text(encoding="utf-8", errors="replace")
        pieces = SEIZURE_BLOCK_RE.split(text)
        preamble = pieces[0]
        blocks = [value for value in pieces[1:] if value.strip()]
        preamble_source_files = _file_candidates(preamble)
        preamble_registration = _field_times(
            preamble, ("Registration start time",)
        )

        raw_rows: list[dict[str, object]] = []
        registration_by_file: dict[str, list[dict[str, object]]] = {}
        for block in blocks:
            number_match = SEIZURE_NUMBER_RE.search(block)
            source_files = _file_candidates(block) or preamble_source_files
            files, filename_corrections = _resolve_source_files(
                patient_id, source_files
            )
            registration = _field_times(block, ("Registration start time",))
            if not registration and len(files) == 1 and preamble_registration:
                registration = preamble_registration
            if len(files) == 1 and registration:
                registration_by_file.setdefault(files[0], []).extend(registration)
            raw_rows.append(
                {
                    "seizure_number": int(number_match.group(1)) if number_match else None,
                    "source_file_candidates": source_files,
                    "file_candidates": files,
                    "filename_corrections": filename_corrections,
                    "start_candidates": _field_times(
                        block, ("Seizure start time", "Start time")
                    ),
                    "end_candidates": _field_times(
                        block, ("Seizure end time", "End time")
                    ),
                    "registration_start_candidates": registration,
                    "source_block": block.strip(),
                }
            )

        # Missing registration lines may be inherited only from a unique value
        # explicitly supplied elsewhere for the exact same EDF file.
        for row in raw_rows:
            files = row["file_candidates"]
            if not row["registration_start_candidates"] and len(files) == 1:
                candidates = registration_by_file.get(files[0], [])
                if len(_unique_seconds(candidates)) == 1:
                    row["registration_start_candidates"] = candidates
                    row["registration_inherited_from_same_edf"] = True
            row.setdefault("registration_inherited_from_same_edf", False)
            header = (
                header_by_subject_name.get((patient_id, files[0]))
                if len(files) == 1
                else None
            )
            duration = float(header["duration_sec"]) if header is not None else None
            timing_status = _event_timing_status(
                    start_rows=row["start_candidates"],
                    end_rows=row["end_candidates"],
                    registration_start_rows=row["registration_start_candidates"],
                    duration_sec=duration,
                )
            row.update(timing_status)
            if len(files) != 1:
                row["timing_ready"] = False
                row["reason_codes"] = list(row["reason_codes"]) + [
                    "edf_filename_not_unique"
                ]
            elif header is None:
                row["timing_ready"] = False
                row["reason_codes"] = list(row["reason_codes"]) + [
                    "edf_not_listed"
                ]
            row.update(_strict_external_status(row, header))
        subject_events[patient_id] = raw_rows

    event_rows = [row for values in subject_events.values() for row in values]
    declared_events = sum(int(row["declared_seizure_count"]) for row in subjects)
    if len(event_rows) != declared_events:
        raise ValueError(
            f"parsed event count {len(event_rows)} != declared count {declared_events}"
        )
    localization_counts: dict[str, int] = {}
    lateralization_counts: dict[str, int] = {}
    timing_reason_counts: dict[str, int] = {}
    strict_reason_counts: dict[str, int] = {}
    for event in event_rows:
        for reason in set(str(value) for value in event["reason_codes"]):
            timing_reason_counts[reason] = timing_reason_counts.get(reason, 0) + 1
        for reason in set(
            str(value) for value in event["strict_external_reason_codes"]
        ):
            strict_reason_counts[reason] = strict_reason_counts.get(reason, 0) + 1
    for row in subjects:
        localization_counts[str(row["weak_localization"])] = (
            localization_counts.get(str(row["weak_localization"]), 0) + 1
        )
        lateralization_counts[str(row["weak_lateralization"])] = (
            lateralization_counts.get(str(row["weak_lateralization"]), 0) + 1
        )
    return {
        "schema_version": SCHEMA,
        "dataset": "Siena Scalp EEG Database v1.0.0",
        "root": str(root),
        "source_url": "https://physionet.org/content/siena-scalp-eeg/1.0.0/",
        "license": "CC BY 4.0",
        "counts": {
            "patients": len(subjects),
            "edf_records": len(edf_rows),
            "declared_seizures": declared_events,
            "parsed_seizures": len(event_rows),
            "timing_ready_seizures": sum(bool(row["timing_ready"]) for row in event_rows),
            "timing_not_ready_seizures": sum(
                not bool(row["timing_ready"]) for row in event_rows
            ),
            "filename_corrected_seizures": sum(
                bool(row["filename_corrections"]) for row in event_rows
            ),
            "strict_external_ready_seizures": sum(
                bool(row["strict_external_ready"]) for row in event_rows
            ),
            "strict_external_not_ready_seizures": sum(
                not bool(row["strict_external_ready"]) for row in event_rows
            ),
            "complete_unique_standard19_edfs": sum(
                bool(row["standard19_complete_unique"]) for row in edf_rows
            ),
        },
        "weak_label_distribution": {
            "localization": localization_counts,
            "lateralization": lateralization_counts,
        },
        "timing_not_ready_reason_counts": dict(sorted(timing_reason_counts.items())),
        "strict_external_not_ready_reason_counts": dict(
            sorted(strict_reason_counts.items())
        ),
        "scientific_boundary": {
            "soz_channel_targets_present": False,
            "soz_region_gold_present": False,
            "localization_field_role": "patient_level_weak_temporal_or_frontal_metadata",
            "lateralization_field_role": "patient_level_weak_external_metadata",
            "allowed_use": (
                "frozen_external_seizure_phenotype_and_region_laterality_robustness_audit"
            ),
            "forbidden_use": (
                "c18_soz_training_or_model_selection_or_private_reopening"
            ),
        },
        "subjects": subjects,
        "edf_headers": edf_rows,
        "events_by_subject": subject_events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.root)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
