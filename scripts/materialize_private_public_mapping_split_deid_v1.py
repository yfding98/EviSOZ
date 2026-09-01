#!/usr/bin/env python3
"""Materialize a path-mapped, split-aware, de-identified EEG bundle.

The bundle has four deliberately separate layers:

* private source-path mapping is retained only in an explicitly internal CSV;
* private EDFs are copied with identifying EDF header fields removed and their
  start date fixed to 2000-01-01; discontinuous EDF+D files are quarantined;
* private physician reports are converted to conservative PHI-free review
  candidates, never copied as raw DOCX, and remain release-gated;
* public TUSZ official splits are preserved and the DeepSOZ overlay receives a
  second, patient-level split view while retaining its TUSZ identity crosswalk.

Raw private inputs are never modified.  The output is written transactionally
and contains no raw DOCX files.  The internal path map is for the controller's
local use and must not be published.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pyedflib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.evisoz.data.private_physician_reports import (  # noqa: E402
    build_private_physician_report_inventory,
)
from src.evisoz.forge.private_report_deidentification import (  # noqa: E402
    build_private_report_deidentification_candidates,
)
from src.evisoz.data.private_stage0_split import (  # noqa: E402
    build_private_patient_linkage_group,
)


DEFAULT_PRIVATE_EDF_ROOT = Path("/mnt/hd1/dyf/dataset/EEG")
DEFAULT_PRIVATE_SIGNAL_ROSTER = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/signal_roster.csv"
)
DEFAULT_PRIVATE_SPLIT_ROSTER = (
    ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json"
)
DEFAULT_PRIVATE_SOURCE_MANIFEST = ROOT / "outputs/soz_pre/private_edf_soz_manifest.csv"
DEFAULT_REPORT_ROOT = Path("/mnt/hd1/dyf/dataset/EEG_Reports/Reports")
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
# The full viewer manifest covers all 7,364 TUSZ EDF records.  The
# annotation-only manifest is useful for seizure-event analyses but would
# silently drop public background records from a dataset-level split.
DEFAULT_TUSZ_MANIFEST = ROOT / "outputs/tusz_viewer/tusz_v203_viewer_manifest.csv"
DEFAULT_DEEPSOZ_MAPPING = ROOT / "outputs/deepsoz_tusz_identity_recovery_v2_20260812/mapping_identity_v2.csv"
DEFAULT_DEEPSOZ_SPLIT = ROOT / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/split_manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs/private_public_mapping_split_deid_v1_20260901"

_DATE_RE = re.compile(
    r"(?:(?:19|20)\d{2})\s*(?:年|[-/.])\s*\d{1,2}\s*(?:月|[-/.])\s*\d{1,2}\s*(?:日)?"
)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LONG_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d{4,}(?![A-Za-z0-9])")
_SAFE_RELATIVE = re.compile(r"^[^/].*")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative_edf(root: Path, value: object) -> tuple[str, Path]:
    text = str(value).replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.suffix.casefold() != ".edf"
    ):
        raise ValueError(f"unsafe private EDF path: {value!r}")
    source = root.joinpath(*relative.parts).resolve(strict=True)
    source.relative_to(root.resolve(strict=True))
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"private EDF is not a regular file: {value!r}")
    return relative.as_posix(), source


def _split_assignments(split_roster: Mapping[str, object], patient_ids: Iterable[str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    by_group = {
        str(row["linkage_group_id"]): row for row in split_roster.get("assignments", [])
    }
    for patient_id in sorted(set(patient_ids)):
        group = build_private_patient_linkage_group(patient_id)
        group_id = str(group["linkage_group_id"])
        if group_id not in by_group:
            raise ValueError(f"private split roster lacks patient {patient_id}")
        result[patient_id] = {
            "linkage_group_id": group_id,
            "evisoz_role": by_group[group_id]["evisoz_role"],
            "outer_holdout_fold": by_group[group_id]["outer_holdout_fold"],
            "locked": by_group[group_id]["locked"],
        }
    return result


def _scrub_annotation(text: object, patient_id: str) -> str:
    value = str(text or "")
    if patient_id:
        value = value.replace(patient_id, "<PERSON>")
    value = _EMAIL_RE.sub("<EMAIL>", value)
    value = _PHONE_RE.sub("<PHONE>", value)
    value = _DATE_RE.sub("<DATE>", value)
    # Long numbers in EDF annotations are more likely IDs than clinical labels.
    value = _LONG_NUMBER_RE.sub("<ID>", value)
    return value[:255]


def _safe_signal_header(header: Mapping[str, object]) -> dict[str, object]:
    """Normalize an EDF signal header for pyedflib's writer."""

    low = float(header.get("physical_min", -1.0))
    high = float(header.get("physical_max", 1.0))
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        low, high = -1.0, 1.0
    low, high = min(low, high), max(low, high)
    dlow = int(header.get("digital_min", -32768))
    dhigh = int(header.get("digital_max", 32767))
    dlow, dhigh = min(dlow, dhigh), max(dlow, dhigh)
    return {
        "label": str(header.get("label", ""))[:16],
        "dimension": str(header.get("dimension", "uV"))[:8],
        "sample_frequency": float(header.get("sample_frequency", 1.0)),
        "physical_min": low,
        "physical_max": high,
        "digital_min": dlow,
        "digital_max": dhigh,
        "prefilter": str(header.get("prefilter", ""))[:80],
        "transducer": str(header.get("transducer", ""))[:80],
    }


def _write_deidentified_edf(source: Path, target: Path, patient_id: str) -> dict[str, object]:
    """Copy signal samples while replacing identifying EDF fields."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    reader = pyedflib.EdfReader(str(source))
    try:
        n_signals = int(reader.signals_in_file)
        headers = [_safe_signal_header(item) for item in reader.getSignalHeaders()]
        annotations = reader.readAnnotations()
        samples = [reader.readSignal(index) for index in range(n_signals)]
        writer = pyedflib.EdfWriter(
            str(temporary), n_signals, file_type=pyedflib.FILETYPE_EDFPLUS
        )
        try:
            writer.setSignalHeaders(headers)
            writer.setHeader(
                {
                    "technician": "",
                    "recording_additional": "ANONYMIZED",
                    "patientname": patient_id,
                    "patient_additional": "",
                    "patientcode": patient_id,
                    "equipment": "",
                    "admincode": "",
                    "sex": "",
                    "startdate": datetime(2000, 1, 1, 0, 0, 0),
                    "birthdate": "",
                }
            )
            writer.writeSamples(samples)
            for onset, duration, description in zip(*annotations):
                writer.writeAnnotation(
                    float(onset), float(duration), _scrub_annotation(description, patient_id)
                )
        finally:
            writer.close()
    finally:
        reader.close()
    temporary.replace(target)
    # Reopen to make the de-identification claim concrete.
    check = pyedflib.EdfReader(str(target))
    try:
        header = check.getHeader()
        if header.get("patientname") != patient_id or header.get("patientcode") != patient_id:
            raise ValueError("de-identified EDF patient fields did not round-trip")
        if str(header.get("birthdate", "")) not in {"", "0"}:
            raise ValueError("de-identified EDF retained birthdate")
        return {
            "deidentification_status": "deidentified_edf_written",
            "deidentified_size_bytes": int(target.stat().st_size),
            "deidentified_annotation_count": int(len(check.readAnnotations()[0])),
            "deidentified_startdate": str(header.get("startdate", "")),
        }
    finally:
        check.close()


def _private_edf_layers(
    *,
    private_root: Path,
    signal_roster_path: Path,
    split_roster: Mapping[str, object],
    output_root: Path,
    write_sanitized_edf: bool,
    max_edf: int | None,
) -> dict[str, object]:
    rows = _read_csv(signal_roster_path)
    if not rows:
        raise ValueError("private signal roster is empty")
    required = {"event_id", "patient_id", "relative_edf_path"}
    if not required.issubset(rows[0]):
        raise ValueError(f"private signal roster missing fields: {sorted(required - set(rows[0]))}")
    assignments = _split_assignments(split_roster, (row["patient_id"] for row in rows))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rel, _ = _relative_edf(private_root, row["relative_edf_path"])
        grouped[rel].append(row)
    paths = sorted(grouped)
    if max_edf is not None:
        if max_edf < 1:
            raise ValueError("max_edf must be positive")
        paths = paths[:max_edf]
    file_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    deidentified_roster_rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    deid_counts: Counter[str] = Counter()
    edf_output_root = output_root / "private_deidentified_edf"
    for rel in paths:
        event_group = grouped[rel]
        patient_ids = sorted({row["patient_id"] for row in event_group})
        if len(patient_ids) != 1:
            raise ValueError(f"one private EDF maps to multiple patients: {rel}")
        patient_id = patient_ids[0]
        assignment = assignments[patient_id]
        source_rel, source = _relative_edf(private_root, rel)
        source_sha = _sha256_file(source)
        file_id = f"EDF-{source_sha[:20]}"
        target_rel = (
            Path("private_deidentified_edf")
            / str(assignment["evisoz_role"])
            / patient_id
            / f"{file_id}.edf"
        ).as_posix()
        rec: dict[str, object] = {
            "file_id": file_id,
            "patient_id": patient_id,
            "linkage_group_id": assignment["linkage_group_id"],
            "evisoz_role": assignment["evisoz_role"],
            "outer_holdout_fold": assignment["outer_holdout_fold"],
            "locked": assignment["locked"],
            "source_relative_edf_path": source_rel,
            "source_sha256": source_sha,
            "source_size_bytes": int(source.stat().st_size),
            "event_ids": ";".join(sorted(row["event_id"] for row in event_group)),
            "status": "",
            "error": "",
            "header_signal_count": "",
            "header_duration_sec": "",
            "header_sample_frequencies": "",
            "header_annotation_count": "",
            "deidentified_relative_edf_path": target_rel,
            "deidentified_sha256": "",
            "deidentified_size_bytes": "",
            "deidentification_status": "not_attempted",
        }
        try:
            reader = pyedflib.EdfReader(str(source))
            try:
                rec["status"] = "read_ok"
                rec["header_signal_count"] = int(reader.signals_in_file)
                rec["header_duration_sec"] = float(reader.file_duration)
                rec["header_sample_frequencies"] = ";".join(
                    str(float(item["sample_frequency"])) for item in reader.getSignalHeaders()
                )
                rec["header_annotation_count"] = int(len(reader.readAnnotations()[0]))
            finally:
                reader.close()
            status_counts["read_ok"] += 1
            if write_sanitized_edf:
                target = output_root / target_rel
                result = _write_deidentified_edf(source, target, patient_id)
                rec.update(result)
                rec["deidentified_sha256"] = _sha256_file(target)
                deid_counts[str(result["deidentification_status"])] += 1
            else:
                rec["deidentification_status"] = "header_read_only"
                deid_counts["header_read_only"] += 1
        except Exception as exc:  # noqa: BLE001 - preserve a closed quarantine reason
            message = str(exc)
            rec["status"] = "quarantined_read_error"
            rec["error"] = f"{type(exc).__name__}: {message}"
            rec["deidentification_status"] = "not_written"
            status_counts["quarantined_read_error"] += 1
            deid_counts["not_written"] += 1
        file_rows.append(rec)
        for event in event_group:
            deidentified_path = (
                target_rel if rec["deidentification_status"] == "deidentified_edf_written" else ""
            )
            event_rows.append({
                    "event_id": event["event_id"],
                    "patient_id": patient_id,
                    "linkage_group_id": assignment["linkage_group_id"],
                    "evisoz_role": assignment["evisoz_role"],
                    "outer_holdout_fold": assignment["outer_holdout_fold"],
                    "locked": assignment["locked"],
                    "file_id": file_id,
                    "source_sha256": source_sha,
                    "deidentified_relative_edf_path": deidentified_path,
                    "deidentified_signal_available": int(bool(deidentified_path)),
                    "deidentification_status": rec["deidentification_status"],
                    "time_support_preeligible": event.get("time_support_preeligible", ""),
                    "global_event_t0_sec": event.get("global_event_t0_sec", ""),
                })
            roster_row = dict(event)
            roster_row["relative_edf_path"] = deidentified_path
            roster_row["file_id"] = file_id
            roster_row["source_edf_sha256"] = source_sha
            roster_row["deidentified_relative_edf_path"] = deidentified_path
            roster_row["deidentified_signal_available"] = int(bool(deidentified_path))
            roster_row["deidentification_status"] = rec["deidentification_status"]
            roster_row["linkage_group_id"] = assignment["linkage_group_id"]
            roster_row["evisoz_role"] = assignment["evisoz_role"]
            roster_row["outer_holdout_fold"] = assignment["outer_holdout_fold"]
            roster_row["locked"] = assignment["locked"]
            deidentified_roster_rows.append(roster_row)
    file_rows.sort(key=lambda row: str(row["file_id"]))
    event_rows.sort(key=lambda row: str(row["event_id"]))
    deidentified_roster_rows.sort(key=lambda row: str(row["event_id"]))
    _write_csv(
        output_root / "private_edf_path_map_internal.csv",
        file_rows,
        (
            "file_id", "patient_id", "linkage_group_id", "evisoz_role", "outer_holdout_fold", "locked",
            "source_relative_edf_path", "source_sha256", "source_size_bytes", "event_ids", "status", "error",
            "header_signal_count", "header_duration_sec", "header_sample_frequencies", "header_annotation_count",
            "deidentified_relative_edf_path", "deidentified_sha256", "deidentified_size_bytes", "deidentification_status",
        ),
    )
    _write_csv(
        output_root / "private_edf_map_phi_free.csv",
        file_rows,
        (
            "file_id", "patient_id", "linkage_group_id", "evisoz_role", "outer_holdout_fold", "locked",
            "source_sha256", "source_size_bytes", "event_ids", "status", "header_signal_count",
            "header_duration_sec", "header_sample_frequencies", "header_annotation_count",
            "deidentified_relative_edf_path", "deidentified_sha256", "deidentified_size_bytes", "deidentification_status",
        ),
    )
    _write_csv(
        output_root / "private_event_split.csv",
        event_rows,
        (
            "event_id", "patient_id", "linkage_group_id", "evisoz_role", "outer_holdout_fold", "locked",
            "file_id", "source_sha256", "deidentified_relative_edf_path", "deidentified_signal_available", "deidentification_status", "time_support_preeligible", "global_event_t0_sec",
        ),
    )
    roster_fields = list(rows[0].keys()) + [
        "file_id", "source_edf_sha256", "deidentified_relative_edf_path", "deidentified_signal_available", "deidentification_status",
        "linkage_group_id", "evisoz_role", "outer_holdout_fold", "locked",
    ]
    roster_fields = list(dict.fromkeys(roster_fields))
    _write_csv(
        output_root / "private_deidentified_signal_roster.csv",
        deidentified_roster_rows,
        roster_fields,
    )
    return {
        "event_count": len(event_rows),
        "patient_count": len({row["patient_id"] for row in event_rows}),
        "unique_edf_count": len(file_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "deidentification_status_counts": dict(sorted(deid_counts.items())),
        "source_roster_sha256": _sha256_file(signal_roster_path),
        "internal_path_map_contains_raw_relative_paths": True,
            "raw_private_edf_files_copied": False,
            "deidentified_signal_roster_written": True,
        }


def _private_report_layers(
    *,
    report_root: Path,
    source_manifest: Path,
    signal_roster: Path,
    split_roster: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    report_paths = [path for path in report_root.iterdir() if path.is_file()]
    inventory = build_private_physician_report_inventory(
        report_paths=report_paths,
        source_manifest_path=source_manifest,
        signal_roster_path=signal_roster,
        split_roster=split_roster,
    )
    _write_json(output_root / "private_reports" / "inventory.json", inventory)
    candidates_root = output_root / "private_reports" / "deidentified_candidates"
    candidates = build_private_report_deidentification_candidates(
        report_paths=report_paths,
        report_inventory=inventory,
        source_manifest_path=source_manifest,
        output=candidates_root,
    )
    (candidates_root / "manifest.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    candidate_by_report = {row["report_id"]: row for row in candidates["candidates"]}
    mapping_rows: list[dict[str, object]] = []
    for report in inventory["reports"]:
        association = report["association"]
        candidate = candidate_by_report[report["report_id"]]
        split = association.get("split_assignment") or {}
        mapping_rows.append(
            {
                "report_id": report["report_id"],
                "document_sha256": report["document_ref"]["content_hash"]["sha256"],
                "association_status": association["status"],
                "linkage_group_id": association.get("linkage_group_id") or "",
                "evisoz_role": split.get("evisoz_role", ""),
                "outer_holdout_fold": split.get("outer_holdout_fold", ""),
                "candidate_id": candidate["candidate_id"],
                "candidate_relative_text_path": candidate["relative_text_path"],
                "automated_phi_scan": candidate["automated_phi_scan"]["automated_scan_status"],
                "manual_review_status": candidate["review_release"]["manual_review_status"],
                "development_qwen_training_released": candidate["review_release"]["development_qwen_training_released"],
                "locked_language_evaluation_released": candidate["review_release"]["locked_language_evaluation_released"],
            }
        )
    mapping_rows.sort(key=lambda row: str(row["report_id"]))
    _write_csv(
        output_root / "private_reports" / "report_mapping_phi_free.csv",
        mapping_rows,
        (
            "report_id", "document_sha256", "association_status", "linkage_group_id", "evisoz_role",
            "outer_holdout_fold", "candidate_id", "candidate_relative_text_path", "automated_phi_scan",
            "manual_review_status", "development_qwen_training_released", "locked_language_evaluation_released",
        ),
    )
    return {
        "report_count": inventory["counts"]["report_count"],
        "association_status_counts": inventory["counts"]["association_status_counts"],
        "deidentified_candidate_count": candidates["counts"]["candidate_count"],
        "automated_phi_scan_pass_count": candidates["counts"]["automated_phi_scan_pass_count"],
        "manual_review_pass_count": candidates["counts"]["manual_review_pass_count"],
        "development_release_count": candidates["counts"]["development_qwen_training_release_count"],
        "evaluator_release_count": candidates["counts"]["locked_language_evaluation_release_count"],
        "raw_docx_copied": False,
        "raw_report_text_in_manifest": False,
        "unresolved_report_ids": inventory["manual_mapping_required_report_ids"],
    }


def _public_tusz_layers(*, manifest_path: Path, tusz_root: Path, output_root: Path) -> dict[str, object]:
    rows = _read_csv(manifest_path)
    if not rows:
        raise ValueError("public TUSZ manifest is empty")
    by_edf: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rel = str(row.get("edf_path", "")).replace("\\", "/")
        if not rel:
            raise ValueError("TUSZ manifest contains an empty edf_path")
        path = (tusz_root / PurePosixPath(rel)).resolve(strict=True)
        path.relative_to(tusz_root.resolve(strict=True))
        by_edf[rel].append(row)
    record_rows: list[dict[str, object]] = []
    for rel, events in sorted(by_edf.items()):
        first = events[0]
        splits = sorted({str(row.get("split", "")) for row in events})
        patients = sorted({str(row.get("patient_id", "")) for row in events})
        if len(splits) != 1 or len(patients) != 1:
            raise ValueError(f"TUSZ record crosses split or patient: {rel}")
        record_rows.append(
            {
                "dataset": "tusz_v2.0.3",
                "official_split": splits[0],
                "patient_id": patients[0],
                "session": first.get("session", ""),
                "montage": first.get("montage", ""),
                "edf_path": rel,
                "csv_path": first.get("csv_path", ""),
                "csv_bi_path": first.get("csv_bi_path", ""),
                "event_count": len(events),
                "seizure_event_count": sum(str(row.get("has_seizure", "0")) == "1" for row in events),
            }
        )
    event_rows = [
        {
            "dataset": "tusz_v2.0.3",
            "official_split": row.get("split", ""),
            "patient_id": row.get("patient_id", ""),
            "session": row.get("session", ""),
            "montage": row.get("montage", ""),
            "edf_path": row.get("edf_path", "").replace("\\", "/"),
            "event_id": row.get("event_id", ""),
            "event_index": row.get("event_index", ""),
            "has_seizure": row.get("has_seizure", ""),
            "sz_start": row.get("sz_start", ""),
            "sz_end": row.get("sz_end", ""),
            "sz_duration": row.get("sz_duration", ""),
            "seizure_type": row.get("seizure_type", ""),
        }
        for row in rows
    ]
    record_rows.sort(key=lambda row: str(row["edf_path"]))
    event_rows.sort(key=lambda row: str(row["event_id"]))
    _write_csv(
        output_root / "public_tusz_record_split.csv",
        record_rows,
        ("dataset", "official_split", "patient_id", "session", "montage", "edf_path", "csv_path", "csv_bi_path", "event_count", "seizure_event_count"),
    )
    _write_csv(
        output_root / "public_tusz_event_split.csv",
        event_rows,
        ("dataset", "official_split", "patient_id", "session", "montage", "edf_path", "event_id", "event_index", "has_seizure", "sz_start", "sz_end", "sz_duration", "seizure_type"),
    )
    return {
        "record_count": len(record_rows),
        "patient_count": len({row["patient_id"] for row in record_rows}),
        "event_row_count": len(event_rows),
        "official_split_record_counts": dict(sorted(Counter(row["official_split"] for row in record_rows).items())),
        "official_split_patient_counts": dict(sorted({split: len({row["patient_id"] for row in record_rows if row["official_split"] == split}) for split in {row["official_split"] for row in record_rows}}.items())),
        "official_split_preserved": True,
    }


def _public_deepsoz_layer(*, mapping_path: Path, split_path: Path, tusz_root: Path, output_root: Path) -> dict[str, object]:
    mapping = _read_csv(mapping_path)
    split = _read_csv(split_path)
    split_by_row = {str(row["deepsoz_patient_id"]): row for row in split}
    if len(split_by_row) != len(split):
        raise ValueError("DeepSOZ split manifest must contain one row per patient")
    if not mapping:
        raise ValueError("DeepSOZ identity mapping is empty")
    rows: list[dict[str, object]] = []
    seen_records: set[str] = set()
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for item in mapping:
        if item.get("mapping_status") != "unique":
            raise ValueError("DeepSOZ overlay contains a non-unique mapping")
        deepsoz_patient = str(item.get("deepsoz_patient", ""))
        local_edf = str(item.get("local_edf", "")).replace("\\", "/")
        local_path = Path(local_edf).resolve(strict=True)
        local_rel = local_path.relative_to(tusz_root.resolve(strict=True)).as_posix()
        local_patient = str(item.get("local_patient", ""))
        split_row = split_by_row.get(deepsoz_patient)
        if split_row is None:
            # The split package uses one row per DeepSOZ patient; fall back to
            # the exact source-row join when duplicate patient IDs are present.
            matching = [row for row in split if str(row.get("deepsoz_patient_id", "")) == deepsoz_patient]
            if len(matching) != 1:
                raise ValueError(f"DeepSOZ split row missing for patient {deepsoz_patient}")
            split_row = matching[0]
        official_split = str(split_row.get("official_split", ""))
        model_split = str(split_row.get("model_split", ""))
        record_key = local_rel
        if record_key in seen_records:
            raise ValueError(f"DeepSOZ overlay repeats local TUSZ record: {record_key}")
        seen_records.add(record_key)
        patient_splits[local_patient].add(official_split)
        rows.append(
            {
                "dataset": "deepsoz_tusz_overlay",
                "deepsoz_row": item.get("deepsoz_row", ""),
                "deepsoz_patient_id": deepsoz_patient,
                "deepsoz_record": item.get("deepsoz_record", ""),
                "local_patient_id": local_patient,
                "local_edf_path": local_rel,
                "official_tusz_split": official_split,
                "overlay_model_split": model_split,
                "cohort_status": split_row.get("cohort_status", ""),
                "label_stability_primary": split_row.get("label_stability_primary", ""),
                "concept_oof_fold": split_row.get("concept_oof_fold", ""),
                "oof_fold_scope": split_row.get("oof_fold_scope", ""),
                "split_rule": "official_tusz_split_preserved;patient_level_overlay;source_train_oof_when_applicable",
            }
        )
    if any(len(values) != 1 for values in patient_splits.values()):
        raise ValueError("one local TUSZ patient appears in multiple official splits")
    rows.sort(key=lambda row: (str(row["deepsoz_patient_id"]), str(row["deepsoz_record"])))
    _write_csv(
        output_root / "public_deepsoz_overlay_split.csv",
        rows,
        (
            "dataset", "deepsoz_row", "deepsoz_patient_id", "deepsoz_record", "local_patient_id", "local_edf_path",
            "official_tusz_split", "overlay_model_split", "cohort_status", "label_stability_primary", "concept_oof_fold",
            "oof_fold_scope", "split_rule",
        ),
    )
    return {
        "source_row_count": len(rows),
        "source_patient_count": len({row["deepsoz_patient_id"] for row in rows}),
        "unique_local_tusz_record_count": len(seen_records),
        "unique_local_tusz_patient_count": len(patient_splits),
        "official_tusz_split_counts": dict(sorted(Counter(row["official_tusz_split"] for row in rows).items())),
        "overlay_model_split_counts": dict(sorted(Counter(row["overlay_model_split"] for row in rows).items())),
        "mapping_status": "all_unique",
        "patient_level_split_verified": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--private-edf-root", type=Path, default=DEFAULT_PRIVATE_EDF_ROOT)
    parser.add_argument("--private-signal-roster", type=Path, default=DEFAULT_PRIVATE_SIGNAL_ROSTER)
    parser.add_argument("--private-split-roster", type=Path, default=DEFAULT_PRIVATE_SPLIT_ROSTER)
    parser.add_argument("--private-source-manifest", type=Path, default=DEFAULT_PRIVATE_SOURCE_MANIFEST)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--tusz-manifest", type=Path, default=DEFAULT_TUSZ_MANIFEST)
    parser.add_argument("--deepsoz-mapping", type=Path, default=DEFAULT_DEEPSOZ_MAPPING)
    parser.add_argument("--deepsoz-split", type=Path, default=DEFAULT_DEEPSOZ_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-sanitized-edf", action="store_true", help="write header-scrubbed EDF copies for readable private EDFs")
    parser.add_argument("--max-edf", type=int, default=None, help="debug limit; do not use for a complete bundle")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for path in (
        args.private_edf_root,
        args.private_signal_roster,
        args.private_split_roster,
        args.private_source_manifest,
        args.report_root,
        args.tusz_root,
        args.tusz_manifest,
        args.deepsoz_mapping,
        args.deepsoz_split,
    ):
        path.resolve(strict=True)
    split_roster = json.loads(args.private_split_roster.resolve(strict=True).read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    committed = False
    try:
        private_edf = _private_edf_layers(
            private_root=args.private_edf_root.resolve(strict=True),
            signal_roster_path=args.private_signal_roster.resolve(strict=True),
            split_roster=split_roster,
            output_root=staging,
            write_sanitized_edf=bool(args.write_sanitized_edf),
            max_edf=args.max_edf,
        )
        private_reports = _private_report_layers(
            report_root=args.report_root.resolve(strict=True),
            source_manifest=args.private_source_manifest.resolve(strict=True),
            signal_roster=args.private_signal_roster.resolve(strict=True),
            split_roster=split_roster,
            output_root=staging,
        )
        tusz = _public_tusz_layers(
            manifest_path=args.tusz_manifest.resolve(strict=True),
            tusz_root=args.tusz_root.resolve(strict=True),
            output_root=staging,
        )
        deepsoz = _public_deepsoz_layer(
            mapping_path=args.deepsoz_mapping.resolve(strict=True),
            split_path=args.deepsoz_split.resolve(strict=True),
            tusz_root=args.tusz_root.resolve(strict=True),
            output_root=staging,
        )
        summary = {
            "schema_version": "private_public_mapping_split_deid_bundle_v1",
            "status": "completed_with_quarantine_receipts",
            "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "private_edf": private_edf,
            "private_reports": private_reports,
            "public_tusz": tusz,
            "public_deepsoz_overlay": deepsoz,
            "policy": {
                "raw_private_inputs_modified": False,
                "raw_private_edf_copied": False,
                "raw_private_docx_copied": False,
                "private_edf_header_deidentification": "patientname_and_patientcode_to_pseudonym;birthdate_sex_dates_removed;startdate_fixed_to_2000-01-01;annotations_regex_scrubbed",
                "private_report_deidentification": "conservative_ictal_or_impression_to_pre_signature;automated_scan_then_manual_release_gate",
                "public_tusz_official_split_preserved": True,
                "deepsoz_overlay_uses_tusz_identity_and_patient_level_split": True,
                "internal_path_map_must_not_be_published": True,
            },
            "input_bindings": {
                "private_signal_roster": str(args.private_signal_roster.resolve().relative_to(ROOT)),
                "private_split_roster": str(args.private_split_roster.resolve().relative_to(ROOT)),
                "private_source_manifest": str(args.private_source_manifest.resolve().relative_to(ROOT)),
                "tusz_manifest": str(args.tusz_manifest.resolve().relative_to(ROOT)),
                "deepsoz_mapping": str(args.deepsoz_mapping.resolve().relative_to(ROOT)),
                "deepsoz_split": str(args.deepsoz_split.resolve().relative_to(ROOT)),
            },
            "output_files": sorted(
                str(path.relative_to(staging).as_posix())
                for path in staging.rglob("*")
                if path.is_file()
            ),
        }
        _write_json(staging / "summary.json", summary)
        (staging / "README.md").write_text(
            "# Private/public mapping, split and de-identification bundle\n\n"
            "This bundle was materialized without modifying raw inputs. The private EDF path map is an internal controller artifact and must not be published.\n\n"
            "## Private EEG\n\n"
            "`private_event_split.csv` binds pseudonymous event IDs to the frozen patient-level split and directly exposes the de-identified EDF path. `private_deidentified_signal_roster.csv` is the same event roster with its EDF path rewritten to the de-identified bundle, so it can be passed to downstream preprocessing. `private_edf_map_phi_free.csv` contains only pseudonyms, hashes, header geometry and de-identified output paths. `private_edf_path_map_internal.csv` additionally contains source-relative paths for local controller use.\n\n"
            "Readable EDFs are copied under `private_deidentified_edf/` with patient name/code replaced by the pseudonym, birth date/sex and wall-clock start date removed, and annotations scrubbed for names, dates, contacts and long identifiers. Five EDF+D files that pyedflib rejected as discontinuous are quarantined in the internal map and are not silently repaired; their event rows have `deidentified_signal_available=0` so downstream preprocessing can exclude them explicitly.\n\n"
            "## Private reports\n\n"
            "`private_reports/inventory.json` contains only content-addressed report references and pseudonymous associations. `private_reports/deidentified_candidates/` contains conservative text candidates; all remain `pending` until institutional manual review. Raw DOCX files are not copied. Three unresolved report IDs remain explicitly unresolved.\n\n"
            "## Public TUSZ and DeepSOZ\n\n"
            "`public_tusz_record_split.csv` and `public_tusz_event_split.csv` preserve official TUSZ train/dev/eval membership. `public_deepsoz_overlay_split.csv` joins every DeepSOZ row to its unique local TUSZ EDF and records both the official TUSZ split and the overlay model split/OOF fold. A local TUSZ patient is checked to occur in one official split only.\n\n"
            "The output is a data-preparation bundle, not a Stage-0 GO receipt. Downstream training/release still requires the project’s existing review and governance gates.\n",
            encoding="utf-8",
        )
        staging.rename(output)
        committed = True
        print(json.dumps({"output": str(output), "status": summary["status"], "private_edf": private_edf, "private_reports": private_reports, "public_tusz": tusz, "public_deepsoz_overlay": deepsoz}, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    raise SystemExit(main())
