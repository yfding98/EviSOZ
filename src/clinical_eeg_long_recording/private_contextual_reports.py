"""Source-attributed private long-term EEG clinical report drafts.

This module deliberately sits *beside* the frozen EEG-only renderer.  It never
feeds a physician workbook value or an EDF annotation back into seizure
detection, signal findings, research ranking, or the automatic EEG impression.
Instead it renders three visibly attributed layers:

* the already frozen signal-only report;
* physician workbook observations, bound by the post-freeze release ledger;
* conservatively classified EDF annotation events.

The output is private and pseudonymous.  A restricted source map is emitted so
the data owner can map one report back to each physical EDF without placing raw
patient/file names in report bodies or the cohort index.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
from html import escape as html_escape
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from src.soz.long_term_clinical_context import (
    EEG_POINT_MARKER_TYPES,
    SOURCE_BEHAVIOR_TYPES,
    classify_edf_annotation_description,
)


SCHEMA_VERSION = "private_contextual_clinical_eeg_report_v1"
MANIFEST_SCHEMA_VERSION = "private_contextual_clinical_eeg_report_manifest_v1"

DEFAULT_DATASET_ROOT = Path("/mnt/hd1/dyf/dataset/EEG")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "outputs/private_long_recording_inventory_v1_full141_20260819.json"
DEFAULT_BASE_REPORT_ROOT = ROOT / "outputs/private_long_recording_reports_v2_3_full141_20260820"
DEFAULT_DOCTOR_BUNDLE = ROOT / "outputs/private_clinical_eeg_doctor_labels_postfreeze_v2_3_20260820.json"
DEFAULT_OUTPUT = ROOT / "outputs/private-reports"
DEFAULT_ANNOTATIONS = DEFAULT_DATASET_ROOT / "edf_annotations.csv"
DEFAULT_WORKBOOKS = (
    DEFAULT_DATASET_ROOT / "EEG-fMRI颞叶癫痫(1).xls",
    DEFAULT_DATASET_ROOT / "头皮扩散.xlsx",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SZ_RE = re.compile(r"SZ\d+(?:[-_]\d+)?", re.IGNORECASE)
_UNCERTAIN_RE = re.compile(
    r"(?:\?|？|疑似|可能|可疑|不确定|待定|uncertain|possible|questionable)",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_IDENTITY_FIELD_RE = re.compile(
    r"(?:姓名|患者姓名|住院号|门诊号|病例号|病案号|身份证号|电话|手机号|地址)"
    r"\s*[:：]?\s*[^,;，；\s]+",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s]+|/(?:mnt|home|data|dataset|Users)/[^\s]+)",
    re.IGNORECASE,
)

_ANNOTATION_ZH = {
    "event_marker": "事件点标记",
    "eeg_event_marker": "脑电事件点标记",
    "end_marker": "结束点标记",
    "motor_activity": "运动表现",
    "behavioral_arrest": "行为停止/减少",
    "responsiveness_change": "反应性改变",
    "vocalization": "发声",
    "eye_or_head_deviation": "眼/头偏转",
    "automotor_activity": "自动症",
    "autonomic_or_salivation": "自主神经表现/流涎",
}

_DOCTOR_STATUS_ZH = {
    "available": "可唯一绑定",
    "source_conflict": "来源冲突",
    "ambiguous_mapping": "映射歧义",
    "not_available": "未获得可绑定标注",
}


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _finite(value: object, *, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _cell_text(value: object) -> str:
    """Convert one workbook cell without collapsing the explicit value '无'."""

    if value is None:
        return ""
    try:
        if bool(math.isnan(value)):  # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value).strip())


def sanitize_private_text(
    value: object,
    *,
    sensitive_tokens: Iterable[str] = (),
    maximum_length: int = 500,
) -> str:
    """Return a display-only, control-safe and conservatively de-identified string."""

    text = _cell_text(value)
    if not text:
        return ""
    text = _CONTROL_RE.sub(" ", text)
    text = _IDENTITY_FIELD_RE.sub("[身份字段已脱敏]", text)
    text = _EMAIL_RE.sub("[邮箱已脱敏]", text)
    text = _LONG_NUMBER_RE.sub("[长数字已脱敏]", text)
    text = _ABSOLUTE_PATH_RE.sub("[路径已脱敏]", text)
    for raw in sorted({token.strip() for token in sensitive_tokens if token.strip()}, key=len, reverse=True):
        if len(raw) < 2:
            continue
        text = re.sub(re.escape(raw), "[来源标识已脱敏]", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > maximum_length:
        text = text[: maximum_length - 1].rstrip() + "…"
    return text


def normalize_annotation_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("annotation EDF path is empty")
    normalized = value.strip().replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        relative.is_absolute()
        or relative.suffix.lower() != ".edf"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("annotation EDF path is unsafe")
    return relative.as_posix()


def format_clock(value: object) -> str:
    seconds = _finite(value, default=0.0) or 0.0
    if -0.01 < seconds < 0.0:
        seconds = 0.0
    sign = "-" if seconds < 0 else ""
    milliseconds = int(round(abs(seconds) * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole, millis = divmod(remainder, 1000)
    return f"{sign}{hours:02d}:{minutes:02d}:{whole:02d}.{millis:03d}"


def _safe_relative_file(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(resolved_root)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("source EDF must be a regular file")
    return resolved


def parse_doctor_workbook_cells(
    workbook_paths: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Read canonical workbooks and key raw source cells by frozen locator fingerprint."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pandas is required to read physician workbooks") from exc

    by_fingerprint: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for raw_path in workbook_paths:
        path = raw_path.resolve(strict=True)
        if path.is_symlink() or path.suffix.lower() not in {".xls", ".xlsx", ".xlsm"}:
            raise ValueError("physician workbook is not a supported regular file")
        workbook_sha = sha256_file(path)
        source_event_count = 0
        excel = pd.ExcelFile(path)
        for sheet_name in excel.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet_name, header=None)
            if frame.shape[0] < 3 or frame.shape[1] < 8:
                continue
            top = [_cell_text(item) for item in frame.iloc[0].tolist()]
            sub = [_cell_text(item) for item in frame.iloc[1].tolist()]
            groups: dict[str, dict[str, int]] = {}
            current: str | None = None
            for column, (top_value, sub_value) in enumerate(zip(top, sub)):
                if _SZ_RE.fullmatch(top_value):
                    current = top_value.upper()
                    groups.setdefault(current, {})
                if current is None:
                    continue
                if "起始" in sub_value:
                    groups[current]["onset"] = column
                elif "显著" in sub_value:
                    groups[current]["significant"] = column
                elif "扩散" in sub_value:
                    groups[current]["spread"] = column
                elif "覆盖" in sub_value:
                    groups[current]["coverage"] = column
            for row_index in range(2, len(frame)):
                if not _cell_text(frame.iat[row_index, 0]):
                    continue
                for slot, columns in groups.items():
                    values = {
                        name: _cell_text(frame.iat[row_index, column])
                        for name, column in columns.items()
                    }
                    if not any(values.values()):
                        continue
                    source_row = row_index + 1
                    fingerprint = canonical_sha256(
                        {
                            "workbook_sha256": workbook_sha,
                            "sheet_token_sha256": hashlib.sha256(
                                str(sheet_name).encode("utf-8")
                            ).hexdigest(),
                            "source_row": source_row,
                            "source_event_slot": slot,
                        }
                    )
                    if fingerprint in by_fingerprint:
                        raise ValueError("physician source locator fingerprint repeats")
                    by_fingerprint[fingerprint] = {
                        "source_locator_fingerprint": fingerprint,
                        "source_event_slot": slot,
                        "onset_text": sanitize_private_text(values.get("onset", "")),
                        "significant_electrodes_text": sanitize_private_text(
                            values.get("significant", "")
                        ),
                        "early_spread_text": sanitize_private_text(values.get("spread", "")),
                        "all_channel_coverage_text": sanitize_private_text(
                            values.get("coverage", "")
                        ),
                        "explicit_no_significant": values.get("significant", "") == "无",
                        "explicit_no_spread": values.get("spread", "") == "无",
                        "coverage_kept_separate_from_spread": True,
                    }
                    source_event_count += 1
        receipts.append(
            {
                "workbook_sha256": workbook_sha,
                "source_event_count": source_event_count,
                "raw_patient_identity_released": False,
                "coverage_promoted_to_spread": False,
            }
        )
    return by_fingerprint, receipts


def parse_annotation_csv(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Project a private EDF annotation export to closed, source-attributed rows."""

    resolved = path.resolve(strict=True)
    file_sha = sha256_file(resolved)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with resolved.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            relative = normalize_annotation_relative_path(row.get("edf_path"))
            grouped[relative].append(dict(row))

    output: dict[str, dict[str, Any]] = {}
    totals: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    for relative, rows in grouped.items():
        relative_path = PurePosixPath(relative)
        sensitive_tokens = {
            relative_path.parent.name,
            relative_path.stem,
            re.sub(r"[_-]?SZ\d.*$", "", relative_path.stem, flags=re.IGNORECASE),
        }
        projected: list[dict[str, Any]] = []
        valid_rows = 0
        for source_row, row in enumerate(rows, start=2):
            description = _cell_text(row.get("description"))
            onset = _finite(row.get("onset_sec"))
            if not description or onset is None:
                totals["placeholder_rows"] += 1
                continue
            valid_rows += 1
            codes = classify_edf_annotation_description(description)
            if not codes:
                totals["excluded_or_unclassified_rows"] += 1
                continue
            duration = _finite(row.get("duration_ann_sec"), default=0.0) or 0.0
            display = sanitize_private_text(
                description,
                sensitive_tokens=sensitive_tokens,
                maximum_length=300,
            )
            for code in codes:
                code_counts[code] += 1
            projected.append(
                {
                    "recording_offset_seconds": float(onset),
                    "display_time": format_clock(onset),
                    "annotation_duration_seconds": max(0.0, float(duration)),
                    "annotation_types": list(codes),
                    "annotation_type_display_zh": [
                        _ANNOTATION_ZH[code] for code in codes
                    ],
                    "source_text_redacted": display,
                    "uncertain": bool(_UNCERTAIN_RE.search(description)),
                    "source_row": source_row,
                    "source_description_sha256": hashlib.sha256(
                        description.encode("utf-8")
                    ).hexdigest(),
                    "source_attribution": "edf_annotation_export",
                }
            )
        projected.sort(
            key=lambda item: (
                item["recording_offset_seconds"],
                item["source_row"],
                item["annotation_types"],
            )
        )
        behavior_count = sum(
            any(code in SOURCE_BEHAVIOR_TYPES for code in item["annotation_types"])
            for item in projected
        )
        marker_count = sum(
            any(code in EEG_POINT_MARKER_TYPES for code in item["annotation_types"])
            for item in projected
        )
        output[relative] = {
            "source_file_sha256": file_sha,
            "csv_rows": len(rows),
            "valid_annotation_rows": valid_rows,
            "included_closed_rows": len(projected),
            "excluded_or_unclassified_rows": valid_rows - len(projected),
            "behavior_row_count": behavior_count,
            "marker_row_count": marker_count,
            "annotations": projected,
        }
        totals["source_paths"] += 1
        totals["csv_rows"] += len(rows)
        totals["valid_annotation_rows"] += valid_rows
        totals["included_closed_rows"] += len(projected)
        totals["behavior_rows"] += behavior_count
        totals["marker_rows"] += marker_count
    receipt = {
        "annotation_csv_sha256": file_sha,
        **dict(sorted(totals.items())),
        "annotation_type_counts": dict(sorted(code_counts.items())),
        "raw_patient_identity_released": False,
        "unclassified_text_released": False,
    }
    return output, receipt


def inspect_edf_header(path: Path) -> dict[str, Any]:
    """Read acquisition metadata without loading signal samples.

    pyEDFlib is preferred because it exposes native per-signal sample rates.
    MNE is a metadata-only fallback for malformed files accepted by MNE's EDF
    reader.  Fixed-header identity fields and start date/time are never
    returned.
    """

    pyedflib_error: str | None = None
    try:
        import pyedflib

        with pyedflib.EdfReader(str(path)) as reader:
            frequencies = sorted(
                {round(float(value), 6) for value in reader.getSampleFrequencies()}
            )
            annotations = reader.readAnnotations()
            return {
                "status": "readable",
                "reader": "pyedflib_header_only",
                "recording_duration_seconds": float(reader.file_duration),
                "signal_channel_count": int(reader.signals_in_file),
                "sample_frequencies_hz": frequencies,
                "raw_edf_annotation_count": int(len(annotations[2])),
                "identity_header_fields_released": False,
                "start_datetime_released": False,
            }
    except Exception as exc:  # malformed private files are expected
        pyedflib_error = type(exc).__name__

    try:
        import mne

        raw = mne.io.read_raw_edf(
            str(path),
            preload=False,
            verbose="ERROR",
        )
        try:
            sfreq = float(raw.info["sfreq"])
            duration = float(raw.n_times) / sfreq if sfreq > 0 else None
            return {
                "status": "readable_with_fallback",
                "reader": "mne_preload_false",
                "recording_duration_seconds": duration,
                "signal_channel_count": int(len(raw.ch_names)),
                "sample_frequencies_hz": [sfreq] if sfreq > 0 else [],
                "raw_edf_annotation_count": int(len(raw.annotations)),
                "pyedflib_failure_class": pyedflib_error,
                "identity_header_fields_released": False,
                "start_datetime_released": False,
            }
        finally:
            raw.close()
    except Exception as exc:
        return {
            "status": "technical_unreadable",
            "reader": None,
            "recording_duration_seconds": None,
            "signal_channel_count": None,
            "sample_frequencies_hz": [],
            "raw_edf_annotation_count": None,
            "pyedflib_failure_class": pyedflib_error,
            "mne_failure_class": type(exc).__name__,
            "identity_header_fields_released": False,
            "start_datetime_released": False,
        }


def build_physical_file_roster(
    *,
    dataset_root: Path,
    inventory: Mapping[str, Any],
    expected_physical_count: int | None = None,
    expected_unique_signal_count: int | None = None,
) -> list[dict[str, Any]]:
    """Map every physical EDF to one frozen unique-signal report unit."""

    root = dataset_root.resolve(strict=True)
    records = inventory.get("records")
    if not isinstance(records, list):
        raise TypeError("inventory records must be a list")
    if expected_unique_signal_count is not None and len(records) != expected_unique_signal_count:
        raise ValueError("inventory unique-signal count differs from expectation")

    by_relative: dict[str, Mapping[str, Any]] = {}
    by_sha: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("inventory record must be an object")
        relative = str(record.get("edf_relative_path", ""))
        signal_sha = str(record.get("source_signal_sha256", ""))
        if not relative or _SHA256_RE.fullmatch(signal_sha) is None:
            raise ValueError("inventory EDF identity is invalid")
        if relative in by_relative or signal_sha in by_sha:
            raise ValueError("inventory unique-signal identity repeats")
        path = _safe_relative_file(root, relative)
        if path.stat().st_size != int(record["source_size_bytes"]):
            raise ValueError("inventory source size drifted")
        by_relative[relative] = record
        by_sha[signal_sha] = record

    physical_paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".edf"
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if expected_physical_count is not None and len(physical_paths) != expected_physical_count:
        raise ValueError("physical EDF count differs from expectation")

    provisional: list[dict[str, Any]] = []
    for path in physical_paths:
        relative = path.relative_to(root).as_posix()
        record = by_relative.get(relative)
        is_canonical = record is not None
        if record is None:
            signal_sha = sha256_file(path)
            record = by_sha.get(signal_sha)
            if record is None:
                raise ValueError("physical EDF does not map to the frozen inventory")
            if path.stat().st_size != int(record["source_size_bytes"]):
                raise ValueError("physical alias size differs from frozen signal")
        provisional.append(
            {
                "source_path": path,
                "source_relative_path": relative,
                "recording_id": str(record["recording_id"]),
                "patient_pseudonym": str(record["patient_pseudonym"]),
                "source_signal_sha256": str(record["source_signal_sha256"]),
                "source_size_bytes": int(record["source_size_bytes"]),
                "canonical_relative_path": str(record["edf_relative_path"]),
                "is_canonical_inventory_path": is_canonical,
            }
        )

    aliases_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in provisional:
        if not row["is_canonical_inventory_path"]:
            aliases_by_record[row["recording_id"]].append(row)
    for rows in aliases_by_record.values():
        rows.sort(key=lambda item: item["source_relative_path"])
        for index, row in enumerate(rows, start=1):
            row["file_report_id"] = f"{row['recording_id']}-ALIAS{index:02d}"
            row["alias_index"] = index
    for row in provisional:
        if row["is_canonical_inventory_path"]:
            row["file_report_id"] = row["recording_id"]
            row["alias_index"] = 0

    report_ids = [row["file_report_id"] for row in provisional]
    if len(report_ids) != len(set(report_ids)):
        raise ValueError("physical report IDs repeat")
    if {row["recording_id"] for row in provisional} != set(by_relative_record["recording_id"] for by_relative_record in by_relative.values()):
        raise ValueError("physical roster does not cover every inventory record")
    return provisional


def _doctor_context_view(
    record: Mapping[str, Any],
    raw_source_by_fingerprint: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    status = str(record.get("doctor_label_status", "not_available"))
    if status not in _DOCTOR_STATUS_ZH:
        raise ValueError("doctor label status is unsupported")
    labels: list[dict[str, Any]] = []
    for raw_label in record.get("doctor_labels", []):
        if not isinstance(raw_label, Mapping):
            raise TypeError("doctor label must be an object")
        source_receipt = raw_label.get("source_receipt")
        if not isinstance(source_receipt, Mapping):
            raise TypeError("doctor label source receipt is missing")
        fingerprint = str(source_receipt.get("source_locator_fingerprint", ""))
        raw_cells = raw_source_by_fingerprint.get(fingerprint)
        if raw_cells is None:
            raise ValueError("published doctor label cannot be replayed to source cells")
        channel_reference = raw_label.get("physician_channel_reference")
        if not isinstance(channel_reference, Mapping):
            raise TypeError("doctor channel reference is missing")
        labels.append(
            {
                "label_id": str(raw_label["label_id"]),
                "source_event_slot": str(raw_label["source_event_slot"]),
                "source_conflict_variant": bool(raw_label["source_conflict_variant"]),
                "duplicate_equivalent_source_count": int(
                    raw_label["duplicate_equivalent_source_count"]
                ),
                "onset_projection": deepcopy(raw_label["onset"]),
                "significant_electrodes_normalized": list(
                    channel_reference.get("significant_electrodes", [])
                ),
                "early_spread_electrodes_normalized": list(
                    channel_reference.get("spread_electrodes", [])
                ),
                "excluded_nonstandard_significant_token_count": int(
                    channel_reference.get(
                        "excluded_out_of_scope_significant_token_count", 0
                    )
                ),
                "excluded_nonstandard_spread_token_count": int(
                    channel_reference.get("excluded_out_of_scope_spread_token_count", 0)
                ),
                # Never project the legacy diffuse_spread_present field: the
                # historical parser folded "coverage all channels" into it.
                "legacy_diffuse_spread_field_used": False,
                "source_cells": deepcopy(dict(raw_cells)),
            }
        )
    labels.sort(key=lambda item: (item["source_event_slot"], item["label_id"]))
    if status == "available" and not labels:
        raise ValueError("available doctor label record contains no labels")
    if status == "source_conflict" and len(labels) < 2:
        raise ValueError("source conflict must expose competing variants")
    if status in {"ambiguous_mapping", "not_available"} and labels:
        raise ValueError("unbound doctor record unexpectedly contains labels")
    return {
        "status": status,
        "status_display_zh": _DOCTOR_STATUS_ZH[status],
        "label_count": len(labels),
        "labels": labels,
        "mapping_is_recording_level_only": True,
        "bound_to_automatic_detector_event": False,
        "source_conflict_not_silently_resolved": True,
        "coverage_promoted_to_spread": False,
    }


def _validated_report_html(
    *,
    base_report_root: Path,
    report_receipt: Mapping[str, Any],
) -> Path:
    relative = str(report_receipt.get("report_html_relative_path", ""))
    if not relative:
        raise ValueError("frozen report HTML locator is missing")
    path = _safe_relative_file(base_report_root, relative)
    expected = str(report_receipt.get("report_html_sha256", ""))
    if _SHA256_RE.fullmatch(expected) is None or sha256_file(path) != expected:
        raise ValueError("frozen report HTML hash drifted")
    return path


def _automatic_signal_context(
    *,
    base_report_root: Path,
    doctor_record: Mapping[str, Any],
) -> dict[str, Any]:
    report_receipt = doctor_record.get("report_receipt")
    if not isinstance(report_receipt, Mapping):
        raise TypeError("post-freeze record lacks a report receipt")
    report_html = _validated_report_html(
        base_report_root=base_report_root,
        report_receipt=report_receipt,
    )
    report_kind = str(report_receipt.get("report_kind", ""))
    context: dict[str, Any] = {
        "report_kind": report_kind,
        "diagnostic_status": str(report_receipt.get("diagnostic_status", "")),
        "event_count": int(report_receipt.get("event_count", 0)),
        "frozen_report_html_path": report_html,
        "frozen_report_html_sha256": str(report_receipt["report_html_sha256"]),
        "doctor_or_annotation_used_in_signal_analysis": False,
    }
    if report_kind == "eeg_report":
        manifest_relative = str(report_receipt["report_manifest_relative_path"])
        report_dir = _safe_relative_file(base_report_root, manifest_relative).parent
        bundle_path = report_dir / "bundle.json"
        if bundle_path.is_symlink() or not bundle_path.is_file():
            raise FileNotFoundError(bundle_path)
        bundle_raw = _load_json(bundle_path)
        from src.clinical_eeg_long_recording import render as signal_render

        bundle = signal_render._validated_bundle(bundle_raw)  # noqa: SLF001
        impression = signal_render._automatic_eeg_impression(bundle)  # noqa: SLF001
        event_rows = signal_render._event_rows(bundle, {})  # noqa: SLF001
        selected, analyzable, rejected = signal_render._analysis_candidate_counts(  # noqa: SLF001
            bundle
        )
        settings = signal_render._recording_signal_settings(bundle)  # noqa: SLF001
        context.update(
            {
                "status": "completed_signal_report",
                "recording_duration_seconds": float(
                    bundle["recording_duration_seconds"]
                ),
                "detector_selected_candidate_count": selected,
                "analyzable_candidate_count": analyzable,
                "rejected_candidate_count": rejected,
                "qualified_event_finding_count": len(event_rows),
                "signal_settings": settings,
                "event_findings": [
                    {
                        "event_and_interval": row[0],
                        "finding_text_zh": row[1],
                        "supporting_derivations": row[2],
                    }
                    for row in event_rows
                ],
                "automatic_impression": impression,
            }
        )
    elif report_kind == "technical_unassessable_report":
        technical_json = report_html.with_suffix(".json")
        value = _load_json(technical_json)
        context.update(
            {
                "status": "technical_unassessable",
                "failure_stage": value.get("failure_stage"),
                "technical_conclusion_zh": value.get("conclusion_zh"),
                "recording_duration_seconds": None,
                "signal_settings": {},
                "event_findings": [],
                "automatic_impression": None,
            }
        )
    else:
        raise ValueError("unsupported frozen report kind")
    return context


def _duration_text(value: object) -> str:
    number = _finite(value)
    if number is None:
        return "无法可靠获取"
    return f"{number:.3f} 秒（{format_clock(number)}）"


def _frequency_text(values: object) -> str:
    if not isinstance(values, list) or not values:
        return "无法可靠获取"
    return "、".join(f"{float(value):g} Hz" for value in values)


def _html_table(
    css_class: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    heading = "".join(f"<th>{html_escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html_escape(str(item))}</td>" for item in row)
        + "</tr>"
        for row in rows
    )
    return (
        f'<table class="{html_escape(css_class, quote=True)}"><thead><tr>{heading}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _doctor_html(context: Mapping[str, Any]) -> str:
    status = str(context["status"])
    boundary = ""
    if status == "source_conflict":
        boundary = (
            '<p class="warning">同一患者-SZ 键存在工作簿来源冲突；'
            "下方并列变体，不形成单一起始/扩散结论。</p>"
        )
    elif status == "ambiguous_mapping":
        boundary = (
            '<p class="warning">工作簿键对应多份独立 EDF 信号，'
            "无法唯一绑定；本报告不猜测归属。</p>"
        )
    elif status == "not_available":
        boundary = '<p class="muted">未获得可唯一绑定的医生工作簿标注。</p>'

    rows: list[tuple[str, str, str, str, str]] = []
    for label in context["labels"]:
        cells = label["source_cells"]
        onset_projection = label["onset_projection"]
        onset = cells["onset_text"] or "未填写"
        projection = str(onset_projection.get("display_zh", ""))
        if projection:
            onset += "\n" + projection
        significant = cells["significant_electrodes_text"] or "未填写"
        spread = cells["early_spread_text"] or "未填写"
        coverage = cells["all_channel_coverage_text"] or "未填写"
        variant = "冲突变体" if label["source_conflict_variant"] else "来源标注"
        rows.append(
            (
                f"{label['source_event_slot']}\n{variant}",
                onset,
                significant,
                spread,
                coverage,
            )
        )
    table = (
        _html_table(
            "doctor",
            ("SZ 槽位/状态", "起始信息", "显著电极", "早期扩散", "覆盖全导"),
            rows,
        )
        if rows
        else ""
    )
    return (
        f'<p><strong>绑定状态：</strong>{html_escape(str(context["status_display_zh"]))}</p>'
        + boundary
        + table
        + '<p class="source-boundary">上表为医生工作簿来源转录；显著电极为来源强标注，'
        "早期扩散只表示后续受累，不等于起始电极。“覆盖全导”单独呈现，"
        "未被改写为早期扩散或 SOZ。</p>"
    )


def _annotation_html(context: Mapping[str, Any]) -> str:
    annotations = context.get("annotations", [])
    if not annotations:
        return (
            '<p class="muted">未形成可进入闭集时间线的 EDF annotation。'
            "这不表示无临床事件。</p>"
        )
    rows = []
    for item in annotations:
        duration = float(item["annotation_duration_seconds"])
        duration_text = f"{duration:g} s" if duration > 0 else "点标记"
        source_text = item["source_text_redacted"]
        if item["uncertain"]:
            source_text += "（来源含不确定表述）"
        rows.append(
            (
                item["display_time"],
                duration_text,
                "、".join(item["annotation_type_display_zh"]),
                source_text,
            )
        )
    return _html_table(
        "annotations",
        ("相对记录时间", "时长", "闭集类型", "EDF 来源标注（已脱敏）"),
        rows,
    ) + (
        '<p class="source-boundary">时间点和表现来自 EDF annotation，未经本流程独立视频核实；'
        "点标记不自动等于医师确认的脑电起始或终止。"
        "未识别的自由文本不进入报告叙述。</p>"
    )


def _signal_html(context: Mapping[str, Any], frozen_href: str) -> str:
    link = (
        f'<p><a href="{html_escape(frozen_href, quote=True)}">打开冻结的 EEG-only 证据报告与波形</a></p>'
    )
    if context["status"] == "technical_unassessable":
        return (
            '<p class="warning">'
            + html_escape(str(context.get("technical_conclusion_zh", "自动 EEG 分析技术不可评价。")))
            + "</p>"
            + link
        )
    settings = context.get("signal_settings", {})
    metadata = _html_table(
        "signal-meta",
        ("字段", "自动信号层内容"),
        (
            ("粗筛候选", context["detector_selected_candidate_count"]),
            ("进入分析", context["analyzable_candidate_count"]),
            ("合格事件级所见", context["qualified_event_finding_count"]),
            ("采样率", settings.get("sampling_rate", "无法评价")),
            ("滤波", settings.get("filter", "无法评价")),
            ("参考/蒙太奇", settings.get("reference_montage", "无法评价")),
        ),
    )
    rows = [
        (
            item["event_and_interval"],
            item["finding_text_zh"],
            item["supporting_derivations"],
        )
        for item in context["event_findings"]
    ]
    events = (
        _html_table(
            "signal-events",
            ("自动候选/区间", "经资格门的信号所见", "相关双极导联"),
            rows,
        )
        if rows
        else '<p class="muted">未形成通过资格门的事件级信号所见。</p>'
    )
    return metadata + events + link + (
        '<p class="source-boundary">本节复用已冻结的纯 EEG 信号分析；'
        "工作簿、EDF annotation 和医生标签均未进入检测、排序、Findings 或自动印象。</p>"
    )


def _impression_html(report: Mapping[str, Any]) -> str:
    doctor = report["physician_workbook_context"]
    annotations = report["edf_annotation_context"]
    signal = report["automatic_signal_context"]
    status = doctor["status"]
    if status == "available":
        doctor_text = (
            f"可唯一绑定 {doctor['label_count']} 条医生来源标注；"
            "起始、显著电极和早期扩散详见上表。"
        )
    elif status == "source_conflict":
        doctor_text = "医生来源标注存在冲突，不形成单一工作簿定位结论。"
    elif status == "ambiguous_mapping":
        doctor_text = "医生来源标注映射不唯一，本报告对其弃权。"
    else:
        doctor_text = "未获得可绑定的医生工作簿标注。"

    annotation_text = (
        f"EDF 来源时间线包含 {annotations['marker_row_count']} 条事件点标记、"
        f"{annotations['behavior_row_count']} 条临床表现标注。"
        if annotations["included_closed_rows"]
        else "未形成可报告的 EDF 闭集事件时间线。"
    )
    if signal["status"] == "technical_unassessable":
        signal_text = str(signal.get("technical_conclusion_zh", "自动 EEG 信号分析技术不可评价。"))
    else:
        impression = signal["automatic_impression"]
        signal_text = (
            str(impression["findings"])
            + " "
            + str(impression["diagnostic_conclusion"])
        )
    return (
        f"<p><strong>一、医生工作簿来源：</strong>{html_escape(doctor_text)}</p>"
        f"<p><strong>二、EDF annotation 来源：</strong>{html_escape(annotation_text)}</p>"
        f"<p><strong>三、自动 EEG 信号层：</strong>{html_escape(signal_text)}</p>"
        '<p class="warning"><strong>四、综合边界：</strong>'
        "三个来源层并列呈现，不把医生标注或 annotation 冒充为模型独立发现，"
        "也不用自动弃权覆盖医生来源记录。本流程未对全记录背景、睡眠分期、诱发试验"
        "和发作间期放电完成系统资格化分析，因此不作正常/异常总判定。</p>"
    )


def render_private_contextual_html(report: Mapping[str, Any], *, frozen_href: str) -> str:
    metadata = report["acquisition_metadata"]
    alias_text = (
        f"是（物理别名 {report['alias_index']:02d}）"
        if report["is_duplicate_signal_alias"]
        else "否"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>长程视频脑电临床综合报告（AI草稿）</title>
<style>
@page {{ size:A4; margin:15mm; }}
body {{ margin:0 auto; max-width:1080px; padding:26px; color:#20242b; font-family:"Noto Serif CJK SC","SimSun",serif; line-height:1.58; }}
h1 {{ text-align:center; margin:0; font-size:26px; }} h2 {{ border-top:3px solid #30353d; padding-top:8px; }}
.draft {{ text-align:center; color:#982f2f; font-weight:700; margin:6px 0 18px; }}
table {{ width:100%; border-collapse:collapse; table-layout:fixed; margin:10px 0 18px; }}
thead {{ display:table-header-group; }} tr {{ break-inside:avoid; }} th,td {{ border:1px solid #4b515a; padding:7px; vertical-align:top; white-space:pre-line; overflow-wrap:anywhere; }}
th {{ background:#e8edf2; }} .metadata th:first-child,.signal-meta th:first-child {{ width:28%; }}
.doctor th:nth-child(1) {{ width:13%; }} .doctor th:nth-child(2) {{ width:31%; }}
.annotations th:nth-child(1) {{ width:18%; }} .annotations th:nth-child(2) {{ width:10%; }} .annotations th:nth-child(3) {{ width:22%; }}
.signal-events th:nth-child(1) {{ width:23%; }} .signal-events th:nth-child(2) {{ width:52%; }}
.warning {{ border:1px solid #aa7620; background:#fff7e6; padding:10px; }}
.source-boundary {{ border-left:4px solid #50728d; background:#eef5f8; padding:9px; }} .muted {{ color:#66717c; }}
.review {{ margin-top:26px; border:1px solid #9ba3ad; padding:12px; }}
a {{ color:#185a8d; }}
</style></head><body>
<h1>长程视频脑电临床综合报告</h1>
<div class="draft">AI 草稿 · 未经脑电医师签署 · 不得直接用于诊疗</div>
<h2>记录与技术信息</h2>
{_html_table("metadata", ("记录字段", "内容"), (
    ("文件级报告 ID", report["file_report_id"]),
    ("唯一信号记录 ID", report["recording_id"]),
    ("去标识患者 ID", report["patient_pseudonym"]),
    ("重复信号别名", alias_text),
    ("EDF 头读取状态", metadata["status"]),
    ("记录时长", _duration_text(metadata.get("recording_duration_seconds"))),
    ("信号通道数", metadata.get("signal_channel_count") if metadata.get("signal_channel_count") is not None else "无法可靠获取"),
    ("原始采样率", _frequency_text(metadata.get("sample_frequencies_hz"))),
))}
<p class="source-boundary">EDF 固定头中的姓名、设备患者字段和采集日期均未进入报告。仅凭通道标签不推断参考电极。</p>
<h2>脑电图表现</h2>
<h3>背景、睡眠、诱发与发作间期</h3>
<p class="muted">本流程未对这些栏目完成全记录系统资格化分析，不生成缺乏证据的阴性或正常结论。</p>
<h3>医生工作簿起始、显著与扩散标注</h3>
{_doctor_html(report["physician_workbook_context"])}
<h3>EDF annotation 临床表现与事件时间线</h3>
{_annotation_html(report["edf_annotation_context"])}
<h3>自动 EEG 信号所见（独立证据层）</h3>
{_signal_html(report["automatic_signal_context"], frozen_href)}
<h2>脑电图印象</h2>
{_impression_html(report)}
<p class="warning">最终结论须由脑电医师联合原始长程 EEG、视频和临床资料复核后签署。</p>
<div class="review"><strong>审核状态：</strong>AI 草稿<br><strong>审核医师：</strong>________________<br><strong>签署日期：</strong>________________</div>
</body></html>"""


def render_private_contextual_markdown(report: Mapping[str, Any], *, frozen_href: str) -> str:
    doctor = report["physician_workbook_context"]
    annotations = report["edf_annotation_context"]
    signal = report["automatic_signal_context"]
    lines = [
        "# 长程视频脑电临床综合报告",
        "",
        "> AI 草稿，未经脑电医师签署，不得直接用于诊疗。",
        "",
        "## 记录与技术信息",
        "",
        f"- 文件级报告 ID：{report['file_report_id']}",
        f"- 唯一信号记录 ID：{report['recording_id']}",
        f"- 去标识患者 ID：{report['patient_pseudonym']}",
        f"- 重复信号别名：{'是' if report['is_duplicate_signal_alias'] else '否'}",
        f"- EDF 头读取：{report['acquisition_metadata']['status']}",
        f"- 记录时长：{_duration_text(report['acquisition_metadata'].get('recording_duration_seconds'))}",
        "",
        "## 脑电图表现",
        "",
        "### 背景、睡眠、诱发与发作间期",
        "",
        "本流程未完成这些栏目的全记录系统资格化分析，不生成缺乏证据的阴性结论。",
        "",
        "### 医生工作簿起始、显著与扩散标注",
        "",
        f"绑定状态：{doctor['status_display_zh']}",
        "",
    ]
    for label in doctor["labels"]:
        cells = label["source_cells"]
        lines.extend(
            [
                f"- {label['source_event_slot']}{'（冲突变体）' if label['source_conflict_variant'] else ''}",
                f"  - 起始：{cells['onset_text'] or '未填写'}",
                f"  - 显著电极：{cells['significant_electrodes_text'] or '未填写'}",
                f"  - 早期扩散：{cells['early_spread_text'] or '未填写'}",
                f"  - 覆盖全导：{cells['all_channel_coverage_text'] or '未填写'}",
            ]
        )
    lines.extend(["", "### EDF annotation 临床表现与事件时间线", ""])
    if annotations["annotations"]:
        for item in annotations["annotations"]:
            lines.append(
                f"- {item['display_time']}：{'、'.join(item['annotation_type_display_zh'])}；"
                f"来源转录：{item['source_text_redacted']}"
            )
    else:
        lines.append("未形成可进入闭集时间线的 EDF annotation。")
    lines.extend(["", "### 自动 EEG 信号所见（独立证据层）", ""])
    if signal["status"] == "technical_unassessable":
        lines.append(str(signal.get("technical_conclusion_zh", "自动 EEG 分析技术不可评价。")))
    else:
        for item in signal["event_findings"]:
            lines.extend(
                [
                    f"- {item['event_and_interval']}",
                    f"  - {item['finding_text_zh']}",
                    f"  - 相关双极导联：{item['supporting_derivations']}",
                ]
            )
        if not signal["event_findings"]:
            lines.append("未形成通过资格门的事件级信号所见。")
    lines.extend(
        [
            "",
            f"[打开冻结 EEG-only 证据报告]({frozen_href})",
            "",
            "## 脑电图印象",
            "",
            "医生工作簿、EDF annotation 与自动 EEG 信号层均以来源归属形式并列呈现。",
            "本流程不把来源标注冒充为模型独立发现，也不用自动弃权覆盖医生记录。",
            "",
            "最终结论须由脑电医师联合原始长程 EEG、视频和临床资料复核后签署。",
            "",
            "审核医师：________________  ",
            "签署日期：________________",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


def _write_json(path: Path, value: object) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _set_private_permissions(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("private report tree must not contain symlinks")
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chmod(root, 0o700)


def _relative_href(from_path: Path, to_path: Path) -> str:
    return PurePosixPath(os.path.relpath(to_path, start=from_path.parent)).as_posix()


def _convert_docx(html_paths: Sequence[Path], docx_dir: Path) -> dict[str, Path]:
    executable = shutil.which("libreoffice")
    if executable is None:
        raise RuntimeError("LibreOffice is required for --docx")
    docx_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(docx_dir, 0o700)
    with tempfile.TemporaryDirectory(prefix="private-eeg-docx-") as temporary_name:
        temporary = Path(temporary_name)
        runtime = temporary / "runtime"
        config = temporary / "config"
        cache = temporary / "cache"
        profile = temporary / "profile"
        for directory in (runtime, config, cache, profile):
            directory.mkdir(mode=0o700)
        environment = os.environ.copy()
        environment.update(
            {
                "XDG_RUNTIME_DIR": str(runtime),
                "XDG_CONFIG_HOME": str(config),
                "XDG_CACHE_HOME": str(cache),
                "SAL_USE_VCLPLUGIN": "svp",
            }
        )
        command_prefix = [
            executable,
            "--headless",
            "--invisible",
            "--nodefault",
            "--nolockcheck",
            "--nologo",
            "--nofirststartwizard",
            f"-env:UserInstallation={profile.resolve().as_uri()}",
            "--infilter=HTML (StarWriter)",
            "--convert-to",
            "docx:Office Open XML Text",
            "--outdir",
            str(docx_dir),
        ]
        # LibreOffice may silently skip an input in very large one-shot
        # conversions.  Use bounded batches and retry any skipped file alone.
        for start in range(0, len(html_paths), 20):
            batch = html_paths[start : start + 20]
            result = subprocess.run(
                [*command_prefix, *[str(path) for path in batch]],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "LibreOffice DOCX conversion failed: "
                    + sanitize_private_text(result.stderr, maximum_length=400)
                )
            for html_path in batch:
                expected = docx_dir / f"{html_path.stem}.docx"
                if expected.is_file():
                    continue
                retry = subprocess.run(
                    [*command_prefix, str(html_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                if retry.returncode != 0 or not expected.is_file():
                    raise RuntimeError(
                        "LibreOffice DOCX conversion skipped one report: "
                        + html_path.stem
                    )
    converted: dict[str, Path] = {}
    for html_path in html_paths:
        docx = docx_dir / f"{html_path.stem}.docx"
        if not docx.is_file():
            raise RuntimeError(f"DOCX conversion did not create {html_path.stem}")
        os.chmod(docx, 0o600)
        converted[html_path.stem] = docx
    if len(converted) != len(html_paths):
        raise RuntimeError("DOCX conversion count is incomplete")
    return converted


def _render_index(rows: Sequence[Mapping[str, Any]], *, include_docx: bool) -> str:
    body_rows = []
    for row in rows:
        report_id = str(row["file_report_id"])
        links = [
            f'<a href="html/{html_escape(report_id, quote=True)}.html">HTML</a>',
            f'<a href="markdown/{html_escape(report_id, quote=True)}.md">Markdown</a>',
            f'<a href="json/{html_escape(report_id, quote=True)}.json">JSON</a>',
        ]
        if include_docx:
            links.append(
                f'<a href="docx/{html_escape(report_id, quote=True)}.docx">DOCX</a>'
            )
        body_rows.append(
            "<tr>"
            f"<td>{html_escape(report_id)}</td>"
            f"<td>{html_escape(str(row['recording_id']))}</td>"
            f"<td>{html_escape(str(row['patient_pseudonym']))}</td>"
            f"<td>{'是' if row['is_duplicate_signal_alias'] else '否'}</td>"
            f"<td>{html_escape(str(row['doctor_label_status_display_zh']))}</td>"
            f"<td>{int(row['annotation_closed_row_count'])}</td>"
            f"<td>{html_escape(str(row['automatic_signal_status']))}</td>"
            f"<td>{' / '.join(links)}</td>"
            "</tr>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>私有长程 EEG 临床综合报告索引</title>
<style>body{{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;margin:24px;color:#20242b}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #667;padding:6px;vertical-align:top}}th{{background:#e8edf2;position:sticky;top:0}}.warning{{background:#fff6e3;border:1px solid #ad7b20;padding:10px}}</style>
</head><body><h1>私有长程 EEG 临床综合报告索引</h1>
<p class="warning">AI 草稿，未经脑电医师签署，不得直接用于诊疗。源文件路径仅保存在受限的 source-map.csv 中。</p>
<p>共 {len(rows)} 份物理 EDF 文件级报告。</p>
<table><thead><tr><th>报告 ID</th><th>唯一信号 ID</th><th>患者伪名</th><th>别名</th><th>医生标注</th><th>annotation 闭集行</th><th>自动信号层</th><th>文件</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table></body></html>"""


def _readme_text(*, physical_count: int, unique_count: int, include_docx: bool) -> str:
    formats = "HTML / Markdown / JSON" + (" / DOCX" if include_docx else "")
    return f"""# 私有长程 EEG 临床综合报告

本目录共包含 {physical_count} 份物理 EDF 文件级报告，对应 {unique_count} 个唯一信号。
三份重复信号路径保留独立 alias 报告，但复用同一冻结信号证据。

每份报告严格区分：

1. 冻结的自动 EEG 信号所见；
2. 医生 XLS/XLSX 工作簿中的起始、显著电极、早期扩散与覆盖全导字段；
3. EDF annotation 中经保守闭集分类的事件点和临床表现。

冲突、多对多映射歧义、缺失标注和技术不可评价均保持显式状态，不自动补齐。

- 报告格式：{formats}
- `index.html`：去标识队列索引
- `manifest.json`：覆盖、来源哈希与产物收据
- `source-map.csv`：**受限映射表**，含私有源相对路径，不应对外发布

> 所有报告均为 AI 草稿，未经脑电医师签署，不得直接用于诊疗。
"""


def materialize_private_contextual_reports(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    inventory_path: Path = DEFAULT_INVENTORY,
    base_report_root: Path = DEFAULT_BASE_REPORT_ROOT,
    doctor_bundle_path: Path = DEFAULT_DOCTOR_BUNDLE,
    annotation_csv_path: Path = DEFAULT_ANNOTATIONS,
    workbook_paths: Sequence[Path] = DEFAULT_WORKBOOKS,
    output_root: Path = DEFAULT_OUTPUT,
    include_docx: bool = False,
    expected_physical_count: int | None = 144,
    expected_unique_signal_count: int | None = 141,
) -> dict[str, Any]:
    """Materialize one private source-attributed draft for every physical EDF."""

    destination = output_root.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    inventory_snapshot = sha256_file(inventory_path.resolve(strict=True))
    doctor_snapshot = sha256_file(doctor_bundle_path.resolve(strict=True))
    annotation_snapshot = sha256_file(annotation_csv_path.resolve(strict=True))
    inventory = _load_json(inventory_path)
    doctor_bundle = _load_json(doctor_bundle_path)
    if doctor_bundle.get("inventory_id") != inventory.get("inventory_id"):
        raise ValueError("doctor label bundle and inventory identity differ")
    if doctor_bundle.get("record_count") != len(inventory.get("records", [])):
        raise ValueError("doctor label bundle record count differs from inventory")

    raw_source_by_fingerprint, workbook_receipts = parse_doctor_workbook_cells(
        workbook_paths
    )
    selected_workbook_hashes = {item["workbook_sha256"] for item in workbook_receipts}
    receipted_workbook_hashes = {
        str(item["workbook_sha256"])
        for item in doctor_bundle.get("source_receipts", {}).get("workbooks", [])
    }
    if selected_workbook_hashes != receipted_workbook_hashes:
        raise ValueError("selected workbooks differ from the frozen doctor-label sources")

    annotations_by_relative, annotation_receipt = parse_annotation_csv(
        annotation_csv_path
    )
    if annotation_receipt["annotation_csv_sha256"] != annotation_snapshot:
        raise RuntimeError("annotation CSV snapshot changed while reading")
    roster = build_physical_file_roster(
        dataset_root=dataset_root,
        inventory=inventory,
        expected_physical_count=expected_physical_count,
        expected_unique_signal_count=expected_unique_signal_count,
    )
    doctor_records = {
        str(item["recording_id"]): item for item in doctor_bundle.get("records", [])
    }
    if len(doctor_records) != len(inventory["records"]):
        raise ValueError("doctor bundle recording IDs repeat or are incomplete")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    os.chmod(staging, 0o700)
    html_dir = staging / "html"
    markdown_dir = staging / "markdown"
    json_dir = staging / "json"
    for directory in (html_dir, markdown_dir, json_dir):
        directory.mkdir(mode=0o700)

    report_rows: list[dict[str, Any]] = []
    artifact_receipts: list[dict[str, Any]] = []
    source_map_rows: list[dict[str, Any]] = []
    html_paths: list[Path] = []
    try:
        for roster_row in roster:
            recording_id = str(roster_row["recording_id"])
            doctor_record = doctor_records.get(recording_id)
            if doctor_record is None:
                raise ValueError("physical EDF has no post-freeze doctor record")
            doctor_context = _doctor_context_view(
                doctor_record,
                raw_source_by_fingerprint,
            )
            annotation_context = deepcopy(
                annotations_by_relative.get(
                    str(roster_row["source_relative_path"]),
                    {
                        "source_file_sha256": annotation_snapshot,
                        "csv_rows": 0,
                        "valid_annotation_rows": 0,
                        "included_closed_rows": 0,
                        "excluded_or_unclassified_rows": 0,
                        "behavior_row_count": 0,
                        "marker_row_count": 0,
                        "annotations": [],
                    },
                )
            )
            acquisition = inspect_edf_header(Path(roster_row["source_path"]))
            automatic = _automatic_signal_context(
                base_report_root=base_report_root,
                doctor_record=doctor_record,
            )
            report = {
                "schema_version": SCHEMA_VERSION,
                "file_report_id": roster_row["file_report_id"],
                "recording_id": recording_id,
                "patient_pseudonym": roster_row["patient_pseudonym"],
                "source_signal_sha256": roster_row["source_signal_sha256"],
                "source_size_bytes": roster_row["source_size_bytes"],
                "source_relative_path_sha256": hashlib.sha256(
                    str(roster_row["source_relative_path"]).encode("utf-8")
                ).hexdigest(),
                "is_duplicate_signal_alias": not roster_row[
                    "is_canonical_inventory_path"
                ],
                "alias_index": roster_row["alias_index"],
                "acquisition_metadata": acquisition,
                "physician_workbook_context": doctor_context,
                "edf_annotation_context": annotation_context,
                "automatic_signal_context": {
                    key: value
                    for key, value in automatic.items()
                    if key != "frozen_report_html_path"
                },
                "claim_boundary": {
                    "ai_draft_not_physician_signed": True,
                    "doctor_workbook_is_source_attributed": True,
                    "edf_annotation_is_source_attributed": True,
                    "automatic_signal_layer_remains_eeg_only": True,
                    "background_sleep_activation_interictal_not_systematically_assessed": True,
                    "normal_abnormal_overall_diagnosis_not_automatically_generated": True,
                    "requires_original_video_eeg_physician_review": True,
                },
            }
            report_id = str(roster_row["file_report_id"])
            html_path = html_dir / f"{report_id}.html"
            markdown_path = markdown_dir / f"{report_id}.md"
            json_path = json_dir / f"{report_id}.json"
            frozen_path = Path(automatic["frozen_report_html_path"])
            frozen_href_html = _relative_href(html_path, frozen_path)
            frozen_href_markdown = _relative_href(markdown_path, frozen_path)
            _write_text(
                html_path,
                render_private_contextual_html(report, frozen_href=frozen_href_html),
            )
            _write_text(
                markdown_path,
                render_private_contextual_markdown(
                    report,
                    frozen_href=frozen_href_markdown,
                ),
            )
            _write_json(json_path, report)
            html_paths.append(html_path)
            report_rows.append(
                {
                    "file_report_id": report_id,
                    "recording_id": recording_id,
                    "patient_pseudonym": roster_row["patient_pseudonym"],
                    "is_duplicate_signal_alias": report[
                        "is_duplicate_signal_alias"
                    ],
                    "doctor_label_status": doctor_context["status"],
                    "doctor_label_status_display_zh": doctor_context[
                        "status_display_zh"
                    ],
                    "annotation_closed_row_count": annotation_context[
                        "included_closed_rows"
                    ],
                    "automatic_signal_status": automatic["status"],
                    "acquisition_status": acquisition["status"],
                }
            )
            artifact_receipts.append(
                {
                    "file_report_id": report_id,
                    "html_sha256": sha256_file(html_path),
                    "markdown_sha256": sha256_file(markdown_path),
                    "json_sha256": sha256_file(json_path),
                }
            )
            source_map_rows.append(
                {
                    "file_report_id": report_id,
                    "recording_id": recording_id,
                    "patient_pseudonym": roster_row["patient_pseudonym"],
                    "source_relative_path": roster_row["source_relative_path"],
                    "canonical_source_relative_path": roster_row[
                        "canonical_relative_path"
                    ],
                    "source_signal_sha256": roster_row["source_signal_sha256"],
                    "is_canonical_inventory_path": roster_row[
                        "is_canonical_inventory_path"
                    ],
                }
            )

        docx_paths: dict[str, Path] = {}
        if include_docx:
            docx_paths = _convert_docx(html_paths, staging / "docx")
            by_id = {item["file_report_id"]: item for item in artifact_receipts}
            for report_id, path in docx_paths.items():
                by_id[report_id]["docx_sha256"] = sha256_file(path)

        _write_text(
            staging / "index.html",
            _render_index(report_rows, include_docx=include_docx),
        )
        _write_text(
            staging / "README.md",
            _readme_text(
                physical_count=len(roster),
                unique_count=len(inventory["records"]),
                include_docx=include_docx,
            ),
        )
        source_map_path = staging / "source-map.csv"
        with source_map_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(source_map_rows[0]))
            writer.writeheader()
            writer.writerows(source_map_rows)
        os.chmod(source_map_path, 0o600)

        doctor_counts = Counter(item["doctor_label_status"] for item in report_rows)
        automatic_counts = Counter(item["automatic_signal_status"] for item in report_rows)
        acquisition_counts = Counter(item["acquisition_status"] for item in report_rows)
        discovered_workbooks = []
        selected_hashes = selected_workbook_hashes
        for workbook in sorted(
            path
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}
        ):
            workbook_sha = sha256_file(workbook)
            discovered_workbooks.append(
                {
                    "workbook_sha256": workbook_sha,
                    "selected_canonical_source": workbook_sha in selected_hashes,
                    "disposition": (
                        "selected_by_frozen_doctor_label_bundle"
                        if workbook_sha in selected_hashes
                        else "not_receipted_version_or_duplicate_copy"
                    ),
                    "source_path_released": False,
                }
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "completed",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_count": len(report_rows),
            "physical_edf_count": len(roster),
            "unique_signal_count": len(inventory["records"]),
            "duplicate_signal_alias_count": sum(
                bool(item["is_duplicate_signal_alias"]) for item in report_rows
            ),
            "subject_count": len({item["patient_pseudonym"] for item in report_rows}),
            "format_counts": {
                "html": len(report_rows),
                "markdown": len(report_rows),
                "json": len(report_rows),
                "docx": len(docx_paths),
            },
            "doctor_label_status_counts": dict(sorted(doctor_counts.items())),
            "automatic_signal_status_counts": dict(sorted(automatic_counts.items())),
            "acquisition_status_counts": dict(sorted(acquisition_counts.items())),
            "annotation_closed_row_count": sum(
                int(item["annotation_closed_row_count"]) for item in report_rows
            ),
            "source_receipts": {
                "inventory_sha256": inventory_snapshot,
                "inventory_id": inventory["inventory_id"],
                "doctor_label_bundle_sha256": doctor_snapshot,
                "doctor_label_release_id": doctor_bundle["label_release_id"],
                "annotation_csv": annotation_receipt,
                "canonical_workbooks": workbook_receipts,
                "workbook_inventory": discovered_workbooks,
                "frozen_base_report_root_path_released": False,
                "dataset_root_path_released": False,
            },
            "reports": report_rows,
            "artifact_receipts": artifact_receipts,
            "privacy_boundary": {
                "raw_source_paths_only_in_restricted_source_map": True,
                "raw_patient_names_in_reports_or_index": False,
                "edf_identity_header_fields_released": False,
                "unclassified_annotation_text_released": False,
                "file_mode": "0600",
                "directory_mode": "0700",
            },
            "claim_boundary": {
                "ai_draft_not_physician_signed": True,
                "not_for_direct_diagnosis_or_treatment": True,
                "source_layers_remain_attributed": True,
                "automatic_signal_layer_unchanged_and_eeg_only": True,
            },
        }
        _write_json(staging / "manifest.json", manifest)

        if sha256_file(inventory_path.resolve(strict=True)) != inventory_snapshot:
            raise RuntimeError("inventory changed while materializing reports")
        if sha256_file(doctor_bundle_path.resolve(strict=True)) != doctor_snapshot:
            raise RuntimeError("doctor label bundle changed while materializing reports")
        if sha256_file(annotation_csv_path.resolve(strict=True)) != annotation_snapshot:
            raise RuntimeError("annotation CSV changed while materializing reports")
        _set_private_permissions(staging)
        os.replace(staging, destination)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "DEFAULT_ANNOTATIONS",
    "DEFAULT_BASE_REPORT_ROOT",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_DOCTOR_BUNDLE",
    "DEFAULT_INVENTORY",
    "DEFAULT_OUTPUT",
    "DEFAULT_WORKBOOKS",
    "MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_physical_file_roster",
    "format_clock",
    "inspect_edf_header",
    "materialize_private_contextual_reports",
    "normalize_annotation_relative_path",
    "parse_annotation_csv",
    "parse_doctor_workbook_cells",
    "render_private_contextual_html",
    "render_private_contextual_markdown",
    "sanitize_private_text",
]
