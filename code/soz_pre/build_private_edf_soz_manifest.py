#!/usr/bin/env python3
"""Build a private EDF SOZ manifest from doctor labels and EDF annotations.

Inputs:
  - Doctor summary CSV (preferred) or two-row-header xlsx workbooks.
  - ``edf_annotations.csv`` extracted from all private EDF files.
  - Private EEG root used to keep EDF paths relative and auditable.

Output:
  A canonical event-level manifest compatible with the rest of
  ``code.soz_pre``. Doctor significant electrodes are strong SOZ labels;
  uncertain/diffuse descriptions can contribute weaker SOZ labels through
  fractional label masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from soz_pre.constants import CANONICAL_MANIFEST_FIELDS, HEMISPHERE_INDEX  # noqa: E402
from soz_pre.label_mapping import map_private_doctor_labels, vectors_to_manifest_fields  # noqa: E402
from soz_pre.utils import (  # noqa: E402
    clean_cell,
    normalize_hemisphere,
    normalize_patient_name,
    normalize_sz_id,
    parse_electrodes,
    parse_float,
    parse_sz_ids_from_stem,
    semicolon,
    write_csv_rows,
)


DEFAULT_EEG_ROOT = "/mnt/hd1/dyf/dataset/EEG"
DEFAULT_DOCTOR_SUMMARY = "/mnt/hd1/dyf/dataset/EEG/发作起始通道汇总.csv"
DEFAULT_EDF_ANNOTATIONS = "/mnt/hd1/dyf/dataset/EEG/edf_annotations.csv"
DEFAULT_OUTPUT = "outputs/soz_pre/private_edf_soz_manifest.csv"


@dataclass
class DoctorEvent:
    patient_name: str
    patient_key: str
    sz_id: str
    sex: str = ""
    age: str = ""
    hemisphere: str = ""
    source_name: str = ""
    onset_text: str = ""
    raw_significant: str = ""
    raw_spread: str = ""
    significant: List[str] = None
    spread: List[str] = None
    source_file: str = ""
    source_row: int = -1


def _load_flat_doctor_summary(path: Path) -> List[DoctorEvent]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    events: List[DoctorEvent] = []
    for row_idx, row in df.iterrows():
        patient_name = clean_cell(row.get("姓名"))
        patient_key = normalize_patient_name(patient_name)
        if not patient_key:
            continue
        hemisphere = normalize_hemisphere(row.get("致痫灶侧别"))
        for sz_num in range(1, 8):
            sz_id = f"SZ{sz_num}"
            sig_col = f"{sz_id}_显著电极"
            spread_col = f"{sz_id}_早期扩散"
            onset_col = f"{sz_id}_发作起始通道"
            if sig_col not in df.columns and spread_col not in df.columns and onset_col not in df.columns:
                continue
            raw_sig = clean_cell(row.get(sig_col))
            raw_spread = clean_cell(row.get(spread_col))
            onset_text = clean_cell(row.get(onset_col))
            if not (raw_sig or raw_spread or onset_text):
                continue
            events.append(
                DoctorEvent(
                    patient_name=patient_name,
                    patient_key=patient_key,
                    sz_id=sz_id,
                    sex=clean_cell(row.get("性别")),
                    age=clean_cell(row.get("年龄")),
                    hemisphere=hemisphere,
                    source_name=clean_cell(row.get("来源")),
                    onset_text=onset_text,
                    raw_significant=raw_sig,
                    raw_spread=raw_spread,
                    significant=parse_electrodes(raw_sig),
                    spread=parse_electrodes(raw_spread),
                    source_file=str(path),
                    source_row=int(row_idx) + 2,
                )
            )
    return events


def _iter_sz_groups_from_two_row_header(df: pd.DataFrame):
    header0 = [clean_cell(v) for v in df.iloc[0].tolist()]
    header1 = [clean_cell(v) for v in df.iloc[1].tolist()]
    current: Optional[str] = None
    groups: Dict[str, Dict[str, int]] = {}
    order: List[str] = []
    for idx, (top, sub) in enumerate(zip(header0, header1)):
        if re.fullmatch(r"(?i)SZ\d+", top):
            current = top.upper()
            groups.setdefault(current, {})
            if current not in order:
                order.append(current)
        if not current:
            continue
        if "起始" in sub:
            groups[current]["onset"] = idx
        elif "显著" in sub:
            groups[current]["significant"] = idx
        elif "扩散" in sub:
            groups[current]["spread"] = idx
        elif "覆盖" in sub:
            groups[current]["coverage"] = idx
    for key in order:
        yield key, groups[key]


def _load_doctor_xlsx(path: Path) -> List[DoctorEvent]:
    events: List[DoctorEvent] = []
    read_path = _doctor_excel_read_path(path)
    excel = pd.ExcelFile(read_path)
    for sheet in excel.sheet_names:
        df = pd.read_excel(read_path, sheet_name=sheet, header=None)
        if df.shape[0] < 3 or df.shape[1] < 8:
            continue
        groups = list(_iter_sz_groups_from_two_row_header(df))
        for row_idx in range(2, len(df)):
            patient_name = clean_cell(df.iat[row_idx, 0])
            patient_key = normalize_patient_name(patient_name)
            if not patient_key:
                continue
            hemisphere = normalize_hemisphere(df.iat[row_idx, 3] if df.shape[1] > 3 else "")
            for sz_id, cols in groups:
                onset_text = clean_cell(df.iat[row_idx, cols["onset"]]) if "onset" in cols else ""
                raw_sig = clean_cell(df.iat[row_idx, cols["significant"]]) if "significant" in cols else ""
                raw_spread = clean_cell(df.iat[row_idx, cols["spread"]]) if "spread" in cols else ""
                coverage = clean_cell(df.iat[row_idx, cols["coverage"]]) if "coverage" in cols else ""
                if not (onset_text or raw_sig or raw_spread or coverage):
                    continue
                if coverage == "是" and "弥漫性" not in raw_spread:
                    raw_spread = f"{raw_spread},弥漫性" if raw_spread else "弥漫性"
                events.append(
                    DoctorEvent(
                        patient_name=patient_name,
                        patient_key=patient_key,
                        sz_id=normalize_sz_id(sz_id),
                        sex=clean_cell(df.iat[row_idx, 1] if df.shape[1] > 1 else ""),
                        age=clean_cell(df.iat[row_idx, 2] if df.shape[1] > 2 else ""),
                        hemisphere=hemisphere,
                        source_name=path.stem,
                        onset_text=onset_text,
                        raw_significant=raw_sig,
                        raw_spread=raw_spread,
                        significant=parse_electrodes(raw_sig),
                        spread=parse_electrodes(raw_spread),
                        source_file=f"{path}:{sheet}",
                        source_row=int(row_idx) + 1,
                    )
                )
    return events


def _doctor_excel_read_path(path: Path) -> Path:
    """Return a pandas-readable Excel path, converting legacy .xls if needed."""

    try:
        pd.ExcelFile(path).close()
        return path
    except ImportError as exc:
        if path.suffix.lower() != ".xls" or "xlrd" not in str(exc).lower():
            raise

    cache_dir = Path("/tmp/soz_pre_xls_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path("/tmp/soz_pre_xls_runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(0o700)
    lo_home = Path("/tmp/soz_pre_libreoffice_home")
    lo_home.mkdir(parents=True, exist_ok=True)
    user_install = (lo_home / "user").resolve().as_uri()
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    out_path = cache_dir / f"{path.stem}_{digest}.xlsx"
    if out_path.is_file():
        return out_path

    env = os.environ.copy()
    env.update({
        "HOME": str(lo_home),
        "XDG_CONFIG_HOME": str(lo_home / "config"),
        "XDG_CACHE_HOME": str(lo_home / "cache"),
        "XDG_RUNTIME_DIR": str(runtime_dir),
        "SAL_USE_VCLPLUGIN": "svp",
    })
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--invisible",
            "--nodefault",
            "--nolockcheck",
            "--nologo",
            "--nofirststartwizard",
            f"-env:UserInstallation={user_install}",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(cache_dir),
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    converted = cache_dir / f"{path.stem}.xlsx"
    if result.returncode != 0 or not converted.is_file():
        raise RuntimeError(
            "Failed to convert legacy .xls with libreoffice. "
            f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )
    converted.replace(out_path)
    return out_path


def load_doctor_events(paths: Sequence[Path]) -> Tuple[List[DoctorEvent], List[str]]:
    events: List[DoctorEvent] = []
    warnings: List[str] = []
    seen: set[Tuple[str, str]] = set()
    for path in paths:
        if not path.is_file():
            warnings.append(f"doctor_file_missing,{path}")
            continue
        try:
            if path.suffix.lower() == ".csv":
                parsed = _load_flat_doctor_summary(path)
            elif path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
                parsed = _load_doctor_xlsx(path)
            else:
                warnings.append(f"doctor_file_unsupported,{path}")
                continue
        except Exception as exc:
            warnings.append(f"doctor_file_failed,{path},{exc}")
            continue
        for item in parsed:
            key = (item.patient_key, item.sz_id)
            if key in seen:
                warnings.append(f"duplicate_doctor_event,{item.patient_key},{item.sz_id},{path}")
                continue
            seen.add(key)
            events.append(item)
    return events, warnings


def _is_sz_marker(desc: str) -> bool:
    return bool(re.fullmatch(r"(?i)SZ\d*(?:[-_]\d+)?|SZ", desc.strip()))


def _marker_sz_id(desc: str) -> str:
    match = re.search(r"(?i)SZ\s*(\d+)", desc.strip())
    return f"SZ{int(match.group(1))}" if match else ""


def _is_eeg_sz(desc: str) -> bool:
    return bool(re.search(r"(?i)EEG\s*SZ", desc))


def _is_end(desc: str) -> bool:
    return bool(re.fullmatch(r"(?i)END", desc.strip()))


def load_edf_annotation_groups(path: Path) -> Tuple[Dict[Tuple[str, str], List[Dict[str, object]]], Dict[str, Dict[str, object]]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "patient_key" not in df.columns:
        df["patient_key"] = df["patient"].map(normalize_patient_name)
    df["desc"] = df["description"].fillna("").astype(str).str.strip()
    df["file_stem"] = df["file_name"].fillna("").astype(str).map(lambda x: Path(x).stem)

    by_key: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    meta: Dict[str, Dict[str, object]] = {}
    for edf_path, group in df.groupby("edf_path", sort=False):
        first = group.iloc[0]
        patient_key = normalize_patient_name(first.get("patient"))
        stem = clean_cell(first.get("file_stem"))
        sz_ids = set(parse_sz_ids_from_stem(stem))
        for desc in group["desc"].tolist():
            marker = _marker_sz_id(desc)
            if marker:
                sz_ids.add(marker)
        if not sz_ids:
            sz_ids = {normalize_sz_id(stem)}
        record = {
            "edf_path": clean_cell(edf_path),
            "subset": clean_cell(first.get("subset")),
            "patient": clean_cell(first.get("patient")),
            "patient_key": patient_key,
            "file_name": clean_cell(first.get("file_name")),
            "sfreq": parse_float(first.get("sfreq")),
            "n_channels": int(parse_float(first.get("n_channels"), 0)),
            "duration_sec": parse_float(first.get("duration_sec")),
            "rows": group.sort_values("onset_sec").to_dict("records"),
            "sz_ids": sorted(sz_ids),
        }
        meta[clean_cell(edf_path)] = record
        for sz_id in sorted(sz_ids):
            by_key[(patient_key, sz_id)].append(record)
    return by_key, meta


def infer_event_times(edf_record: Dict[str, object], sz_id: str, fallback_duration_sec: float) -> Dict[str, object]:
    rows = [
        row for row in edf_record.get("rows", [])
        if row.get("ann_idx") == row.get("ann_idx")
    ]
    marker_rows = [
        row for row in rows
        if _is_sz_marker(clean_cell(row.get("desc", row.get("description", ""))))
    ]
    exact_marker_rows = [
        row for row in marker_rows
        if _marker_sz_id(clean_cell(row.get("desc", row.get("description", "")))) == sz_id
    ]
    eeg_sz_rows = [
        row for row in rows
        if _is_eeg_sz(clean_cell(row.get("desc", row.get("description", ""))))
    ]
    end_rows = [
        row for row in rows
        if _is_end(clean_cell(row.get("desc", row.get("description", ""))))
    ]

    marker_source = "exact_sz_marker" if exact_marker_rows else "first_sz_marker"
    marker_row = (exact_marker_rows or marker_rows or eeg_sz_rows or [None])[0]
    t_event = parse_float(marker_row.get("onset_sec") if marker_row else None)

    eeg_candidates = [
        row for row in eeg_sz_rows
        if not np.isfinite(t_event) or parse_float(row.get("onset_sec")) >= max(0.0, t_event - 2.0)
    ]
    eeg_row = (eeg_candidates or eeg_sz_rows or [None])[0]
    t_eeg = parse_float(eeg_row.get("onset_sec") if eeg_row else None)
    if not np.isfinite(t_eeg):
        t_eeg = t_event

    start = t_eeg if np.isfinite(t_eeg) else t_event
    end_candidates = [
        row for row in end_rows
        if np.isfinite(start) and parse_float(row.get("onset_sec")) > start
    ]
    end_row = (end_candidates or [None])[0]
    t_end = parse_float(end_row.get("onset_sec") if end_row else None)

    quality_flags: List[str] = []
    if len(marker_rows) > 1:
        quality_flags.append("multi_sz_markers_in_file")
    if len(end_rows) > 1:
        quality_flags.append("multi_end_markers_in_file")
    if not np.isfinite(t_event):
        quality_flags.append("missing_sz_marker")
    if eeg_row is None:
        quality_flags.append("missing_eeg_sz_marker")
    if not np.isfinite(t_end):
        quality_flags.append("missing_end_marker")
        duration = parse_float(edf_record.get("duration_sec"), 0.0)
        t_end = min(duration, start + float(fallback_duration_sec)) if np.isfinite(start) else float("nan")
    if np.isfinite(start) and np.isfinite(t_end) and t_end <= start:
        quality_flags.append("invalid_end_before_start")
    sz_duration = t_end - start if np.isfinite(start) and np.isfinite(t_end) else float("nan")
    if np.isfinite(sz_duration) and sz_duration > 300:
        quality_flags.append("suspicious_long_event")

    return {
        "t_event_marker": t_event,
        "t_eeg_onset": t_eeg,
        "t_end": t_end,
        "sz_start": start,
        "sz_end": t_end,
        "sz_duration": sz_duration,
        "time_source": marker_source,
        "quality_flags": quality_flags,
    }


def choose_edf_record(candidates: Sequence[Dict[str, object]], sz_id: str) -> Optional[Dict[str, object]]:
    if not candidates:
        return None
    exact = [
        item for item in candidates
        if sz_id in parse_sz_ids_from_stem(item.get("file_name", ""))
    ]
    return (exact or list(candidates))[0]


def build_private_manifest(
    doctor_events: Sequence[DoctorEvent],
    edf_by_key: Dict[Tuple[str, str], List[Dict[str, object]]],
    *,
    fallback_duration_sec: float,
    include_unmatched_doctor: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    stats = Counter()
    unmatched: List[Dict[str, str]] = []

    for event_idx, doctor in enumerate(doctor_events):
        stats["doctor_events"] += 1
        candidates = edf_by_key.get((doctor.patient_key, doctor.sz_id), [])
        edf_record = choose_edf_record(candidates, doctor.sz_id)
        if edf_record is None:
            stats["unmatched_doctor_events"] += 1
            unmatched.append({"patient": doctor.patient_key, "sz_id": doctor.sz_id})
            if not include_unmatched_doctor:
                continue
            edf_record = {
                "edf_path": "",
                "duration_sec": float("nan"),
                "sfreq": float("nan"),
                "n_channels": 0,
                "subset": "",
                "file_name": "",
                "rows": [],
            }

        time_info = infer_event_times(edf_record, doctor.sz_id, fallback_duration_sec=fallback_duration_sec)
        significant = doctor.significant or []
        spread = doctor.spread or []
        confidence = 1.0 if significant else 0.45
        mapped = map_private_doctor_labels(
            significant,
            spread,
            onset_text=doctor.onset_text,
            hemisphere=doctor.hemisphere,
            confidence=confidence,
        )
        label_confidence = float(mapped.get("label_confidence", confidence) or 0.0)
        spatial_loss_weight = float(mapped.get("spatial_loss_weight", confidence) or 0.0)
        quality_flags = list(time_info["quality_flags"]) + list(mapped.get("quality_flags", []))
        onset_lower = doctor.onset_text.lower()
        if "起始不清" in doctor.onset_text or "可疑" in doctor.onset_text or "?" in doctor.onset_text or "？" in doctor.onset_text:
            quality_flags.append("uncertain_onset_description")
        if not significant:
            quality_flags.append("no_doctor_significant_electrodes")
        review_status = "auto_accepted" if not quality_flags else "needs_human_review"
        if not edf_record.get("edf_path"):
            review_status = "failed"

        hemisphere = doctor.hemisphere
        hemisphere_label = HEMISPHERE_INDEX.get(hemisphere, "")
        raw_label_text = " | ".join(
            item for item in (
                f"onset={doctor.onset_text}" if doctor.onset_text else "",
                f"significant={doctor.raw_significant}" if doctor.raw_significant else "",
                f"spread={doctor.raw_spread}" if doctor.raw_spread else "",
            )
            if item
        )
        row: Dict[str, object] = {
            "source": "private",
            "split": "private",
            "patient_id": f"{doctor.patient_key}_{doctor.sz_id}",
            "base_patient_id": doctor.patient_key,
            "edf_path": edf_record.get("edf_path", ""),
            "event_id": f"{doctor.patient_key}_{doctor.sz_id}",
            "event_index": event_idx,
            "duration_sec": edf_record.get("duration_sec", ""),
            "t_event_marker": time_info["t_event_marker"],
            "t_eeg_onset": time_info["t_eeg_onset"],
            "t_end": time_info["t_end"],
            "sz_start": time_info["sz_start"],
            "sz_end": time_info["sz_end"],
            "sz_duration": time_info["sz_duration"],
            "seizure_type": "fnsz",
            "hemisphere": hemisphere,
            "hemisphere_label": hemisphere_label,
            "label_source": "doctor_significant_spread",
            "label_type": "clinical_soz",
            "label_confidence": label_confidence,
            "spatial_loss_weight": spatial_loss_weight,
            "raw_label_text": raw_label_text,
            "doctor_significant_electrodes": semicolon(significant),
            "doctor_spread_electrodes": semicolon(spread),
            "onset_channels": semicolon(significant),
            "soz_bipolar": ",".join(mapped.get("soz_bipolar", [])),
            "regions": semicolon(mapped.get("regions", [])),
            "propagation_regions": semicolon(mapped.get("propagation_regions", [])),
            "review_status": review_status,
            "quality_flags": semicolon(sorted(set(quality_flags))),
            "source_file": doctor.source_file,
            "doctor_source": doctor.source_name,
            "sex": doctor.sex,
            "age": doctor.age,
            "edf_sfreq": edf_record.get("sfreq", ""),
            "edf_n_channels": edf_record.get("n_channels", ""),
            "edf_subset": edf_record.get("subset", ""),
            "edf_file_name": edf_record.get("file_name", ""),
            "time_source": time_info["time_source"],
        }
        row.update(vectors_to_manifest_fields(mapped))
        rows.append(row)
        stats["matched_events"] += int(bool(edf_record.get("edf_path")))
        stats[f"review_{review_status}"] += 1
        for flag in set(quality_flags):
            stats[f"flag_{flag}"] += 1

    summary = {
        "stats": dict(stats),
        "n_rows": len(rows),
        "unmatched_doctor_events": unmatched[:100],
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build private EDF SOZ manifest from doctor labels and EDF annotations")
    parser.add_argument("--eeg_root", default=DEFAULT_EEG_ROOT)
    parser.add_argument("--doctor_summary", default=DEFAULT_DOCTOR_SUMMARY)
    parser.add_argument("--doctor_xlsx", action="append", default=[], help="Optional two-row-header xlsx doctor workbook")
    parser.add_argument("--edf_annotations", default=DEFAULT_EDF_ANNOTATIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--fallback_duration_sec", type=float, default=60.0)
    parser.add_argument("--include_unmatched_doctor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    doctor_paths = [Path(args.doctor_summary)] if args.doctor_summary else []
    doctor_paths.extend(Path(path) for path in args.doctor_xlsx)
    doctor_events, warnings = load_doctor_events(doctor_paths)
    edf_by_key, edf_meta = load_edf_annotation_groups(Path(args.edf_annotations))
    rows, summary = build_private_manifest(
        doctor_events,
        edf_by_key,
        fallback_duration_sec=args.fallback_duration_sec,
        include_unmatched_doctor=bool(args.include_unmatched_doctor),
    )
    summary.update({
        "eeg_root": str(args.eeg_root),
        "doctor_files": [str(path) for path in doctor_paths],
        "edf_annotations": str(args.edf_annotations),
        "warnings": warnings,
        "n_edf_files_in_annotations": len(edf_meta),
    })
    output = Path(args.output)
    write_csv_rows(output, rows, CANONICAL_MANIFEST_FIELDS)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} private events to {output}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
