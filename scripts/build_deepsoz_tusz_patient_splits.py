#!/usr/bin/env python3
"""Build a leakage-aware patient-level DeepSOZ/TUSZ split package.

This builder deliberately separates three objects that older adapters mixed:

* the DeepSOZ-to-local-record crosswalk;
* one weak, clinical-note-derived target object per patient; and
* TUSZ seizure events that may be used to generate EEG evidence.

No monopolar label is expanded to a bipolar lead.  A zero, an unlisted
electrode, or a missing cell is never promoted to a confirmed negative.  The
duplicated upstream ``pz`` and ``pz.1`` fields are retained as separate audit
policies and the deployed PZ target remains unavailable until that schema is
resolved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEEPSOZ_MANIFEST = (
    ROOT / "outputs/deepsoz_llm_tusz_all_607_20260801/source/TUH_manifest_final.csv"
)
DEFAULT_MAPPING = ROOT / "outputs/deepsoz_tusz_adapted_manifest_20260803/mapping.csv"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT_DIR = ROOT / "outputs/deepsoz_tusz_patient_splits_v1"

SCHEMA_VERSION = "deepsoz_tusz_patient_split_positive_only_v1"
DEFAULT_FOLD_SEED = 20260806
DEFAULT_N_OOF_FOLDS = 5

# Deployed names use the modern 10-10 aliases.  DeepSOZ and TUSZ v2.0.3 use
# the legacy names T3/T4/T5/T6, so this is an identity alias only, never a
# spatial expansion.
CHANNEL_SOURCES: tuple[tuple[str, str], ...] = (
    ("FP1", "fp1"),
    ("FP2", "fp2"),
    ("F3", "f3"),
    ("F4", "f4"),
    ("C3", "c3"),
    ("C4", "c4"),
    ("P3", "p3"),
    ("P4", "p4"),
    ("O1", "o1"),
    ("O2", "o2"),
    ("F7", "f7"),
    ("F8", "f8"),
    ("T7", "t3"),
    ("T8", "t4"),
    ("P7", "t5"),
    ("P8", "t6"),
    ("FZ", "fz"),
    ("CZ", "cz"),
)
CANONICAL_CHANNELS: tuple[str, ...] = tuple(name for name, _ in CHANNEL_SOURCES) + ("PZ",)
OUTSIDE_HEAD_SOURCES: tuple[tuple[str, str], ...] = (
    ("OZ", "oz"),
    ("A1", "a1"),
    ("A2", "a2"),
)
LEGACY_TO_CANONICAL = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

SEIZURE_LABELS = {
    "seiz", "fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz",
    "cnsz", "tcsz", "atsz", "mysz", "nesz",
}
EXPECTED_OUTPUT_FILES = (
    "record_crosswalk.csv",
    "patient_targets.csv",
    "event_inputs.csv",
    "split_manifest.csv",
    "summary.json",
    "README.md",
)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalized_id(value: Any) -> str:
    text = clean_text(value)
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def numeric_positive(value: Any) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(numeric) and float(numeric) > 0.0)


def is_missing(value: Any) -> bool:
    return bool(pd.isna(value) or clean_text(value) == "")


def source_split_from_path(value: Any) -> str:
    text = clean_text(value).replace("\\", "/")
    match = re.search(r"/edf/(train|dev|eval)/", text)
    if match:
        return match.group(1)
    parts = [part for part in text.split("/") if part]
    return next((part for part in parts if part in {"train", "dev", "eval"}), "")


def local_split_from_path(value: Any, tusz_root: Path | None = None) -> str:
    text = clean_text(value)
    if not text:
        return ""
    path = Path(text)
    if tusz_root is not None:
        try:
            rel = path.resolve().relative_to(tusz_root.resolve())
            return rel.parts[0] if rel.parts else ""
        except (OSError, ValueError):
            pass
    return source_split_from_path(text)


def _pz_positive(row: Mapping[str, Any], policy: str) -> bool:
    first = numeric_positive(row.get("pz"))
    second = numeric_positive(row.get("pz.1"))
    if policy == "first":
        return first
    if policy == "second":
        return second
    if policy == "or":
        return first or second
    if policy == "none":
        return False
    raise ValueError(f"Unknown PZ policy: {policy}")


def positive_set(row: Mapping[str, Any], pz_policy: str) -> tuple[str, ...]:
    """Return observed positives only; absence is not interpreted as negative."""

    result = {
        canonical
        for canonical, source in CHANNEL_SOURCES
        if numeric_positive(row.get(source))
    }
    if _pz_positive(row, pz_policy):
        result.add("PZ")
    return tuple(sorted(result))


def summarize_positive_state(values: Sequence[bool]) -> str:
    """Summarize positive evidence without assigning a negative class."""

    if values and all(values):
        return "positive"
    if any(values):
        return "variable"
    return "unknown"


def label_signatures(rows: pd.DataFrame, policy: str) -> tuple[str, ...]:
    signatures = {
        "+".join(positive_set(row, policy)) or "<no_observed_positive>"
        for row in rows.to_dict("records")
    }
    return tuple(sorted(signatures))


def _validate_source_columns(source: pd.DataFrame) -> None:
    required = {
        "pt_id", "fn", "loc", "nsz", "sz_starts", "sz_ends",
        *(column for _, column in CHANNEL_SOURCES),
        "pz", "pz.1", *(column for _, column in OUTSIDE_HEAD_SOURCES),
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"DeepSOZ source is missing required columns: {missing}")


def _mapping_with_source_index(source: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if "deepsoz_row" not in mapping:
        raise ValueError("Mapping must contain deepsoz_row")
    if mapping["deepsoz_row"].duplicated().any():
        raise ValueError("Mapping contains duplicate deepsoz_row values")
    expected = set(range(len(source)))
    actual = set(pd.to_numeric(mapping["deepsoz_row"], errors="raise").astype(int))
    if actual != expected:
        raise ValueError(
            f"Mapping/source row mismatch: missing={sorted(expected-actual)[:5]}, "
            f"extra={sorted(actual-expected)[:5]}"
        )
    result = mapping.copy()
    result["deepsoz_row"] = pd.to_numeric(result["deepsoz_row"], errors="raise").astype(int)
    return result


def build_patient_targets(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
    tusz_root: Path | None = None,
) -> pd.DataFrame:
    """Build one positive-only weak-target audit row per DeepSOZ patient."""

    _validate_source_columns(source)
    mapping = _mapping_with_source_index(source, mapping)
    source = source.copy().reset_index(drop=True)
    source["deepsoz_row"] = source.index.astype(int)
    source["deepsoz_patient_id"] = source["pt_id"].map(normalized_id)
    joined = source.merge(mapping, on="deepsoz_row", how="left", validate="one_to_one")
    unique = joined.loc[joined["mapping_status"].eq("unique")].copy()

    rows: list[dict[str, Any]] = []
    for patient_id, patient_rows in source.groupby("deepsoz_patient_id", sort=True):
        patient_map = unique.loc[unique["deepsoz_patient_id"].eq(patient_id)]
        local_patients = sorted({clean_text(v) for v in patient_map.get("local_patient", []) if clean_text(v)})
        local_edfs = [clean_text(v) for v in patient_map.get("local_edf", []) if clean_text(v)]
        splits = sorted({local_split_from_path(v, tusz_root) for v in local_edfs if local_split_from_path(v, tusz_root)})
        identity_ok = len(local_patients) == 1 and len(splits) == 1

        signatures = {
            policy: label_signatures(patient_rows, policy)
            for policy in ("first", "second", "or", "none")
        }
        n_sets = {policy: len(values) for policy, values in signatures.items()}
        primary_variable = n_sets["or"] > 1

        row: dict[str, Any] = {
            "source": "deepsoz_tusz_overlay",
            "deepsoz_patient_id": patient_id,
            "local_patient_id": local_patients[0] if len(local_patients) == 1 else "",
            "official_split": splits[0] if len(splits) == 1 else "",
            "source_record_count": int(len(patient_rows)),
            "unique_mapped_record_count": int(len(patient_map)),
            "identity_crosswalk_status": "unique_one_to_one" if identity_ok else "unresolved",
            "label_stability_primary": "variable" if primary_variable else "stable",
            "primary_stability_scope": "all_652_source_records_observed_positive_sets",
            "n_label_sets_pz_first": n_sets["first"],
            "n_label_sets_pz_second": n_sets["second"],
            "n_label_sets_pz_or": n_sets["or"],
            "n_label_sets_non_pz": n_sets["none"],
            "label_signatures_pz_first": " | ".join(signatures["first"]),
            "label_signatures_pz_second": " | ".join(signatures["second"]),
            "label_signatures_pz_or": " | ".join(signatures["or"]),
            "label_signatures_non_pz": " | ".join(signatures["none"]),
            "pz_first_state": summarize_positive_state(
                [numeric_positive(value) for value in patient_rows["pz"].tolist()]
            ),
            "pz_second_state": summarize_positive_state(
                [numeric_positive(value) for value in patient_rows["pz.1"].tolist()]
            ),
            "pz_or_state": summarize_positive_state(
                [
                    numeric_positive(record.get("pz")) or numeric_positive(record.get("pz.1"))
                    for record in patient_rows.to_dict("records")
                ]
            ),
            "target_state_PZ": "schema_ambiguous",
            "label_value_PZ": 0,
            "label_mask_PZ": 0,
            "bce_available_PZ": 0,
            "zero_indicator_semantics": "unverified_not_negative",
            "confirmed_negative_count": 0,
            "ordinary_bce_ready": 0,
            "patient_quarantine_reason": (
                "variable_label_across_all_source_records" if primary_variable
                else ("identity_or_split_unresolved" if not identity_ok else "")
            ),
        }

        for canonical, source_column in CHANNEL_SOURCES:
            values = [numeric_positive(value) for value in patient_rows[source_column].tolist()]
            state = summarize_positive_state(values)
            row[f"target_state_{canonical}"] = state
            row[f"positive_evidence_{canonical}"] = int(state == "positive")
            # This mask exposes only an invariant observed positive for
            # positive-only or PU research.  It is deliberately zero for an
            # unknown/variable coordinate and for every quarantined patient.
            usable_positive = bool(not primary_variable and state == "positive")
            row[f"label_value_{canonical}"] = int(usable_positive)
            row[f"label_mask_{canonical}"] = int(usable_positive)
            # Even an invariant zero is not a confirmed complement.  BCE stays
            # unavailable until authoritative DeepSOZ schema documentation is
            # pinned and audited.
            row[f"bce_available_{canonical}"] = 0
            row[f"source_missing_records_{canonical}"] = int(patient_rows[source_column].isna().sum())

        for canonical, source_column in OUTSIDE_HEAD_SOURCES:
            values = [numeric_positive(value) for value in patient_rows[source_column].tolist()]
            row[f"outside_head_state_{canonical}"] = summarize_positive_state(values)
            row[f"outside_head_positive_records_{canonical}"] = int(sum(values))
            row[f"source_missing_records_{canonical}"] = int(patient_rows[source_column].isna().sum())

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("deepsoz_patient_id", kind="stable").reset_index(drop=True)
    if result["deepsoz_patient_id"].duplicated().any():
        raise AssertionError("patient_targets is not one row per DeepSOZ patient")
    return result


def normalize_edf_channel_name(name: Any) -> str:
    text = clean_text(name).upper().replace(" ", "")
    if text.startswith("EEG"):
        text = text[3:]
    for suffix in ("-REF", "-LE", "-AR", "-AVG"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.strip("-")
    return LEGACY_TO_CANONICAL.get(text, text)


def read_edf_header(edf_path: Path) -> dict[str, Any]:
    """Read only EDF metadata; MNE is imported lazily for test isolation."""

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/deepsoz_patient_split_numba")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepsoz_patient_split_mpl")
    try:
        import mne  # pylint: disable=import-outside-toplevel

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        normalized = [normalize_edf_channel_name(name) for name in raw.ch_names]
        counts = Counter(normalized)
        raw_names = {
            channel: [raw_name for raw_name, normalized_name in zip(raw.ch_names, normalized) if normalized_name == channel]
            for channel in CANONICAL_CHANNELS
        }
        available = {channel: int(counts[channel] == 1) for channel in CANONICAL_CHANNELS}
        duplicate = sorted(channel for channel in CANONICAL_CHANNELS if counts[channel] > 1)
        missing = sorted(channel for channel, value in available.items() if not value)
        sfreq = float(raw.info["sfreq"])
        return {
            "header_read_ok": 1,
            "header_error": "",
            "sfreq_hz": sfreq,
            "edf_duration_sec": float((raw.n_times - 1) / sfreq),
            "n_raw_channels": int(len(raw.ch_names)),
            "missing_physical_channels": ";".join(missing),
            "duplicate_physical_channels": ";".join(duplicate),
            "full19_available": int(not missing and not duplicate),
            "canonical_channel_map_json": json.dumps(raw_names, ensure_ascii=False, sort_keys=True),
            **{f"signal_available_{channel}": value for channel, value in available.items()},
            **{f"raw_edf_name_{channel}": ";".join(raw_names[channel]) for channel in CANONICAL_CHANNELS},
        }
    except Exception as exc:  # metadata failures belong in the audit table
        return {
            "header_read_ok": 0,
            "header_error": f"{type(exc).__name__}: {exc}",
            "sfreq_hz": "",
            "edf_duration_sec": "",
            "n_raw_channels": "",
            "missing_physical_channels": ";".join(CANONICAL_CHANNELS),
            "duplicate_physical_channels": "",
            "full19_available": 0,
            "canonical_channel_map_json": "{}",
            **{f"signal_available_{channel}": 0 for channel in CANONICAL_CHANNELS},
            **{f"raw_edf_name_{channel}": "" for channel in CANONICAL_CHANNELS},
        }


def _blank_header() -> dict[str, Any]:
    return {
        "header_read_ok": 0,
        "header_error": "not_uniquely_mapped",
        "sfreq_hz": "",
        "edf_duration_sec": "",
        "n_raw_channels": "",
        "missing_physical_channels": "",
        "duplicate_physical_channels": "",
        "full19_available": 0,
        "canonical_channel_map_json": "{}",
        **{f"signal_available_{channel}": 0 for channel in CANONICAL_CHANNELS},
        **{f"raw_edf_name_{channel}": "" for channel in CANONICAL_CHANNELS},
    }


def _relative_or_text(path_text: str, root: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def build_record_crosswalk(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
    patient_targets: pd.DataFrame,
    tusz_root: Path,
    header_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a 652-row crosswalk and raw-signal geometry audit."""

    mapping = _mapping_with_source_index(source, mapping)
    source = source.copy().reset_index(drop=True)
    source["deepsoz_row"] = source.index.astype(int)
    target_lookup = patient_targets.set_index("deepsoz_patient_id")
    header_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for source_row, map_row in zip(
        source.to_dict("records"),
        mapping.sort_values("deepsoz_row").to_dict("records"),
    ):
        deepsoz_row = int(source_row["deepsoz_row"])
        if int(map_row["deepsoz_row"]) != deepsoz_row:
            raise AssertionError("Mapping order does not align with source rows")
        patient_id = normalized_id(source_row.get("pt_id"))
        status = clean_text(map_row.get("mapping_status"))
        local_edf = clean_text(map_row.get("local_edf"))
        local_csv_bi = clean_text(map_row.get("local_csv_bi"))
        if status == "unique" and local_edf:
            if local_edf not in header_cache:
                if header_metadata is not None and local_edf in header_metadata:
                    header_cache[local_edf] = dict(header_metadata[local_edf])
                else:
                    header_cache[local_edf] = read_edf_header(Path(local_edf))
            header = header_cache[local_edf]
        else:
            header = _blank_header()
        target = target_lookup.loc[patient_id]
        source_split = source_split_from_path(source_row.get("loc"))
        local_split = local_split_from_path(local_edf, tusz_root)
        local_csv = str(Path(local_edf).with_suffix(".csv")) if local_edf else ""
        rows.append(
            {
                "source": "deepsoz_tusz_overlay",
                "deepsoz_row": deepsoz_row,
                "deepsoz_patient_id": patient_id,
                "deepsoz_record": clean_text(source_row.get("fn")),
                "source_official_split": source_split,
                "source_event_count": int(pd.to_numeric(source_row.get("nsz"), errors="coerce") or 0),
                "mapping_status": status,
                "candidate_count": int(pd.to_numeric(map_row.get("candidate_count"), errors="coerce") or 0),
                "max_time_error_sec": map_row.get("max_time_error_s", ""),
                "local_patient_id": clean_text(map_row.get("local_patient")),
                "local_official_split": local_split,
                "split_agreement": int(bool(source_split and local_split and source_split == local_split)),
                "local_edf_path": _relative_or_text(local_edf, tusz_root),
                "local_csv_path": _relative_or_text(local_csv, tusz_root),
                "local_csv_bi_path": _relative_or_text(local_csv_bi, tusz_root),
                "local_edf_exists": int(bool(local_edf and Path(local_edf).is_file())),
                "local_csv_exists": int(bool(local_csv and Path(local_csv).is_file())),
                "local_csv_bi_exists": int(bool(local_csv_bi and Path(local_csv_bi).is_file())),
                "candidate_local_csv_bi": clean_text(map_row.get("candidate_local_csv_bi")),
                "candidate_max_errors_sec": clean_text(map_row.get("candidate_max_errors_s")),
                "patient_label_stability_primary": target["label_stability_primary"],
                "patient_quarantine_reason": target["patient_quarantine_reason"],
                **header,
            }
        )

    result = pd.DataFrame(rows).sort_values("deepsoz_row").reset_index(drop=True)
    if len(result) != len(source) or result["deepsoz_row"].duplicated().any():
        raise AssertionError("record_crosswalk must contain exactly one row per source record")
    return result


def annotation_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            line for line in handle if line.strip() and not line.startswith("#")
        )
        return list(reader)


def term_seizure_intervals(path: Path) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for row in annotation_rows(path):
        if clean_text(row.get("channel")).upper() != "TERM":
            continue
        if clean_text(row.get("label")).lower() != "seiz":
            continue
        try:
            start = float(row["start_time"])
            stop = float(row["stop_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(stop) and stop > start:
            intervals.append((start, stop))
    return intervals


def event_seizure_type(path: Path, start: float, stop: float) -> str:
    labels: list[str] = []
    for row in annotation_rows(path):
        label = clean_text(row.get("label")).lower()
        if label not in SEIZURE_LABELS:
            continue
        try:
            row_start = float(row["start_time"])
            row_stop = float(row["stop_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if start - 1e-6 <= row_start < stop and row_start < row_stop:
            labels.append(label)
    return Counter(labels).most_common(1)[0][0] if labels else ""


def _absolute_from_root(value: Any, root: Path) -> Path:
    path = Path(clean_text(value))
    return path if path.is_absolute() else root / path


def build_event_inputs(
    record_crosswalk: pd.DataFrame,
    patient_targets: pd.DataFrame,
    tusz_root: Path,
) -> pd.DataFrame:
    """Build target-free event rows from uniquely mapped local annotations."""

    target_ids = set(patient_targets["deepsoz_patient_id"].map(clean_text))
    rows: list[dict[str, Any]] = []
    unique_records = record_crosswalk.loc[record_crosswalk["mapping_status"].eq("unique")]
    for record in unique_records.to_dict("records"):
        edf_path = _absolute_from_root(record["local_edf_path"], tusz_root)
        csv_path = _absolute_from_root(record["local_csv_path"], tusz_root)
        csv_bi_path = _absolute_from_root(record["local_csv_bi_path"], tusz_root)
        intervals = term_seizure_intervals(csv_bi_path)
        patient_id = clean_text(record["deepsoz_patient_id"])
        if patient_id not in target_ids:
            raise KeyError(f"No patient target foreign key for DeepSOZ patient {patient_id}")
        duration_value = pd.to_numeric(record.get("edf_duration_sec"), errors="coerce")
        duration = float(duration_value) if pd.notna(duration_value) else math.nan
        full19 = bool(int(record.get("full19_available", 0) or 0))
        header_ok = bool(int(record.get("header_read_ok", 0) or 0))

        for event_index, (start, stop) in enumerate(intervals):
            full_window = bool(
                math.isfinite(duration) and start >= 12.0 and start + 48.0 <= duration + 1e-6
            )
            causal_warmup = bool(start >= 42.0)
            signal_input_eligible = bool(header_ok and full19 and full_window)
            warmup_signal_input_eligible = bool(signal_input_eligible and causal_warmup)
            reasons: list[str] = []
            if not header_ok:
                reasons.append("edf_header_unreadable")
            if header_ok and not full19:
                reasons.append("incomplete_19_physical_channels")
            if not full_window:
                reasons.append("window_minus12_plus48_out_of_bounds")
            event_type = event_seizure_type(csv_path, start, stop)
            rows.append(
                {
                    "source": "deepsoz_tusz_overlay",
                    "deepsoz_row": int(record["deepsoz_row"]),
                    "deepsoz_patient_id": patient_id,
                    "patient_target_key": patient_id,
                    "deepsoz_record": record["deepsoz_record"],
                    "local_patient_id": record["local_patient_id"],
                    "official_split": record["local_official_split"],
                    "event_id": f"{edf_path.stem}__ev{event_index:04d}",
                    "event_index": event_index,
                    "local_edf_path": record["local_edf_path"],
                    "local_csv_path": record["local_csv_path"],
                    "local_csv_bi_path": record["local_csv_bi_path"],
                    "t0_sec": start,
                    "t0_provenance": "tusz_csv_bi_TERM_seiz_start",
                    "seizure_end_sec": stop,
                    "seizure_duration_sec": stop - start,
                    "seizure_type": event_type,
                    "window_start_sec": start - 12.0,
                    "window_stop_sec": start + 48.0,
                    "edf_duration_sec": duration if math.isfinite(duration) else "",
                    "sfreq_hz": record.get("sfreq_hz", ""),
                    "header_read_ok": int(header_ok),
                    "full19_available": int(full19),
                    "missing_physical_channels": record.get("missing_physical_channels", ""),
                    "full_minus12_plus48_in_bounds": int(full_window),
                    "causal_warmup_30s_available": int(causal_warmup),
                    "signal_input_eligible": int(signal_input_eligible),
                    "warmup_signal_input_eligible": int(warmup_signal_input_eligible),
                    "fnsz_signal_input_eligible": int(signal_input_eligible and event_type == "fnsz"),
                    "fnsz_warmup_signal_input_eligible": int(
                        warmup_signal_input_eligible and event_type == "fnsz"
                    ),
                    "signal_quarantine_reasons": ";".join(reasons),
                    **{
                        f"signal_available_{channel}": record.get(f"signal_available_{channel}", 0)
                        for channel in CANONICAL_CHANNELS
                    },
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["official_split", "local_patient_id", "local_edf_path", "event_index"],
            kind="stable",
        ).reset_index(drop=True)
        if result.duplicated(["local_edf_path", "event_index"]).any():
            raise AssertionError("Duplicate local event rows were generated")
    prohibited = {"soz_bipolar", "onset_channels", "soz_region"}
    if prohibited.intersection(result.columns):
        raise AssertionError("Target-bearing fields leaked into event_inputs")
    return result


def assign_oof_folds(patient_ids: Iterable[str], n_folds: int, seed: int) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    ids = sorted(
        {clean_text(value) for value in patient_ids if clean_text(value)},
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    return {patient_id: index % n_folds for index, patient_id in enumerate(ids)}


def build_split_manifest(
    patient_targets: pd.DataFrame,
    event_inputs: pd.DataFrame,
    n_oof_folds: int = DEFAULT_N_OOF_FOLDS,
    fold_seed: int = DEFAULT_FOLD_SEED,
) -> pd.DataFrame:
    event_groups = event_inputs.groupby("deepsoz_patient_id") if not event_inputs.empty else None
    rows: list[dict[str, Any]] = []
    for target in patient_targets.to_dict("records"):
        patient_id = clean_text(target["deepsoz_patient_id"])
        if event_groups is not None and patient_id in event_groups.groups:
            events = event_groups.get_group(patient_id)
        else:
            events = pd.DataFrame(columns=event_inputs.columns)
        total_events = int(len(events))
        signal_events = int(pd.to_numeric(events.get("signal_input_eligible", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        warmup_signal_events = int(pd.to_numeric(events.get("warmup_signal_input_eligible", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        signal_fnsz = int(pd.to_numeric(events.get("fnsz_signal_input_eligible", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        warmup_signal_fnsz = int(pd.to_numeric(events.get("fnsz_warmup_signal_input_eligible", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        stable = target["label_stability_primary"] == "stable"
        identity_ok = target["identity_crosswalk_status"] == "unique_one_to_one"
        if not stable:
            cohort = "quarantine_variable_label"
        elif not identity_ok:
            cohort = "quarantine_identity_or_split"
        elif signal_events == 0:
            cohort = "quarantine_no_strict_input_event"
        else:
            cohort = "included_positive_only"
        official_split = clean_text(target["official_split"])
        model_split = f"source_{official_split}" if cohort == "included_positive_only" else "quarantine"
        rows.append(
            {
                "source": "deepsoz_tusz_overlay",
                "deepsoz_patient_id": patient_id,
                "local_patient_id": target["local_patient_id"],
                "official_split": official_split,
                "model_split": model_split,
                "cohort_status": cohort,
                "label_stability_primary": target["label_stability_primary"],
                "source_record_count": target["source_record_count"],
                "unique_mapped_record_count": target["unique_mapped_record_count"],
                "event_count": total_events,
                "signal_input_event_count": signal_events,
                "warmup_signal_input_event_count": warmup_signal_events,
                "primary_analysis_event_count": signal_events if stable and identity_ok else 0,
                "warmup_primary_analysis_event_count": warmup_signal_events if stable and identity_ok else 0,
                "strict_fnsz_event_count": signal_fnsz if stable and identity_ok else 0,
                "warmup_strict_fnsz_event_count": warmup_signal_fnsz if stable and identity_ok else 0,
                "concept_oof_fold": "",
                "oof_fold_scope": "official_train_included_patients_only",
                "ordinary_bce_ready": 0,
                "weak_supervision_blocker": "zero_and_unlisted_negative_semantics_unverified;pz_schema_unresolved",
            }
        )
    result = pd.DataFrame(rows)
    train_ids = result.loc[result["model_split"].eq("source_train"), "deepsoz_patient_id"]
    folds = assign_oof_folds(train_ids, n_oof_folds, fold_seed)
    result["concept_oof_fold"] = pd.array(
        [folds.get(patient_id, pd.NA) for patient_id in result["deepsoz_patient_id"]],
        dtype="Int64",
    )
    result["oof_n_folds"] = n_oof_folds
    result["oof_fold_seed"] = fold_seed
    result = result.sort_values(["official_split", "deepsoz_patient_id"], kind="stable").reset_index(drop=True)

    split_sets = {
        split: set(group["deepsoz_patient_id"])
        for split, group in result.loc[result["official_split"].isin(["train", "dev", "eval"])].groupby("official_split")
    }
    for left, left_ids in split_sets.items():
        for right, right_ids in split_sets.items():
            if left < right and left_ids.intersection(right_ids):
                raise AssertionError(f"Patient overlap between official {left} and {right}")
    if result["deepsoz_patient_id"].duplicated().any():
        raise AssertionError("split_manifest is not patient-unique")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_split_counts(frame: pd.DataFrame, field: str) -> dict[str, int]:
    if frame.empty:
        return {}
    return {str(key): int(value) for key, value in frame[field].value_counts().sort_index().items()}


def build_summary(
    source_path: Path,
    mapping_path: Path,
    source: pd.DataFrame,
    records: pd.DataFrame,
    targets: pd.DataFrame,
    events: pd.DataFrame,
    splits: pd.DataFrame,
    tusz_root: Path,
    n_oof_folds: int,
    fold_seed: int,
) -> dict[str, Any]:
    unique_records = records.loc[records["mapping_status"].eq("unique")]
    stable = targets.loc[targets["label_stability_primary"].eq("stable")]
    stable_ids = set(stable["deepsoz_patient_id"])
    included_ids = set(
        splits.loc[splits["cohort_status"].eq("included_positive_only"), "deepsoz_patient_id"]
    )
    stable_events = events.loc[events["deepsoz_patient_id"].isin(stable_ids)]
    signal_events = events.loc[events["signal_input_eligible"].eq(1)]
    strict_events = signal_events.loc[signal_events["deepsoz_patient_id"].isin(included_ids)]
    warmup_events = events.loc[
        events["warmup_signal_input_eligible"].eq(1)
        & events["deepsoz_patient_id"].isin(included_ids)
    ]
    strict_fnsz = strict_events.loc[strict_events["seizure_type"].eq("fnsz")]
    warmup_fnsz = warmup_events.loc[warmup_events["seizure_type"].eq("fnsz")]
    primary_variable = targets.loc[targets["label_stability_primary"].eq("variable")]

    def patient_split_counts(frame: pd.DataFrame, patient_col: str) -> dict[str, int]:
        return {
            str(split): int(group[patient_col].nunique())
            for split, group in frame.groupby("official_split")
        }

    positive_support = {
        channel: int((stable.get(f"target_state_{channel}") == "positive").sum())
        for channel in CANONICAL_CHANNELS
        if channel != "PZ"
    }
    positive_support["PZ"] = 0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/build_deepsoz_tusz_patient_splits.py",
        "deepsoz_manifest": str(source_path.resolve()),
        "deepsoz_manifest_sha256": sha256_file(source_path),
        "mapping_manifest": str(mapping_path.resolve()),
        "mapping_manifest_sha256": sha256_file(mapping_path),
        "tusz_root": str(tusz_root.resolve()),
        "source_records": int(len(source)),
        "source_patients": int(source["pt_id"].map(normalized_id).nunique()),
        "mapping_status_counts": _nested_split_counts(records, "mapping_status"),
        "unique_mapped_records": int(len(unique_records)),
        "unique_mapped_patients": int(unique_records["deepsoz_patient_id"].nunique()),
        "official_unique_record_counts": _nested_split_counts(unique_records, "local_official_split"),
        "official_event_counts": _nested_split_counts(events, "official_split"),
        "official_patient_counts": patient_split_counts(events, "deepsoz_patient_id"),
        "source_all_record_stability": {
            "primary_pz_or_stable_patients": int(len(stable)),
            "primary_pz_or_variable_patients": int(len(primary_variable)),
            "variable_deepsoz_patient_ids": primary_variable["deepsoz_patient_id"].tolist(),
            "pz_first_stable_patients": int((targets["n_label_sets_pz_first"] == 1).sum()),
            "pz_first_variable_patients": int((targets["n_label_sets_pz_first"] > 1).sum()),
            "pz_second_stable_patients": int((targets["n_label_sets_pz_second"] == 1).sum()),
            "pz_second_variable_patients": int((targets["n_label_sets_pz_second"] > 1).sum()),
            "pz_or_stable_patients": int((targets["n_label_sets_pz_or"] == 1).sum()),
            "pz_or_variable_patients": int((targets["n_label_sets_pz_or"] > 1).sum()),
            "non_pz_stable_patients": int((targets["n_label_sets_non_pz"] == 1).sum()),
            "non_pz_variable_patients": int((targets["n_label_sets_non_pz"] > 1).sum()),
            "pz_policy_patient_state_counts": {
                policy: _nested_split_counts(targets, f"pz_{policy}_state")
                for policy in ("first", "second", "or")
            },
        },
        "stable_patient_split_counts": {
            str(split): int(group["deepsoz_patient_id"].nunique())
            for split, group in stable.groupby("official_split")
        },
        "stable_only_without_signal_filter": {
            "events": int(len(stable_events)),
            "records": int(stable_events["local_edf_path"].nunique()),
            "patients": int(stable_events["deepsoz_patient_id"].nunique()),
            "event_split_counts": _nested_split_counts(stable_events, "official_split"),
            "patient_split_counts": patient_split_counts(stable_events, "deepsoz_patient_id"),
        },
        "signal_geometry": {
            "eligibility_level": "edf_header_channel_map_and_time_bounds_only",
            "full_window_payload_validation": "pending_downstream_preprocessing",
            "edf_headers_read_ok": int(unique_records["header_read_ok"].sum()),
            "records_full19": int(unique_records["full19_available"].sum()),
            "records_incomplete19": int((unique_records["full19_available"] == 0).sum()),
            "incomplete19_patients": int(
                unique_records.loc[unique_records["full19_available"].eq(0), "deepsoz_patient_id"].nunique()
            ),
            "sampling_rate_record_counts": {
                str(float(key)): int(value)
                for key, value in pd.to_numeric(unique_records["sfreq_hz"], errors="coerce").dropna().value_counts().sort_index().items()
            },
        },
        "signal_full19_full_window_without_target_filter": {
            "events": int(len(signal_events)),
            "records": int(signal_events["local_edf_path"].nunique()),
            "patients": int(signal_events["deepsoz_patient_id"].nunique()),
            "event_split_counts": _nested_split_counts(signal_events, "official_split"),
            "patient_split_counts": patient_split_counts(signal_events, "deepsoz_patient_id"),
        },
        "strict_stable_full19_full_window": {
            "events": int(len(strict_events)),
            "records": int(strict_events["local_edf_path"].nunique()),
            "patients": int(strict_events["deepsoz_patient_id"].nunique()),
            "event_split_counts": _nested_split_counts(strict_events, "official_split"),
            "patient_split_counts": patient_split_counts(strict_events, "deepsoz_patient_id"),
        },
        "strict_stable_full19_full_window_plus_causal_warmup30": {
            "events": int(len(warmup_events)),
            "records": int(warmup_events["local_edf_path"].nunique()),
            "patients": int(warmup_events["deepsoz_patient_id"].nunique()),
            "event_split_counts": _nested_split_counts(warmup_events, "official_split"),
            "patient_split_counts": patient_split_counts(warmup_events, "deepsoz_patient_id"),
        },
        "strict_fnsz_sensitivity": {
            "events": int(len(strict_fnsz)),
            "records": int(strict_fnsz["local_edf_path"].nunique()),
            "patients": int(strict_fnsz["deepsoz_patient_id"].nunique()),
            "event_split_counts": _nested_split_counts(strict_fnsz, "official_split"),
            "patient_split_counts": patient_split_counts(strict_fnsz, "deepsoz_patient_id"),
        },
        "warmup_strict_fnsz_sensitivity": {
            "events": int(len(warmup_fnsz)),
            "records": int(warmup_fnsz["local_edf_path"].nunique()),
            "patients": int(warmup_fnsz["deepsoz_patient_id"].nunique()),
            "event_split_counts": _nested_split_counts(warmup_fnsz, "official_split"),
            "patient_split_counts": patient_split_counts(warmup_fnsz, "deepsoz_patient_id"),
        },
        "included_model_split_patient_counts": _nested_split_counts(
            splits.loc[splits["cohort_status"].eq("included_positive_only")], "model_split"
        ),
        "concept_oof": {
            "n_folds": n_oof_folds,
            "seed": fold_seed,
            "method": "sha256_seeded_order_then_round_robin",
            "fold_patient_counts": {
                str(int(key)): int(value)
                for key, value in splits.loc[splits["model_split"].eq("source_train"), "concept_oof_fold"].value_counts().sort_index().items()
            },
        },
        "observed_positive_patient_support_stable_cohort": positive_support,
        "label_semantics": {
            "granularity": "one weak clinical-note-derived target per patient",
            "event_rows_do_not_repeat_targets": True,
            "zero_or_unlisted": "unknown_not_negative",
            "missing": "unknown_not_negative",
            "confirmed_negative_count": 0,
            "ordinary_bce_ready": False,
            "pz": "pz and pz.1 retained separately; canonical PZ masked pending schema resolution",
            "monopolar_to_bipolar_expansion": False,
            "outside_head": ["OZ", "A1", "A2"],
        },
        "split_policy": {
            "official_tusz_split_preserved": True,
            "patient_disjoint": True,
            "train": "weak-source fitting only after label-semantic gate passes",
            "dev": "source-specific choices/calibration only",
            "eval": "descriptive locked result; pretraining-exposed, not external validation",
        },
    }


def render_readme(summary: Mapping[str, Any]) -> str:
    stability = summary["source_all_record_stability"]
    signal = summary["signal_geometry"]
    stable_only = summary["stable_only_without_signal_filter"]
    signal_only = summary["signal_full19_full_window_without_target_filter"]
    strict = summary["strict_stable_full19_full_window"]
    warmup = summary["strict_stable_full19_full_window_plus_causal_warmup30"]
    fnsz = summary["strict_fnsz_sensitivity"]
    return f"""# DeepSOZ–TUSZ patient-level split package v1

This package separates record identity, patient-level weak targets, target-free
event inputs, and split membership. It does **not** convert a positive
monopolar electrode into positive bipolar leads.

## Frozen audit counts

- DeepSOZ source: {summary['source_records']} records / {summary['source_patients']} patients.
- Conservative crosswalk: {summary['unique_mapped_records']} unique records;
  mapping statuses `{json.dumps(summary['mapping_status_counts'], ensure_ascii=False)}`.
- Stability is evaluated across **all source records**, including records that
  could not be mapped locally: {stability['primary_pz_or_stable_patients']}
  stable patients and {stability['primary_pz_or_variable_patients']} quarantined
  variable-label patients under the conservative observed-positive PZ-OR
  primary audit policy. Excluding both ambiguous PZ fields is reported only as
  a sensitivity analysis ({stability['non_pz_stable_patients']} stable /
  {stability['non_pz_variable_patients']} variable), not as the primary cohort.
- EDF geometry: {signal['records_full19']} unique records contain all 19
  deployed physical channels; {signal['records_incomplete19']} do not. These
  are header/channel-map and time-bound checks; full-window finite-sample,
  calibration, and gap validation remains a downstream preprocessing gate.
- Stable-label layer before signal filtering: {stable_only['events']} events,
  {stable_only['records']} records, {stable_only['patients']} patients.
- Signal-only complete-19 + complete `[-12,+48)` layer before target joining:
  {signal_only['events']} events, {signal_only['records']} records,
  {signal_only['patients']} patients.
- Stable + complete-19 + complete `[-12,+48)` input: {strict['events']} events,
  {strict['records']} records, {strict['patients']} patients.
- The additional 30-second causal-filter warm-up requirement (`t0 >= 42 s`)
  leaves {warmup['events']} events, {warmup['records']} records, and
  {warmup['patients']} patients. This is a separate layer, not part of the
  complete-window definition.
- FNSZ-only sensitivity subset: {fnsz['events']} events, {fnsz['records']}
  records, {fnsz['patients']} patients.

## Files

- `record_crosswalk.csv`: all source records, mapping status, paths, header
  readability, sampling rate, physical-channel availability, and the exact
  raw-EDF-name-to-canonical-channel map.
- `patient_targets.csv`: exactly one target audit row per patient. Electrode
  states are `positive`, `unknown`, or `variable`; `label_value_*`/
  `label_mask_*` expose stable observed positives only, and PZ is
  schema-ambiguous with value/mask `0/0`.
- `event_inputs.csv`: local TUSZ seizure events and signal eligibility only.
  It contains only a patient-target foreign key, no repeated DeepSOZ spatial
  target. Full-window and causal-warm-up eligibility are separate fields.
- `split_manifest.csv`: official TUSZ train/dev/eval role, quarantine status,
  and deterministic patient-level OOF folds for included source-train patients.
- `summary.json`: counts, hashes, split rules, and semantic constraints.

## Label boundary

The electrode indicators are clinical-note-derived weak labels, not invasive
SOZ truth and not event-specific labels. A zero, an unlisted electrode, and a
missing cell remain `unknown`; none is a confirmed negative. Therefore all
`bce_available_*` fields are zero and ordinary BCE/ranking supervision is
blocked until authoritative source documentation establishes complete-vector
negative semantics. The duplicate upstream `pz` and `pz.1` fields are retained
as first/second/OR sensitivity columns; canonical PZ remains masked.

`OZ`, `A1`, and `A2` are audited but are outside the deployed 19-electrode
head. Only the identity aliases T3→T7, T4→T8, T5→P7, and T6→P8 are applied.

## Split and leakage rules

Official TUSZ train/dev/eval membership is preserved and patient-disjoint.
Only source-train patients may fit a weak-source reasoner; source-dev is for
source-specific choices/calibration and source-eval is descriptive only.
Train-event evidence must be generated patient-out-of-fold. A concept model
used for source dev/eval must not receive task-specific TUSZ supervision from
those patients. Because TUSZ was exposed during foundation pretraining and in
historical experiments, this package cannot support an external or untouched
validation claim.
"""


def write_package(
    output_dir: Path,
    records: pd.DataFrame,
    targets: pd.DataFrame,
    events: pd.DataFrame,
    splits: pd.DataFrame,
    summary: dict[str, Any],
    overwrite: bool,
) -> None:
    existing = [name for name in EXPECTED_OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output package already contains {existing}; pass --overwrite to replace only these files"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "record_crosswalk.csv": records,
        "patient_targets.csv": targets,
        "event_inputs.csv": events,
        "split_manifest.csv": splits,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
    readme_path = output_dir / "README.md"
    readme_path.write_text(render_readme(summary), encoding="utf-8")
    summary["artifact_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in (*frames.keys(), "README.md")
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_package(
    deepsoz_path: Path,
    mapping_path: Path,
    tusz_root: Path,
    output_dir: Path,
    n_oof_folds: int = DEFAULT_N_OOF_FOLDS,
    fold_seed: int = DEFAULT_FOLD_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = pd.read_csv(deepsoz_path)
    mapping = pd.read_csv(mapping_path)
    targets = build_patient_targets(source, mapping, tusz_root=tusz_root)
    records = build_record_crosswalk(source, mapping, targets, tusz_root=tusz_root)
    events = build_event_inputs(records, targets, tusz_root=tusz_root)
    splits = build_split_manifest(targets, events, n_oof_folds=n_oof_folds, fold_seed=fold_seed)
    summary = build_summary(
        deepsoz_path, mapping_path, source, records, targets, events, splits,
        tusz_root, n_oof_folds, fold_seed,
    )
    write_package(output_dir, records, targets, events, splits, summary, overwrite=overwrite)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build patient-level DeepSOZ/TUSZ targets, inputs, and official splits"
    )
    parser.add_argument("--deepsoz-manifest", type=Path, default=DEFAULT_DEEPSOZ_MANIFEST)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-oof-folds", type=int, default=DEFAULT_N_OOF_FOLDS)
    parser.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.deepsoz_manifest, "DeepSOZ manifest"), (args.mapping, "mapping")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not args.tusz_root.is_dir():
        raise NotADirectoryError(f"TUSZ root not found: {args.tusz_root}")
    summary = build_package(
        args.deepsoz_manifest,
        args.mapping,
        args.tusz_root,
        args.output_dir,
        n_oof_folds=args.n_oof_folds,
        fold_seed=args.fold_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote patient-level split package to {args.output_dir}")


if __name__ == "__main__":
    main()
