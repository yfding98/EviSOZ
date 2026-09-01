"""Deterministic recovery of DeepSOZ records in a renamed local TUSZ tree.

The legacy DeepSOZ manifest uses numeric TUH patient names while TUSZ 2.0.3
uses opaque patient names.  The original conservative crosswalk establishes a
patient bijection from records whose complete seizure timelines are unique.
Once that bijection is established, the public record identity
``(split, patient, session, trial, montage, sample-count)`` is sufficient to
resolve records whose timelines are empty, revised, or non-unique.

This module never reads SOZ target columns and never uses model output.  It
only materializes identity/provenance artifacts; downstream signal and target
eligibility remain separate gates.
"""

from __future__ import annotations

import ast
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


IDENTITY_RECOVERY_SCHEMA = "deepsoz-tusz-identity-recovery-v2.0.0"
IDENTITY_RECOVERY_POLICY = (
    "607-timeline-unique-patient-bijection_then_within-patient-"
    "split-session-trial-montage-nsamples"
)
MAPPING_FILENAME = "mapping_identity_v2.csv"
AUDIT_FILENAME = "identity_recovery_audit.csv"
SUMMARY_FILENAME = "summary.json"

_OFFICIAL_SPLITS = frozenset({"train", "dev", "eval"})
_MONTAGE_RE = re.compile(r"\d{2}_tcp_(?:ar|le)(?:_a)?")
_DEEP_RECORD_RE = re.compile(r"[^/]+_(s\d{3})_(t\d{3})\.edf")
_LOCAL_PATIENT_RE = re.compile(r"[a-z0-9]+")
_LOCAL_SESSION_RE = re.compile(r"(s\d{3})_(\d{4})")
_SOURCE_SESSION_RE = re.compile(r"(s\d{3})_(\d{4})_\d{2}_\d{2}")
_MAPPING_COLUMNS = (
    "deepsoz_row",
    "deepsoz_patient",
    "deepsoz_record",
    "local_patient",
    "local_csv_bi",
    "local_edf",
    "max_time_error_s",
    "candidate_count",
    "mapping_status",
    "candidate_local_csv_bi",
    "candidate_max_errors_s",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _normalize_patient(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("DeepSOZ patient identity cannot be empty")
    try:
        numeric = float(text)
    except ValueError:
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def _strict_number_list(value: object) -> tuple[float, ...]:
    try:
        parsed = ast.literal_eval(str(value).strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError("DeepSOZ seizure times must be a numeric list") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("DeepSOZ seizure times must be a numeric list")
    try:
        result = tuple(float(item) for item in parsed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("DeepSOZ seizure times must be finite numbers") from exc
    if any(not math.isfinite(item) for item in result):
        raise ValueError("DeepSOZ seizure times must be finite numbers")
    return result


def _annotation_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            line for line in handle if line.strip() and not line.startswith("#")
        )
        return list(reader)


def _term_seizure_intervals(path: Path) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for row in _annotation_rows(path):
        if str(row.get("channel", "")).strip().upper() != "TERM":
            continue
        if str(row.get("label", "")).strip().lower() != "seiz":
            continue
        try:
            start = float(row["start_time"])
            stop = float(row["stop_time"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid TERM seizure interval in {path}") from exc
        if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
            raise ValueError(f"Invalid TERM seizure interval in {path}")
        intervals.append((start, stop))
    return tuple(intervals)


def _source_identity(row: Mapping[str, object]) -> dict[str, str]:
    record = str(row.get("fn", "")).strip()
    match = _DEEP_RECORD_RE.fullmatch(record)
    if match is None:
        raise ValueError(f"Invalid DeepSOZ record identity: {record!r}")
    location = str(row.get("loc", "")).strip().replace("\\", "/")
    parts = tuple(part for part in location.split("/") if part)
    splits = tuple(part for part in parts if part in _OFFICIAL_SPLITS)
    montages = tuple(part for part in parts if _MONTAGE_RE.fullmatch(part))
    if len(splits) != 1 or len(montages) != 1:
        raise ValueError("DeepSOZ loc must encode one official split and montage")
    source_session = Path(location).parent.name
    session_match = _SOURCE_SESSION_RE.fullmatch(source_session)
    if session_match is None or session_match.group(1) != match.group(1):
        raise ValueError("DeepSOZ loc/session does not match the record basename")
    return {
        "record": record,
        "session": match.group(1),
        "trial": match.group(2),
        "record_key": f"{match.group(1)}_{match.group(2)}",
        "split": splits[0],
        "montage": montages[0],
        "session_year": f"{session_match.group(1)}_{session_match.group(2)}",
    }


def _local_identity(path: Path, root: Path) -> dict[str, str]:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or resolved.is_symlink():
        raise ValueError("Recovered TUSZ paths cannot be symlinks")
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Recovered TUSZ path escapes the declared root") from exc
    parts = relative.parts
    if len(parts) != 5 or parts[0] not in _OFFICIAL_SPLITS:
        raise ValueError("Recovered TUSZ path is not a canonical five-level EDF path")
    split, patient, session_dir, montage, filename = parts
    if _LOCAL_PATIENT_RE.fullmatch(patient) is None:
        raise ValueError("Recovered local patient identity is not canonical")
    session_match = _LOCAL_SESSION_RE.fullmatch(session_dir)
    record_match = re.fullmatch(
        rf"{re.escape(patient)}_(s\d{{3}})_(t\d{{3}})\.edf", filename
    )
    if (
        session_match is None
        or _MONTAGE_RE.fullmatch(montage) is None
        or record_match is None
        or session_match.group(1) != record_match.group(1)
    ):
        raise ValueError("Recovered TUSZ path has inconsistent session/record fields")
    return {
        "relative_path": relative.as_posix(),
        "split": split,
        "patient": patient,
        "session_year": session_dir,
        "montage": montage,
        "session": record_match.group(1),
        "trial": record_match.group(2),
        "record_key": f"{record_match.group(1)}_{record_match.group(2)}",
    }


def _default_sample_counts(path: Path) -> tuple[int, ...]:
    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - deployment dependency gate
        raise RuntimeError("pyedflib is required for DeepSOZ identity recovery") from exc
    reader = pyedflib.EdfReader(str(path))
    try:
        values = tuple(int(item) for item in reader.getNSamples())
    finally:
        reader.close()
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"EDF has invalid sample counts: {path}")
    return values


def derive_patient_bijection(
    source: pd.DataFrame,
    conservative_mapping: pd.DataFrame,
    *,
    expected_unique_rows: int | None = 607,
    expected_patients: int | None = 124,
) -> dict[str, str]:
    """Derive and validate the numeric-to-opaque patient bijection."""

    required_source = {"pt_id", "fn", "loc", "nsamples", "sz_starts", "sz_ends"}
    required_mapping = set(_MAPPING_COLUMNS)
    missing_source = sorted(required_source - set(source.columns))
    missing_mapping = sorted(required_mapping - set(conservative_mapping.columns))
    if missing_source or missing_mapping:
        raise ValueError(
            f"Identity inputs are incomplete; source={missing_source}, mapping={missing_mapping}"
        )
    if len(source) != len(conservative_mapping):
        raise ValueError("DeepSOZ source and mapping row counts differ")
    mapping = conservative_mapping.copy()
    mapping["deepsoz_row"] = pd.to_numeric(
        mapping["deepsoz_row"], errors="raise"
    ).astype(int)
    if set(mapping["deepsoz_row"]) != set(range(len(source))):
        raise ValueError("Conservative mapping does not cover every source row exactly once")
    if mapping["deepsoz_row"].duplicated().any():
        raise ValueError("Conservative mapping contains duplicate source rows")
    unique = mapping.loc[mapping["mapping_status"].eq("unique")].copy()
    if expected_unique_rows is not None and len(unique) != expected_unique_rows:
        raise ValueError(
            f"Expected {expected_unique_rows} conservative unique rows, got {len(unique)}"
        )
    source_patients = source["pt_id"].map(_normalize_patient)
    if expected_patients is not None and source_patients.nunique() != expected_patients:
        raise ValueError(
            f"Expected {expected_patients} source patients, got {source_patients.nunique()}"
        )
    source_by_row = source.copy().reset_index(drop=True)
    deep_to_local: dict[str, str] = {}
    local_to_deep: dict[str, str] = {}
    for row in unique.sort_values("deepsoz_row").to_dict("records"):
        index = int(row["deepsoz_row"])
        deep_patient = _normalize_patient(source_by_row.iloc[index]["pt_id"])
        declared_patient = _normalize_patient(row["deepsoz_patient"])
        local_patient = str(row["local_patient"]).strip()
        if declared_patient != deep_patient or not local_patient:
            raise ValueError("Conservative mapping patient fields disagree with source")
        previous_local = deep_to_local.setdefault(deep_patient, local_patient)
        previous_deep = local_to_deep.setdefault(local_patient, deep_patient)
        if previous_local != local_patient or previous_deep != deep_patient:
            raise ValueError("Conservative unique rows do not establish a patient bijection")
        source_identity = _source_identity(source_by_row.iloc[index])
        local_identity = _local_identity(Path(str(row["local_edf"])), Path(str(row["local_edf"])).parents[4])
        if source_identity["record_key"] != local_identity["record_key"]:
            raise ValueError("A conservative unique row changes the session/trial key")
    if set(deep_to_local) != set(source_patients):
        raise ValueError("Not every DeepSOZ patient has a conservative identity anchor")
    if len(deep_to_local) != len(local_to_deep):
        raise ValueError("DeepSOZ/local patient mapping is not bijective")
    return dict(sorted(deep_to_local.items(), key=lambda item: int(item[0])))


def _timeline_audit(
    starts: Sequence[float],
    stops: Sequence[float],
    local: Sequence[tuple[float, float]],
    *,
    tolerance_sec: float,
) -> tuple[str, float | None]:
    if len(starts) != len(stops):
        raise ValueError("DeepSOZ seizure start/stop sequence lengths differ")
    if not starts and not local:
        return "both_empty", None
    if len(starts) != len(local):
        return "event_count_mismatch", None
    errors = [
        abs(float(start) - float(interval[0]))
        for start, interval in zip(starts, local)
    ]
    errors.extend(
        abs(float(stop) - float(interval[1]))
        for stop, interval in zip(stops, local)
    )
    maximum = max(errors)
    return (
        "exact_within_tolerance" if maximum <= tolerance_sec else "timing_drift",
        maximum,
    )


def build_identity_recovery_frames(
    source: pd.DataFrame,
    conservative_mapping: pd.DataFrame,
    tusz_root: str | Path,
    *,
    tolerance_sec: float = 0.25,
    expected_source_rows: int | None = 652,
    expected_unique_rows: int | None = 607,
    expected_patients: int | None = 124,
    sample_count_reader: Callable[[Path], Sequence[int]] = _default_sample_counts,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Resolve every source row and return mapping, audit, and summary payload."""

    root = Path(tusz_root).resolve(strict=True)
    if expected_source_rows is not None and len(source) != expected_source_rows:
        raise ValueError(f"Expected {expected_source_rows} source rows, got {len(source)}")
    bijection = derive_patient_bijection(
        source,
        conservative_mapping,
        expected_unique_rows=expected_unique_rows,
        expected_patients=expected_patients,
    )
    old = conservative_mapping.copy()
    old["deepsoz_row"] = pd.to_numeric(old["deepsoz_row"], errors="raise").astype(int)
    old = old.set_index("deepsoz_row", verify_integrity=True)
    output_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    resolved_paths: list[str] = []
    for deepsoz_row, source_row in source.reset_index(drop=True).iterrows():
        patient = _normalize_patient(source_row["pt_id"])
        local_patient = bijection[patient]
        source_identity = _source_identity(source_row)
        pattern = (
            f"*/{local_patient}/**/{local_patient}_"
            f"{source_identity['session']}_{source_identity['trial']}.edf"
        )
        candidates = tuple(sorted(root.glob(pattern), key=lambda path: path.as_posix()))
        if len(candidates) != 1:
            raise ValueError(
                f"Identity recovery for DeepSOZ row {deepsoz_row} found {len(candidates)} paths"
            )
        edf_path = candidates[0]
        local_identity = _local_identity(edf_path, root)
        evidence = {
            "split_match": source_identity["split"] == local_identity["split"],
            "session_year_match": (
                source_identity["session_year"] == local_identity["session_year"]
            ),
            "montage_match": source_identity["montage"] == local_identity["montage"],
            "record_key_match": (
                source_identity["record_key"] == local_identity["record_key"]
            ),
            "patient_binding_match": local_identity["patient"] == local_patient,
        }
        if not all(evidence.values()):
            raise ValueError(
                f"Identity evidence mismatch for DeepSOZ row {deepsoz_row}: {evidence}"
            )
        expected_samples = int(float(str(source_row["nsamples"]).strip()))
        sample_counts = tuple(int(value) for value in sample_count_reader(edf_path))
        nsamples_match = expected_samples in sample_counts
        if not nsamples_match:
            raise ValueError(f"EDF sample count mismatch for DeepSOZ row {deepsoz_row}")
        csv_path = edf_path.with_suffix(".csv")
        csv_bi_path = edf_path.with_suffix(".csv_bi")
        if not csv_path.is_file() or not csv_bi_path.is_file():
            raise FileNotFoundError(f"Recovered sidecars are missing for {edf_path}")
        starts = _strict_number_list(source_row["sz_starts"])
        stops = _strict_number_list(source_row["sz_ends"])
        local_intervals = _term_seizure_intervals(csv_bi_path)
        timeline_class, direct_error = _timeline_audit(
            starts, stops, local_intervals, tolerance_sec=tolerance_sec
        )
        old_row = old.loc[deepsoz_row]
        original_status = str(old_row["mapping_status"]).strip()
        if original_status not in {"unique", "ambiguous", "unmapped"}:
            raise ValueError("Conservative mapping contains an unknown status")
        if original_status == "unique":
            declared = Path(str(old_row["local_edf"])).resolve(strict=True)
            if declared != edf_path.resolve(strict=True):
                raise ValueError("Identity recovery changed a conservative unique mapping")
        resolved_paths.append(local_identity["relative_path"])
        direct_error_text = "" if direct_error is None else f"{direct_error:.12g}"
        output_rows.append(
            {
                "deepsoz_row": deepsoz_row,
                "deepsoz_patient": patient,
                "deepsoz_record": source_identity["record"],
                "local_patient": local_patient,
                "local_csv_bi": str(csv_bi_path.resolve(strict=True)),
                "local_edf": str(edf_path.resolve(strict=True)),
                "max_time_error_s": direct_error_text,
                "candidate_count": 1,
                "mapping_status": "unique",
                "candidate_local_csv_bi": str(csv_bi_path.resolve(strict=True)),
                "candidate_max_errors_s": direct_error_text,
            }
        )
        audit_rows.append(
            {
                "schema_version": IDENTITY_RECOVERY_SCHEMA,
                "policy": IDENTITY_RECOVERY_POLICY,
                "deepsoz_row": deepsoz_row,
                "deepsoz_patient": patient,
                "deepsoz_record": source_identity["record"],
                "original_mapping_status": original_status,
                "recovery_status": (
                    "conservative_unique_preserved"
                    if original_status == "unique"
                    else "identity_recovered"
                ),
                "local_patient": local_patient,
                "relative_edf_path": local_identity["relative_path"],
                "path_candidate_count": 1,
                **{key: int(value) for key, value in evidence.items()},
                "source_nsamples": expected_samples,
                "local_sample_count_values": ";".join(
                    str(value) for value in sorted(set(sample_counts))
                ),
                "nsamples_match": int(nsamples_match),
                "source_event_count": len(starts),
                "local_event_count": len(local_intervals),
                "timeline_class": timeline_class,
                "direct_timeline_max_error_sec": direct_error_text,
            }
        )
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Identity recovery maps multiple source rows to one local EDF")
    mapping = pd.DataFrame(output_rows, columns=_MAPPING_COLUMNS)
    audit = pd.DataFrame(audit_rows)
    status_counts = Counter(audit["original_mapping_status"])
    timeline_counts = Counter(audit["timeline_class"])
    recovered = audit.loc[audit["recovery_status"].eq("identity_recovered")]
    summary: dict[str, object] = {
        "schema_version": IDENTITY_RECOVERY_SCHEMA,
        "policy": IDENTITY_RECOVERY_POLICY,
        "source_rows": len(source),
        "source_patients": len(bijection),
        "patient_bijection_size": len(bijection),
        "original_mapping_status_counts": dict(sorted(status_counts.items())),
        "identity_unique_rows": len(mapping),
        "identity_unique_local_edfs": len(set(resolved_paths)),
        "recovered_row_count": len(recovered),
        "recovered_patient_count": recovered["deepsoz_patient"].nunique(),
        "recovered_rows_with_local_events": int((recovered["local_event_count"] > 0).sum()),
        "recovered_local_event_count": int(recovered["local_event_count"].sum()),
        "recovered_source_event_count": int(recovered["source_event_count"].sum()),
        "timeline_class_counts_all_rows": dict(sorted(timeline_counts.items())),
        "timeline_class_counts_recovered_rows": dict(
            sorted(Counter(recovered["timeline_class"]).items())
        ),
        "identity_evidence_counts": {
            field: int(audit[field].sum())
            for field in (
                "split_match",
                "session_year_match",
                "montage_match",
                "record_key_match",
                "patient_binding_match",
                "nsamples_match",
            )
        },
        "selection_policy": (
            "identity_only_no_soz_target_no_model_output_no_private_data"
        ),
    }
    return mapping, audit, summary


def materialize_identity_recovery(
    source_csv: str | Path,
    conservative_mapping_csv: str | Path,
    tusz_root: str | Path,
    output_directory: str | Path,
    *,
    tolerance_sec: float = 0.25,
) -> dict[str, object]:
    """Build the recovery artifact in a new directory using atomic publish."""

    source_path = Path(source_csv).resolve(strict=True)
    mapping_path = Path(conservative_mapping_csv).resolve(strict=True)
    root = Path(tusz_root).resolve(strict=True)
    output = Path(output_directory).resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    conservative = pd.read_csv(
        mapping_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    frames = build_identity_recovery_frames(
        source,
        conservative,
        root,
        tolerance_sec=tolerance_sec,
    )
    mapping, audit, summary = frames
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        mapping_output = temporary / MAPPING_FILENAME
        audit_output = temporary / AUDIT_FILENAME
        mapping.to_csv(mapping_output, index=False, encoding="utf-8")
        audit.to_csv(audit_output, index=False, encoding="utf-8")
        summary.update(
            {
                "inputs": {
                    "deepsoz_source": str(source_path),
                    "deepsoz_source_sha256": _sha256_file(source_path),
                    "conservative_mapping": str(mapping_path),
                    "conservative_mapping_sha256": _sha256_file(mapping_path),
                    "tusz_root": str(root),
                },
                "artifacts": {
                    MAPPING_FILENAME: _sha256_file(mapping_output),
                    AUDIT_FILENAME: _sha256_file(audit_output),
                },
            }
        )
        summary_output = temporary / SUMMARY_FILENAME
        with summary_output.open("xb") as handle:
            handle.write(_canonical_json_bytes(summary))
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(output):
            raise FileExistsError(output)
        os.rename(temporary, output)
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return summary


__all__ = [
    "AUDIT_FILENAME",
    "IDENTITY_RECOVERY_POLICY",
    "IDENTITY_RECOVERY_SCHEMA",
    "MAPPING_FILENAME",
    "SUMMARY_FILENAME",
    "build_identity_recovery_frames",
    "derive_patient_bijection",
    "materialize_identity_recovery",
]
