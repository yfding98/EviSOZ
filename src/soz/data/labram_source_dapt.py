"""Auditable, annotation-free source-only data path for LaBraM DAPT.

The module deliberately separates *identity discovery* from EEG sampling.
The historical TUSZ ictal index is read only to recover a patient roster and
the frozen 60-patient/433-row audit count.  Continuous windows are then built
by independently enumerating every EDF under those patient directories.  No
event sidecar, event identifier, onset, seizure class, SOZ target, or private
record is accepted by the manifest or loader.
"""

from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..geometry import STANDARD_19, normalize_electrode_name
from ..models.labram import bind_labram_record_positions
from ..preprocessing_arm_runtime import _apply_car
from .edf import CausalEDFConfig, causal_bandpass_resample


SOURCE_DAPT_SCHEMA = "soz_labram_source_only_continuous_dapt_v1"
SOURCE_DAPT_SPLIT_POLICY = "sha256_seeded_patient_disjoint_48_train_12_dev_v1"
SOURCE_DAPT_GRID_POLICY = "record_zero_anchored_8s_grid_after_30s_causal_warmup_v1"
SOURCE_DAPT_PREPROCESSING = "C-CAR19_causal_0.5-45Hz_200Hz_real30s_warmup_v1"
SOURCE_DAPT_PURPOSE = "target_free_official_labram_masked_neural_code_dapt_only"

EXPECTED_DEEPSOZ_PATIENTS = 124
EXPECTED_MASTER_PATIENTS = 129
EXPECTED_MASTER_ROWS = 1519
EXPECTED_SOURCE_PATIENTS = 60
EXPECTED_SOURCE_ROWS = 433
EXPECTED_OVERLAPPING_MASTER_PATIENTS = 69
EXPECTED_TRAIN_PATIENTS = 48
EXPECTED_DEV_PATIENTS = 12
EXPECTED_SCANNED_EDFS = 839

WINDOW_SECONDS = 8
STRIDE_SECONDS = 8
CAUSAL_WARMUP_SECONDS = 30
OUTPUT_SFREQ_HZ = 200
SAMPLES_PER_TOKEN = 200
TOKENS_PER_CHANNEL = 8

_PATIENT_RE = re.compile(r"^[a-z0-9]+$")
_SUPPORTED_UNIT_SCALES = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
_FORBIDDEN_KEY_FRAGMENTS = (
    "event_id",
    "onset",
    "seizure_type",
    "seizure_start",
    "seizure_stop",
    "soz_label",
    "soz_target",
    "significant_electrode",
    "spread_electrode",
    "private_label",
)
_ANNOTATION_SUFFIXES = (
    ".csv",
    ".csv_bi",
    ".tse",
    ".tse_bi",
    ".lbl",
    ".lbl_bi",
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_file_identity(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"File changed while hashing: {path}")
    path_state = os.stat(path, follow_symlinks=False)
    if (path_state.st_dev, path_state.st_ino) != (after.st_dev, after.st_ino):
        raise RuntimeError(f"File path changed inode while hashing: {path}")
    return digest.hexdigest(), identity_after


def _sha256_file(path: Path) -> str:
    return _sha256_file_identity(path)[0]


def _safe_patient_id(value: object) -> str:
    patient = str(value).strip().lower()
    if not patient or _PATIENT_RE.fullmatch(patient) is None:
        raise ValueError(f"Invalid local TUSZ patient identity: {value!r}")
    return patient


def _unit_key(value: object) -> str:
    return str(value).strip().lower().replace("µ", "u").replace("μ", "u")


def _reference_suffix(value: object) -> str | None:
    text = str(value).strip().upper().replace("_", "-")
    for suffix in ("REF", "LE", "AR", "AVG", "AV", "CAR"):
        if text.endswith(f"-{suffix}"):
            return suffix
    return None


def _ceil_ratio(value: int, divisor: int) -> int:
    return int(
        (Decimal(value) / Decimal(divisor)).quantize(
            Decimal("1"), rounding=ROUND_CEILING
        )
    )


def _floor_nonnegative(value: float) -> int:
    return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_FLOOR))


def _assert_no_forbidden_keys(value: object, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key == "t0" or any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError(
                    f"Annotation/SOZ-derived manifest key is forbidden at {location}: {raw_key!r}"
                )
            _assert_no_forbidden_keys(child, location=f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, location=f"{location}[{index}]")


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys changed; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def load_deepsoz_patient_ids(path_value: str | Path) -> tuple[str, ...]:
    """Read only the local identity column from the frozen 124-patient roster."""

    path = Path(path_value).resolve(strict=True)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("DeepSOZ split roster is empty") from exc
        if "local_patient_id" not in header:
            raise ValueError("DeepSOZ split roster lacks local_patient_id")
        column = header.index("local_patient_id")
        patients = []
        for row in reader:
            if len(row) <= column:
                raise ValueError("DeepSOZ split roster contains a short row")
            patients.append(_safe_patient_id(row[column]))
    unique = tuple(sorted(set(patients)))
    if len(patients) != EXPECTED_DEEPSOZ_PATIENTS or len(unique) != EXPECTED_DEEPSOZ_PATIENTS:
        raise ValueError(
            "DeepSOZ exclusion roster must contain exactly 124 unique local identities"
        )
    return unique


@dataclass(frozen=True)
class SourcePoolAudit:
    master_patients: tuple[str, ...]
    source_patients: tuple[str, ...]
    master_row_count: int
    source_row_count: int
    overlapping_patient_count: int


def load_source_pool_audit(
    master_index_value: str | Path,
    *,
    excluded_patients: Sequence[str],
) -> SourcePoolAudit:
    """Use the old event index only for identity discovery and aggregate audit.

    No identifier, time, class, or record path from an index row is returned.
    Sampling is performed later by an independent patient-directory EDF scan.
    """

    path = Path(master_index_value).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("events"), list):
        raise TypeError("Historical TUSZ master index has an invalid structure")
    rows = payload["events"]
    patients: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or "patient_id" not in row:
            raise TypeError("Historical TUSZ master row lacks patient_id")
        patients.append(_safe_patient_id(row["patient_id"]))
    master = tuple(sorted(set(patients)))
    excluded = set(excluded_patients)
    source = tuple(patient for patient in master if patient not in excluded)
    source_rows = sum(patient not in excluded for patient in patients)
    overlap = len(set(master) & excluded)
    declared_patient_count = payload.get("patient_count")
    declared_row_count = payload.get("event_count")
    observed = (
        len(master),
        len(rows),
        len(source),
        source_rows,
        overlap,
    )
    expected = (
        EXPECTED_MASTER_PATIENTS,
        EXPECTED_MASTER_ROWS,
        EXPECTED_SOURCE_PATIENTS,
        EXPECTED_SOURCE_ROWS,
        EXPECTED_OVERLAPPING_MASTER_PATIENTS,
    )
    if observed != expected:
        raise ValueError(f"Frozen source-pool audit changed: expected={expected}, got={observed}")
    if declared_patient_count is not None and declared_patient_count != EXPECTED_MASTER_PATIENTS:
        raise ValueError("Historical TUSZ master index declared counts changed")
    if declared_row_count is not None and declared_row_count != EXPECTED_MASTER_ROWS:
        raise ValueError("Historical TUSZ master index declared counts changed")
    return SourcePoolAudit(
        master_patients=master,
        source_patients=source,
        master_row_count=len(rows),
        source_row_count=source_rows,
        overlapping_patient_count=overlap,
    )


def deterministic_patient_split(
    patients: Sequence[str], *, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized = tuple(sorted({_safe_patient_id(patient) for patient in patients}))
    if len(normalized) != EXPECTED_SOURCE_PATIENTS:
        raise ValueError("Source-only pretext split requires exactly 60 patients")
    ranked = sorted(
        normalized,
        key=lambda patient: hashlib.sha256(
            f"{int(seed)}\0{patient}".encode("ascii")
        ).hexdigest(),
    )
    dev = tuple(sorted(ranked[:EXPECTED_DEV_PATIENTS]))
    train = tuple(sorted(ranked[EXPECTED_DEV_PATIENTS:]))
    if len(train) != EXPECTED_TRAIN_PATIENTS or set(train) & set(dev):
        raise RuntimeError("Patient-disjoint pretext split construction failed")
    return train, dev


def load_deepsoz_mapped_edf_paths(
    crosswalk_value: str | Path,
    *,
    tusz_root: str | Path,
) -> tuple[Path, ...]:
    """Read only mapping status and local EDF path from the DeepSOZ crosswalk."""

    crosswalk = Path(crosswalk_value).resolve(strict=True)
    root = Path(tusz_root).resolve(strict=True)
    with crosswalk.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("DeepSOZ record crosswalk is empty") from exc
        required = {"mapping_status", "local_edf_path"}
        if not required.issubset(header):
            raise ValueError("DeepSOZ crosswalk lacks mapping/path columns")
        status_column = header.index("mapping_status")
        path_column = header.index("local_edf_path")
        paths: list[Path] = []
        for row in reader:
            if len(row) <= max(status_column, path_column):
                raise ValueError("DeepSOZ record crosswalk contains a short row")
            if row[status_column].strip() != "unique":
                continue
            relative = Path(row[path_column].strip())
            if relative.is_absolute() or relative.suffix.lower() != ".edf":
                raise ValueError("Mapped DeepSOZ path must be a relative EDF path")
            resolved = (root / relative).resolve(strict=True)
            if not resolved.is_relative_to(root) or resolved.is_symlink():
                raise ValueError("Mapped DeepSOZ EDF escapes the official TUSZ root")
            paths.append(resolved)
    unique = tuple(sorted(set(paths), key=str))
    if len(paths) != 607 or len(unique) != 607:
        raise ValueError("Expected exactly 607 unique locally mapped DeepSOZ EDFs")
    return unique


def build_deepsoz_content_exclusion(
    paths: Sequence[Path],
) -> tuple[tuple[str, ...], str, str]:
    """Hash mapped DeepSOZ EDFs so renamed/mirrored content is also excluded."""

    resolved = tuple(sorted((path.resolve(strict=True) for path in paths), key=str))
    hashes = tuple(sorted({_sha256_file(path) for path in resolved}))
    path_digest = hashlib.sha256(
        _canonical_json_bytes([str(path) for path in resolved])
    ).hexdigest()
    content_digest = hashlib.sha256(_canonical_json_bytes(list(hashes))).hexdigest()
    return hashes, path_digest, content_digest


def _record_header(path: Path) -> dict[str, object]:
    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyedflib is required for source DAPT manifest building") from exc
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        candidates: dict[str, list[int]] = {channel: [] for channel in STANDARD_19}
        for index, label in enumerate(labels):
            canonical = normalize_electrode_name(label)
            if canonical in candidates:
                candidates[canonical].append(index)
        missing = tuple(channel for channel, indices in candidates.items() if not indices)
        duplicate = tuple(channel for channel, indices in candidates.items() if len(indices) > 1)
        indices = tuple(
            candidates[channel][0] for channel in STANDARD_19 if len(candidates[channel]) == 1
        )
        raw_names: tuple[str, ...] = ()
        units: tuple[str, ...] = ()
        sfreqs: tuple[float, ...] = ()
        sample_counts: tuple[int, ...] = ()
        if not missing and not duplicate and len(indices) == 19:
            raw_names = tuple(labels[index] for index in indices)
            units = tuple(str(reader.getPhysicalDimension(index)).strip() for index in indices)
            sfreqs = tuple(float(reader.getSampleFrequency(index)) for index in indices)
            raw_counts = reader.getNSamples()
            sample_counts = tuple(int(raw_counts[index]) for index in indices)
        return {
            "raw_channel_names": raw_names,
            "raw_units": units,
            "source_sfreqs": sfreqs,
            "source_sample_counts": sample_counts,
            "missing": missing,
            "duplicate": duplicate,
        }
    finally:
        reader.close()


def _scan_one_record(
    path: Path,
    *,
    root: Path,
    patient: str,
    split: str,
    deepsoz_paths: frozenset[Path],
    deepsoz_content_hashes: frozenset[str],
) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Source EDF escapes the official TUSZ root")
    if path.is_symlink() or resolved.is_symlink():
        raise ValueError("Symlinked source EDFs are forbidden")
    relative = resolved.relative_to(root).as_posix()
    if any(relative.lower().endswith(suffix) for suffix in _ANNOTATION_SUFFIXES):
        raise ValueError("Annotation sidecars cannot enter source DAPT")
    digest = _sha256_file(resolved)
    header = _record_header(resolved)
    exclusions: list[str] = []
    if resolved in deepsoz_paths:
        exclusions.append("deepsoz_resolved_path_overlap")
    if digest in deepsoz_content_hashes:
        exclusions.append("deepsoz_content_hash_overlap")
    missing = tuple(header["missing"])
    duplicate = tuple(header["duplicate"])
    if missing:
        exclusions.append("missing_direct_standard19")
    if duplicate:
        exclusions.append("duplicate_direct_standard19")

    raw_names = tuple(header["raw_channel_names"])
    units = tuple(header["raw_units"])
    sfreqs = tuple(header["source_sfreqs"])
    sample_counts = tuple(header["source_sample_counts"])
    position_names: tuple[str, ...] = ()
    position_ids: tuple[int, ...] = ()
    source_reference = "unresolved"
    source_sfreq = 0.0
    sample_count = 0
    duration = 0.0
    first_grid_index = _ceil_ratio(CAUSAL_WARMUP_SECONDS, STRIDE_SECONDS)
    window_count = 0
    if raw_names:
        references = tuple(_reference_suffix(name) for name in raw_names)
        source_reference = references[0] if len(set(references)) == 1 else "mixed"
        if references != ("REF",) * 19:
            exclusions.append("source_reference_not_uniform_REF")
        if len(sfreqs) != 19 or any(not math.isfinite(value) or value <= 90.0 for value in sfreqs):
            exclusions.append("invalid_source_sampling_rate")
        elif any(abs(value - sfreqs[0]) > 1e-9 for value in sfreqs):
            exclusions.append("mixed_source_sampling_rate")
        else:
            source_sfreq = sfreqs[0]
        if len(sample_counts) != 19 or any(value <= 0 for value in sample_counts):
            exclusions.append("invalid_source_sample_count")
        elif len(set(sample_counts)) != 1:
            exclusions.append("mixed_source_sample_count")
        else:
            sample_count = sample_counts[0]
        if len(units) != 19 or any(_unit_key(unit) not in _SUPPORTED_UNIT_SCALES for unit in units):
            exclusions.append("unsupported_physical_unit")
        try:
            binding = bind_labram_record_positions(raw_names)
        except ValueError:
            exclusions.append("invalid_labram_header_position_binding")
        else:
            position_names = binding.position_names
            position_ids = binding.position_ids
        if source_sfreq > 0 and sample_count > 0:
            duration = sample_count / source_sfreq
            first_start = first_grid_index * STRIDE_SECONDS
            if duration + 1e-12 < first_start + WINDOW_SECONDS:
                exclusions.append("insufficient_continuous_duration")
            else:
                window_count = (
                    _floor_nonnegative(
                        (duration - first_start - WINDOW_SECONDS) / STRIDE_SECONDS
                    )
                    + 1
                )
    exclusions = sorted(set(exclusions))
    eligible = not exclusions
    if not eligible:
        window_count = 0
    return {
        "record_uid": hashlib.sha256(relative.encode("utf-8")).hexdigest(),
        "patient_id": patient,
        "pretext_split": split,
        "relative_edf_path": relative,
        "edf_sha256": digest,
        "file_size_bytes": resolved.stat().st_size,
        "source_sfreq_hz": source_sfreq,
        "source_sample_count": sample_count,
        "duration_seconds": duration,
        "raw_channel_names": list(raw_names),
        "raw_units": list(units),
        "labram_position_names": list(position_names),
        "labram_position_ids": list(position_ids),
        "source_reference": source_reference,
        "eligibility": "eligible" if eligible else "excluded",
        "exclusion_codes": exclusions,
        "first_grid_index": first_grid_index,
        "window_count": window_count,
    }


_TOP_KEYS = {
    "schema_version",
    "purpose",
    "identity_discovery_only",
    "target_values_loaded",
    "private_data_loaded",
    "annotation_sidecars_opened",
    "tusz_root",
    "source_pool_audit",
    "deepsoz_exclusion_contract",
    "pretext_split_contract",
    "continuous_grid_contract",
    "preprocessing_contract",
    "records",
    "counts",
}
_RECORD_KEYS = {
    "record_uid",
    "patient_id",
    "pretext_split",
    "relative_edf_path",
    "edf_sha256",
    "file_size_bytes",
    "source_sfreq_hz",
    "source_sample_count",
    "duration_seconds",
    "raw_channel_names",
    "raw_units",
    "labram_position_names",
    "labram_position_ids",
    "source_reference",
    "eligibility",
    "exclusion_codes",
    "first_grid_index",
    "window_count",
}


def build_source_dapt_manifest(
    *,
    tusz_root: str | Path,
    deepsoz_split_roster: str | Path,
    deepsoz_record_crosswalk: str | Path,
    historical_master_index: str | Path,
    split_seed: int = 20260811,
) -> dict[str, object]:
    """Build the complete 60-patient continuous-EDF source-only manifest."""

    root = Path(tusz_root).resolve(strict=True)
    deepsoz_ids = load_deepsoz_patient_ids(deepsoz_split_roster)
    pool = load_source_pool_audit(
        historical_master_index, excluded_patients=deepsoz_ids
    )
    train, dev = deterministic_patient_split(pool.source_patients, seed=split_seed)
    split_by_patient = {patient: "pretext_train" for patient in train}
    split_by_patient.update({patient: "pretext_dev" for patient in dev})

    deepsoz_edfs = load_deepsoz_mapped_edf_paths(
        deepsoz_record_crosswalk, tusz_root=root
    )
    content_hashes, deepsoz_path_digest, deepsoz_content_digest = (
        build_deepsoz_content_exclusion(deepsoz_edfs)
    )
    deepsoz_path_set = frozenset(deepsoz_edfs)
    deepsoz_hash_set = frozenset(content_hashes)

    records: list[dict[str, object]] = []
    discovered_paths: set[Path] = set()
    for patient in pool.source_patients:
        patient_root = (root / "train" / patient).resolve(strict=True)
        if not patient_root.is_dir() or not patient_root.is_relative_to(root):
            raise ValueError(f"Missing official-train patient directory: {patient}")
        paths = tuple(sorted(patient_root.rglob("*.edf"), key=str))
        if not paths:
            raise ValueError(f"Source-only patient has no EDF records: {patient}")
        for path in paths:
            resolved = path.resolve(strict=True)
            if resolved in discovered_paths:
                raise ValueError("One source EDF was discovered more than once")
            discovered_paths.add(resolved)
            records.append(
                _scan_one_record(
                    path,
                    root=root,
                    patient=patient,
                    split=split_by_patient[patient],
                    deepsoz_paths=deepsoz_path_set,
                    deepsoz_content_hashes=deepsoz_hash_set,
                )
            )
    records.sort(key=lambda row: str(row["relative_edf_path"]))
    eligible = [row for row in records if row["eligibility"] == "eligible"]
    excluded = [row for row in records if row["eligibility"] == "excluded"]
    path_overlap = sum(
        "deepsoz_resolved_path_overlap" in row["exclusion_codes"] for row in records
    )
    content_overlap = sum(
        "deepsoz_content_hash_overlap" in row["exclusion_codes"] for row in records
    )
    if path_overlap or content_overlap:
        raise ValueError(
            "Source-only patient directories contain a DeepSOZ path/content mirror"
        )
    train_windows = sum(
        int(row["window_count"])
        for row in eligible
        if row["pretext_split"] == "pretext_train"
    )
    dev_windows = sum(
        int(row["window_count"])
        for row in eligible
        if row["pretext_split"] == "pretext_dev"
    )
    manifest: dict[str, object] = {
        "schema_version": SOURCE_DAPT_SCHEMA,
        "purpose": SOURCE_DAPT_PURPOSE,
        "identity_discovery_only": True,
        "target_values_loaded": False,
        "private_data_loaded": False,
        "annotation_sidecars_opened": False,
        "tusz_root": str(root),
        "source_pool_audit": {
            "historical_index_sha256": _sha256_file(Path(historical_master_index).resolve(strict=True)),
            "master_patient_count": len(pool.master_patients),
            "master_index_row_count": pool.master_row_count,
            "deepsoz_overlap_patient_count": pool.overlapping_patient_count,
            "source_patient_count": len(pool.source_patients),
            "source_index_row_count": pool.source_row_count,
        },
        "deepsoz_exclusion_contract": {
            "split_roster_sha256": _sha256_file(Path(deepsoz_split_roster).resolve(strict=True)),
            "record_crosswalk_sha256": _sha256_file(Path(deepsoz_record_crosswalk).resolve(strict=True)),
            "excluded_patient_count": len(deepsoz_ids),
            "mapped_edf_count": len(deepsoz_edfs),
            "unique_content_count": len(content_hashes),
            "resolved_path_roster_sha256": deepsoz_path_digest,
            "content_sha256_roster_sha256": deepsoz_content_digest,
            "content_sha256s": list(content_hashes),
            "source_resolved_path_overlap_count": path_overlap,
            "source_content_overlap_count": content_overlap,
        },
        "pretext_split_contract": {
            "policy": SOURCE_DAPT_SPLIT_POLICY,
            "seed": int(split_seed),
            "train_patient_ids": list(train),
            "dev_patient_ids": list(dev),
        },
        "continuous_grid_contract": {
            "policy": SOURCE_DAPT_GRID_POLICY,
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "causal_warmup_seconds": CAUSAL_WARMUP_SECONDS,
            "record_zero_anchored": True,
            "annotation_time_used": False,
        },
        "preprocessing_contract": {
            "policy": SOURCE_DAPT_PREPROCESSING,
            "input_reference": "uniform_explicit_REF_only",
            "output_reference": "CAR19",
            "highpass_hz": 0.5,
            "lowpass_hz": 45.0,
            "output_sfreq_hz": OUTPUT_SFREQ_HZ,
            "input_unit_to_model_scale": "physical_volts_x_1e4",
            "standard_19": list(STANDARD_19),
        },
        "records": records,
        "counts": {
            "scanned_edf_count": len(records),
            "eligible_edf_count": len(eligible),
            "excluded_edf_count": len(excluded),
            "pretext_train_window_count": train_windows,
            "pretext_dev_window_count": dev_windows,
        },
    }
    validate_source_dapt_manifest_payload(manifest, excluded_patients=deepsoz_ids)
    return manifest


def write_source_dapt_manifest(payload: Mapping[str, object], path_value: str | Path) -> str:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing source DAPT manifest: {path}"
        )
    content = _canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(temporary, flags, 0o644)
        try:
            offset = 0
            while offset < len(content):
                written = os.write(descriptor, content[offset:])
                if written < 1:
                    raise OSError("Short write while publishing source DAPT manifest")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # A same-directory hard link is an atomic no-overwrite publication.
        # Unlike os.replace(), it cannot clobber a file/symlink created after
        # the initial exists() check.
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(content).hexdigest()


def validate_source_dapt_manifest_payload(
    payload: Mapping[str, object], *, excluded_patients: Sequence[str]
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("Source DAPT manifest must be a mapping")
    _require_exact_keys(payload, _TOP_KEYS, "source DAPT manifest")
    _assert_no_forbidden_keys(payload)
    if payload["schema_version"] != SOURCE_DAPT_SCHEMA or payload["purpose"] != SOURCE_DAPT_PURPOSE:
        raise ValueError("Source DAPT manifest identity changed")
    for flag, expected in (
        ("identity_discovery_only", True),
        ("target_values_loaded", False),
        ("private_data_loaded", False),
        ("annotation_sidecars_opened", False),
    ):
        if payload[flag] is not expected:
            raise ValueError(f"Source DAPT safety flag changed: {flag}")

    pool = payload["source_pool_audit"]
    if not isinstance(pool, Mapping):
        raise TypeError("source_pool_audit must be a mapping")
    expected_pool = {
        "historical_index_sha256",
        "master_patient_count",
        "master_index_row_count",
        "deepsoz_overlap_patient_count",
        "source_patient_count",
        "source_index_row_count",
    }
    _require_exact_keys(pool, expected_pool, "source_pool_audit")
    counts = (
        pool["master_patient_count"],
        pool["master_index_row_count"],
        pool["source_patient_count"],
        pool["source_index_row_count"],
        pool["deepsoz_overlap_patient_count"],
    )
    if counts != (
        EXPECTED_MASTER_PATIENTS,
        EXPECTED_MASTER_ROWS,
        EXPECTED_SOURCE_PATIENTS,
        EXPECTED_SOURCE_ROWS,
        EXPECTED_OVERLAPPING_MASTER_PATIENTS,
    ):
        raise ValueError("Source-pool 129/1519 -> 60/433 audit changed")

    split = payload["pretext_split_contract"]
    if not isinstance(split, Mapping):
        raise TypeError("pretext_split_contract must be a mapping")
    _require_exact_keys(
        split, {"policy", "seed", "train_patient_ids", "dev_patient_ids"}, "pretext split"
    )
    if split["policy"] != SOURCE_DAPT_SPLIT_POLICY:
        raise ValueError("Pretext split policy changed")
    train = tuple(_safe_patient_id(value) for value in split["train_patient_ids"])
    dev = tuple(_safe_patient_id(value) for value in split["dev_patient_ids"])
    if len(train) != EXPECTED_TRAIN_PATIENTS or len(dev) != EXPECTED_DEV_PATIENTS:
        raise ValueError("Pretext split must be 48 train / 12 dev patients")
    if len(set(train)) != len(train) or len(set(dev)) != len(dev) or set(train) & set(dev):
        raise ValueError("Pretext patient split is not unique and disjoint")
    excluded = {_safe_patient_id(value) for value in excluded_patients}
    if (set(train) | set(dev)) & excluded:
        raise ValueError("DeepSOZ identity entered source-only DAPT")

    exclusion = payload["deepsoz_exclusion_contract"]
    if not isinstance(exclusion, Mapping):
        raise TypeError("deepsoz_exclusion_contract must be a mapping")
    exclusion_keys = {
        "split_roster_sha256",
        "record_crosswalk_sha256",
        "excluded_patient_count",
        "mapped_edf_count",
        "unique_content_count",
        "resolved_path_roster_sha256",
        "content_sha256_roster_sha256",
        "content_sha256s",
        "source_resolved_path_overlap_count",
        "source_content_overlap_count",
    }
    _require_exact_keys(exclusion, exclusion_keys, "DeepSOZ exclusion contract")
    if exclusion["excluded_patient_count"] != EXPECTED_DEEPSOZ_PATIENTS:
        raise ValueError("DeepSOZ exclusion identity count changed")
    if exclusion["mapped_edf_count"] != 607:
        raise ValueError("DeepSOZ mapped EDF exclusion count changed")
    hashes = exclusion["content_sha256s"]
    if not isinstance(hashes, list) or len(hashes) != exclusion["unique_content_count"]:
        raise ValueError("DeepSOZ content exclusion hash roster is invalid")
    if hashes != sorted(set(hashes)) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes
    ):
        raise ValueError("DeepSOZ content exclusion hashes are not canonical")
    actual_content_digest = hashlib.sha256(_canonical_json_bytes(hashes)).hexdigest()
    if actual_content_digest != exclusion["content_sha256_roster_sha256"]:
        raise ValueError("DeepSOZ content exclusion roster digest changed")
    if exclusion["source_resolved_path_overlap_count"] != 0 or exclusion["source_content_overlap_count"] != 0:
        raise ValueError("DeepSOZ path/content overlap is non-zero")

    grid = payload["continuous_grid_contract"]
    if grid != {
        "policy": SOURCE_DAPT_GRID_POLICY,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "causal_warmup_seconds": CAUSAL_WARMUP_SECONDS,
        "record_zero_anchored": True,
        "annotation_time_used": False,
    }:
        raise ValueError("Continuous source-only grid contract changed")
    preprocessing = payload["preprocessing_contract"]
    if not isinstance(preprocessing, Mapping) or preprocessing.get("policy") != SOURCE_DAPT_PREPROCESSING:
        raise ValueError("Source DAPT preprocessing contract changed")
    if tuple(preprocessing.get("standard_19", ())) != STANDARD_19:
        raise ValueError("Source DAPT standard-19 order changed")

    rows = payload["records"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source DAPT manifest has no EDF records")
    seen_paths: set[str] = set()
    seen_uids: set[str] = set()
    patient_rows: set[str] = set()
    content_exclusion = set(hashes)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Source DAPT record must be a mapping")
        _require_exact_keys(row, _RECORD_KEYS, "source DAPT record")
        patient = _safe_patient_id(row["patient_id"])
        patient_rows.add(patient)
        expected_split = "pretext_train" if patient in train else "pretext_dev" if patient in dev else None
        if row["pretext_split"] != expected_split:
            raise ValueError("Record split disagrees with patient split")
        relative = Path(str(row["relative_edf_path"]))
        if (
            relative.is_absolute()
            or relative.suffix.lower() != ".edf"
            or ".." in relative.parts
            or len(relative.parts) < 3
            or relative.parts[0] != "train"
            or relative.parts[1] != patient
        ):
            raise ValueError("Source DAPT EDF path is not a safe relative path")
        path_text = relative.as_posix()
        if path_text in seen_paths or row["record_uid"] in seen_uids:
            raise ValueError("Source DAPT records are duplicated")
        seen_paths.add(path_text)
        seen_uids.add(str(row["record_uid"]))
        digest = row["edf_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Source EDF SHA-256 is invalid")
        if digest in content_exclusion:
            raise ValueError("Source EDF content overlaps a mapped DeepSOZ EDF")
        eligibility = row["eligibility"]
        codes = row["exclusion_codes"]
        if eligibility not in {"eligible", "excluded"} or not isinstance(codes, list):
            raise ValueError("Source EDF eligibility is invalid")
        if (eligibility == "eligible") != (codes == []):
            raise ValueError("Source EDF eligibility and exclusion codes disagree")
        if eligibility == "eligible":
            if (
                len(row["raw_channel_names"]) != 19
                or len(row["raw_units"]) != 19
                or len(row["labram_position_names"]) != 19
                or len(row["labram_position_ids"]) != 19
                or row["source_reference"] != "REF"
                or row["first_grid_index"] != 4
                or not isinstance(row["window_count"], int)
                or row["window_count"] < 1
            ):
                raise ValueError("Eligible source EDF violates the frozen signal/grid contract")
            binding = bind_labram_record_positions(row["raw_channel_names"])
            if list(binding.position_names) != row["labram_position_names"] or list(binding.position_ids) != row["labram_position_ids"]:
                raise ValueError("Source EDF LaBraM header binding changed")
        elif row["window_count"] != 0:
            raise ValueError("Excluded source EDF must expose zero scheduled windows")
    if patient_rows != set(train) | set(dev):
        raise ValueError("Source DAPT records do not cover every split patient")
    eligible_patient_rows = {
        str(row["patient_id"]) for row in rows if row["eligibility"] == "eligible"
    }
    if eligible_patient_rows != set(train) | set(dev):
        raise ValueError("Every pretext patient must contribute an eligible continuous window")

    declared = payload["counts"]
    if not isinstance(declared, Mapping):
        raise TypeError("Source DAPT counts must be a mapping")
    _require_exact_keys(
        declared,
        {
            "scanned_edf_count",
            "eligible_edf_count",
            "excluded_edf_count",
            "pretext_train_window_count",
            "pretext_dev_window_count",
        },
        "source DAPT counts",
    )
    eligible_rows = [row for row in rows if row["eligibility"] == "eligible"]
    expected_counts = {
        "scanned_edf_count": len(rows),
        "eligible_edf_count": len(eligible_rows),
        "excluded_edf_count": len(rows) - len(eligible_rows),
        "pretext_train_window_count": sum(
            row["window_count"] for row in eligible_rows if row["pretext_split"] == "pretext_train"
        ),
        "pretext_dev_window_count": sum(
            row["window_count"] for row in eligible_rows if row["pretext_split"] == "pretext_dev"
        ),
    }
    if dict(declared) != expected_counts:
        raise ValueError("Source DAPT aggregate record/window counts changed")
    if declared["scanned_edf_count"] != EXPECTED_SCANNED_EDFS:
        raise ValueError("Official-train source EDF inventory changed from 839 records")


@dataclass(frozen=True)
class LoadedSourceDAPTManifest:
    payload: Mapping[str, object]
    path: Path
    sha256: str
    tusz_root: Path
    excluded_patients: tuple[str, ...]

    @property
    def records(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self.payload["records"])


def load_source_dapt_manifest(
    path_value: str | Path,
    *,
    deepsoz_split_roster: str | Path,
    tusz_root: str | Path | None = None,
    verify_file_inventory: bool = True,
) -> LoadedSourceDAPTManifest:
    path = Path(path_value).resolve(strict=True)
    content = path.read_bytes()
    payload = json.loads(content)
    excluded = load_deepsoz_patient_ids(deepsoz_split_roster)
    validate_source_dapt_manifest_payload(payload, excluded_patients=excluded)
    actual_split_roster_sha256 = _sha256_file(
        Path(deepsoz_split_roster).resolve(strict=True)
    )
    declared_split_roster_sha256 = payload["deepsoz_exclusion_contract"][
        "split_roster_sha256"
    ]
    if actual_split_roster_sha256 != declared_split_roster_sha256:
        raise ValueError(
            "Current DeepSOZ exclusion roster differs from the manifest-bound roster"
        )
    root = Path(payload["tusz_root"] if tusz_root is None else tusz_root).resolve(strict=True)
    if root != Path(payload["tusz_root"]).resolve(strict=True):
        raise ValueError("Runtime TUSZ root differs from the frozen DAPT manifest")
    if verify_file_inventory:
        declared_paths = {str(row["relative_edf_path"]) for row in payload["records"]}
        current_paths: set[str] = set()
        source_patients = (
            list(payload["pretext_split_contract"]["train_patient_ids"])
            + list(payload["pretext_split_contract"]["dev_patient_ids"])
        )
        for patient in source_patients:
            patient_root = (root / "train" / patient).resolve(strict=True)
            for candidate in patient_root.rglob("*.edf"):
                resolved = candidate.resolve(strict=True)
                if candidate.is_symlink() or resolved.is_symlink() or not resolved.is_relative_to(root):
                    raise ValueError("Current source EDF inventory contains a symlink/escape")
                current_paths.add(resolved.relative_to(root).as_posix())
        if current_paths != declared_paths:
            raise ValueError("Current source EDF inventory differs from the frozen manifest")
        for row in payload["records"]:
            source = (root / str(row["relative_edf_path"])).resolve(strict=True)
            if source.stat().st_size != row["file_size_bytes"]:
                raise ValueError("Source EDF size changed after manifest construction")
    return LoadedSourceDAPTManifest(
        payload=payload,
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        tusz_root=root,
        excluded_patients=excluded,
    )


def _read_continuous_window(
    source: Path,
    row: Mapping[str, object],
    *,
    grid_index: int,
    expected_file_identity: tuple[int, int, int, int, int],
) -> np.ndarray:
    """Read one grid window plus real causal warmup; never open a sidecar."""

    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyedflib is required for source DAPT loading") from exc
    if grid_index < int(row["first_grid_index"]):
        raise ValueError("Grid window lacks the frozen causal warmup")
    start_sec = grid_index * STRIDE_SECONDS
    stop_sec = start_sec + WINDOW_SECONDS
    sfreq = float(row["source_sfreq_hz"])
    state_reset = int(math.floor((start_sec - CAUSAL_WARMUP_SECONDS) * sfreq + 1e-12))
    read_stop = min(
        int(row["source_sample_count"]),
        int(math.ceil(stop_sec * sfreq - 1e-12)) + 1,
    )
    if state_reset < 0 or read_stop <= state_reset:
        raise ValueError("Continuous source window has invalid source support")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    before = os.fstat(descriptor)
    observed_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if observed_identity != expected_file_identity:
        os.close(descriptor)
        raise RuntimeError("Source EDF identity changed after its verified hash")
    # Linux /proc binds pyedflib to the already-open O_NOFOLLOW descriptor,
    # closing the path-replacement race between verification and payload read.
    try:
        reader = pyedflib.EdfReader(f"/proc/self/fd/{descriptor}")
    except Exception:
        os.close(descriptor)
        raise
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        expected_names = tuple(row["raw_channel_names"])
        indices: list[int] = []
        for expected in expected_names:
            matches = [index for index, label in enumerate(labels) if label == expected]
            if len(matches) != 1:
                raise ValueError("Source EDF raw-header binding changed at runtime")
            indices.append(matches[0])
        if tuple(float(reader.getSampleFrequency(index)) for index in indices) != (sfreq,) * 19:
            raise ValueError("Source EDF sampling rates changed at runtime")
        if tuple(str(reader.getPhysicalDimension(index)).strip() for index in indices) != tuple(row["raw_units"]):
            raise ValueError("Source EDF physical units changed at runtime")
        n_read = read_stop - state_reset
        raw = np.stack(
            [
                np.asarray(reader.readSignal(index, state_reset, n_read), dtype=np.float64)
                for index in indices
            ]
        )
    finally:
        reader.close()
        after = os.fstat(descriptor)
        os.close(descriptor)
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if after_identity != expected_file_identity:
        raise RuntimeError("Source EDF changed while its continuous window was read")
    if raw.shape != (19, read_stop - state_reset) or not np.isfinite(raw).all():
        raise ValueError("Source EDF returned an invalid continuous payload")
    scales = np.asarray([_SUPPORTED_UNIT_SCALES[_unit_key(unit)] for unit in row["raw_units"]])
    raw_volts = raw * scales[:, None]
    config = CausalEDFConfig(
        output_sfreq_hz=OUTPUT_SFREQ_HZ,
        highpass_hz=0.5,
        lowpass_hz=45.0,
        butterworth_order=4,
        warmup_sec=CAUSAL_WARMUP_SECONDS,
        apply_car19=True,
    )
    processed, _, _, _, latency_sec = causal_bandpass_resample(
        raw_volts, source_sfreq_hz=sfreq, config=config
    )
    segment_start_sec = state_reset / sfreq
    crop_start = int(
        Decimal(
            str(
                (start_sec - segment_start_sec) * OUTPUT_SFREQ_HZ
                + latency_sec * OUTPUT_SFREQ_HZ
            )
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    n_output = WINDOW_SECONDS * OUTPUT_SFREQ_HZ
    crop = processed[:, crop_start : crop_start + n_output]
    values = np.ascontiguousarray(_apply_car(crop), dtype=np.float32)
    if values.shape != (19, n_output) or not np.isfinite(values).all():
        raise ValueError("C-CAR19 continuous source window has invalid output")
    return values.reshape(19, TOKENS_PER_CHANNEL, SAMPLES_PER_TOKEN)


class SourceDAPTWindowDataset(Dataset[dict[str, object]]):
    """Map-style continuous grid dataset with no annotation-time coordinate."""

    def __init__(
        self,
        manifest: LoadedSourceDAPTManifest,
        *,
        split: str,
        verify_hash_on_first_access: bool = True,
    ) -> None:
        if split not in {"pretext_train", "pretext_dev"}:
            raise ValueError("Source DAPT split must be pretext_train or pretext_dev")
        self.manifest = manifest
        self.split = split
        self.verify_hash_on_first_access = bool(verify_hash_on_first_access)
        self.records = tuple(
            row
            for row in manifest.records
            if row["eligibility"] == "eligible" and row["pretext_split"] == split
        )
        if not self.records:
            raise ValueError(f"Source DAPT split has no eligible records: {split}")
        cumulative: list[int] = []
        running = 0
        for row in self.records:
            running += int(row["window_count"])
            cumulative.append(running)
        self._cumulative = tuple(cumulative)
        self._verified_file_identities: dict[
            str, tuple[int, int, int, int, int]
        ] = {}
        self.patient_to_indices: dict[str, list[int]] = {}
        start = 0
        for row, stop in zip(self.records, self._cumulative):
            self.patient_to_indices.setdefault(str(row["patient_id"]), []).extend(range(start, stop))
            start = stop

    def __len__(self) -> int:
        return self._cumulative[-1]

    def locate(self, index: int) -> tuple[Mapping[str, object], int]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        record_index = bisect_right(self._cumulative, index)
        previous = 0 if record_index == 0 else self._cumulative[record_index - 1]
        row = self.records[record_index]
        grid_index = int(row["first_grid_index"]) + (index - previous)
        return row, grid_index

    def __getitem__(self, index: int) -> dict[str, object]:
        row, grid_index = self.locate(index)
        source = (self.manifest.tusz_root / str(row["relative_edf_path"])).resolve(strict=True)
        if source.is_symlink() or not source.is_relative_to(self.manifest.tusz_root):
            raise ValueError("Runtime source EDF is a symlink or path escape")
        uid = str(row["record_uid"])
        if uid not in self._verified_file_identities:
            digest, identity = _sha256_file_identity(source)
            # The public constructor flag is retained for API compatibility,
            # but production integrity is never optional: every record is
            # content-bound on its first access in each worker process.
            if digest != row["edf_sha256"]:
                raise ValueError("Runtime source EDF SHA-256 changed")
            self._verified_file_identities[uid] = identity
        values = _read_continuous_window(
            source,
            row,
            grid_index=grid_index,
            expected_file_identity=self._verified_file_identities[uid],
        )
        return {
            "eeg": torch.from_numpy(values),
            "position_ids": torch.tensor(row["labram_position_ids"], dtype=torch.long),
            "patient_id": str(row["patient_id"]),
            "record_uid": uid,
            "grid_index": grid_index,
        }


class PatientUniformEpochSampler(Sampler[int]):
    """Draw the same number of continuous windows from every patient."""

    def __init__(
        self,
        dataset: SourceDAPTWindowDataset,
        *,
        windows_per_patient: int,
        seed: int,
    ) -> None:
        if not isinstance(windows_per_patient, int) or windows_per_patient < 1:
            raise ValueError("windows_per_patient must be a positive integer")
        self.dataset = dataset
        self.windows_per_patient = windows_per_patient
        self.seed = int(seed)
        self.epoch = 0
        if any(not indices for indices in dataset.patient_to_indices.values()):
            raise ValueError("Every source DAPT patient must have at least one window")

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset.patient_to_indices) * self.windows_per_patient

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        selected: list[int] = []
        for patient in sorted(self.dataset.patient_to_indices):
            candidates = self.dataset.patient_to_indices[patient]
            draws = torch.randint(
                len(candidates), (self.windows_per_patient,), generator=generator
            ).tolist()
            selected.extend(candidates[index] for index in draws)
        order = torch.randperm(len(selected), generator=generator).tolist()
        return iter(selected[index] for index in order)


__all__ = [
    "CAUSAL_WARMUP_SECONDS",
    "EXPECTED_SOURCE_PATIENTS",
    "EXPECTED_SOURCE_ROWS",
    "LoadedSourceDAPTManifest",
    "OUTPUT_SFREQ_HZ",
    "PatientUniformEpochSampler",
    "SOURCE_DAPT_SCHEMA",
    "SourceDAPTWindowDataset",
    "SourcePoolAudit",
    "STRIDE_SECONDS",
    "WINDOW_SECONDS",
    "build_deepsoz_content_exclusion",
    "build_source_dapt_manifest",
    "deterministic_patient_split",
    "load_deepsoz_mapped_edf_paths",
    "load_deepsoz_patient_ids",
    "load_source_dapt_manifest",
    "load_source_pool_audit",
    "validate_source_dapt_manifest_payload",
    "write_source_dapt_manifest",
]
