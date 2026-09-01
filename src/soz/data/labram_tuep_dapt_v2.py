"""Target-free TUEP data path for diversity-preserving LaBraM DAPT-v2.

The corpus directory contains diagnostic partitions, but this module never
opens or returns those labels.  They are treated only as storage containers.
Every TUSZ patient identity is excluded before a deterministic three-way
patient split is made.  The qualification split is never exposed to the
training runner.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from ..geometry import STANDARD_19
from ..models.labram import bind_labram_record_positions
from .labram_source_dapt import (
    PatientUniformEpochSampler,
    _read_continuous_window,
    _record_header,
    _reference_suffix,
    _sha256_file_identity,
    _unit_key,
)


TUEP_DAPT_V2_SCHEMA = "soz_labram_tuep_diversity_dapt_manifest_v2"
TUEP_DAPT_V2_PURPOSE = "target_free_diversity_preserving_labram_dapt_only"
TUEP_DAPT_V2_SPLIT_POLICY = (
    "sha256_seeded_after_all_tusz_identity_exclusion_120_train_24_dev_36_qualification_v1"
)
TUEP_DAPT_V2_GRID_POLICY = "record_zero_anchored_8s_grid_after_30s_causal_warmup_v1"
TUEP_DAPT_V2_PREPROCESSING = "C-CAR19_causal_0.5-45Hz_200Hz_real30s_warmup_v1"

EXPECTED_TUEP_PATIENTS = 200
EXPECTED_TUSZ_PATIENTS = 675
EXPECTED_OVERLAPPING_PATIENTS = 20
EXPECTED_SOURCE_PATIENTS = 180
EXPECTED_TRAIN_PATIENTS = 120
EXPECTED_DEV_PATIENTS = 24
EXPECTED_QUALIFICATION_PATIENTS = 36
EXPECTED_SCANNED_EDFS = 1360

WINDOW_SECONDS = 8
STRIDE_SECONDS = 8
CAUSAL_WARMUP_SECONDS = 30
OUTPUT_SFREQ_HZ = 200
SAMPLES_PER_TOKEN = 200
TOKENS_PER_CHANNEL = 8
SPLIT_SEED = 20260811

_SUPPORTED_UNITS = {"v", "mv", "uv"}
_PATIENT_RE = re.compile(r"^[a-z0-9]+$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_patient(value: object) -> str:
    patient = str(value).strip().lower()
    if not patient or _PATIENT_RE.fullmatch(patient) is None:
        raise ValueError(f"Invalid public patient identity: {value!r}")
    return patient


def _file_sha256(path_value: str | Path) -> str:
    path = Path(path_value).resolve(strict=True)
    return _sha256_file_identity(path)[0]


def _ceil_ratio(value: int, divisor: int) -> int:
    return int(
        (Decimal(value) / Decimal(divisor)).quantize(
            Decimal("1"), rounding=ROUND_CEILING
        )
    )


def _floor_nonnegative(value: float) -> int:
    return int(
        Decimal(str(float(value))).quantize(
            Decimal("1"), rounding=ROUND_FLOOR
        )
    )


def discover_tusz_patient_ids(tusz_edf_root: str | Path) -> tuple[str, ...]:
    root = Path(tusz_edf_root).resolve(strict=True)
    patients: list[str] = []
    for split in ("train", "dev", "eval"):
        directory = (root / split).resolve(strict=True)
        for candidate in sorted(directory.iterdir(), key=lambda path: path.name):
            if candidate.is_dir():
                patients.append(_safe_patient(candidate.name))
    unique = tuple(sorted(set(patients)))
    if len(unique) != EXPECTED_TUSZ_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_TUSZ_PATIENTS} TUSZ patients, got {len(unique)}"
        )
    return unique


def discover_tuep_patient_roots(tuep_root: str | Path) -> Mapping[str, Path]:
    root = Path(tuep_root).resolve(strict=True)
    patients: dict[str, Path] = {}
    for storage_directory in ("00_epilepsy", "01_no_epilepsy"):
        directory = (root / storage_directory).resolve(strict=True)
        for candidate in sorted(directory.iterdir(), key=lambda path: path.name):
            if not candidate.is_dir():
                continue
            patient = _safe_patient(candidate.name)
            if patient in patients:
                raise ValueError(f"TUEP patient appears in two storage directories: {patient}")
            patients[patient] = candidate.resolve(strict=True)
    if len(patients) != EXPECTED_TUEP_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_TUEP_PATIENTS} TUEP patients, got {len(patients)}"
        )
    return patients


def deterministic_three_way_split(
    patients: Sequence[str], *, seed: int = SPLIT_SEED
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    normalized = tuple(sorted({_safe_patient(value) for value in patients}))
    if len(normalized) != EXPECTED_SOURCE_PATIENTS:
        raise ValueError("DAPT-v2 split requires exactly 180 non-TUSZ TUEP patients")
    ranked = sorted(
        normalized,
        key=lambda patient: hashlib.sha256(
            f"{int(seed)}\0{patient}".encode("ascii")
        ).hexdigest(),
    )
    qualification = tuple(sorted(ranked[:EXPECTED_QUALIFICATION_PATIENTS]))
    dev = tuple(
        sorted(
            ranked[
                EXPECTED_QUALIFICATION_PATIENTS : EXPECTED_QUALIFICATION_PATIENTS
                + EXPECTED_DEV_PATIENTS
            ]
        )
    )
    train = tuple(
        sorted(ranked[EXPECTED_QUALIFICATION_PATIENTS + EXPECTED_DEV_PATIENTS :])
    )
    if (
        len(train) != EXPECTED_TRAIN_PATIENTS
        or len(dev) != EXPECTED_DEV_PATIENTS
        or len(qualification) != EXPECTED_QUALIFICATION_PATIENTS
        or set(train) & set(dev)
        or set(train) & set(qualification)
        or set(dev) & set(qualification)
    ):
        raise RuntimeError("DAPT-v2 deterministic patient split failed")
    return train, dev, qualification


def _deepsoz_content_hashes(source_manifest: str | Path) -> tuple[str, ...]:
    path = Path(source_manifest).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    hashes = payload.get("deepsoz_exclusion_contract", {}).get("content_sha256s")
    if (
        not isinstance(hashes, list)
        or len(hashes) != 607
        or hashes != sorted(set(hashes))
        or any(not isinstance(value, str) or _HASH_RE.fullmatch(value) is None for value in hashes)
    ):
        raise ValueError("The audited DeepSOZ content-exclusion roster is invalid")
    return tuple(hashes)


def _scan_record(
    path: Path,
    *,
    root: Path,
    patient: str,
    split: str,
    deepsoz_content_hashes: frozenset[str],
) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or resolved.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError("TUEP EDF is a symlink or escapes the public root")
    relative = resolved.relative_to(root).as_posix()
    digest = _file_sha256(resolved)
    header = _record_header(resolved)
    exclusions: list[str] = []
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
        if source_reference not in {"REF", "LE"} or references != (source_reference,) * 19:
            exclusions.append("source_reference_not_uniform_REF_or_LE")
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
        if len(units) != 19 or any(_unit_key(unit) not in _SUPPORTED_UNITS for unit in units):
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
    if exclusions:
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
        "eligibility": "eligible" if not exclusions else "excluded",
        "exclusion_codes": exclusions,
        "first_grid_index": first_grid_index,
        "window_count": window_count,
    }


def build_tuep_dapt_v2_manifest(
    *,
    tuep_root: str | Path,
    tusz_edf_root: str | Path,
    audited_source_manifest: str | Path,
    split_seed: int = SPLIT_SEED,
) -> dict[str, object]:
    root = Path(tuep_root).resolve(strict=True)
    tuep = discover_tuep_patient_roots(root)
    tusz = discover_tusz_patient_ids(tusz_edf_root)
    overlap = set(tuep) & set(tusz)
    if len(overlap) != EXPECTED_OVERLAPPING_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_OVERLAPPING_PATIENTS} TUEP/TUSZ identities, got {len(overlap)}"
        )
    source_patients = tuple(sorted(set(tuep) - set(tusz)))
    if len(source_patients) != EXPECTED_SOURCE_PATIENTS:
        raise ValueError("The non-TUSZ TUEP patient pool changed")
    train, dev, qualification = deterministic_three_way_split(
        source_patients, seed=split_seed
    )
    split_by_patient = {patient: "pretext_train" for patient in train}
    split_by_patient.update({patient: "pretext_dev" for patient in dev})
    split_by_patient.update(
        {patient: "pretext_qualification" for patient in qualification}
    )
    deepsoz_hashes = _deepsoz_content_hashes(audited_source_manifest)
    records: list[dict[str, object]] = []
    for patient in source_patients:
        paths = tuple(sorted(tuep[patient].rglob("*.edf"), key=str))
        if not paths:
            raise ValueError(f"TUEP patient has no EDF records: {patient}")
        records.extend(
            _scan_record(
                path,
                root=root,
                patient=patient,
                split=split_by_patient[patient],
                deepsoz_content_hashes=frozenset(deepsoz_hashes),
            )
            for path in paths
        )
    records.sort(key=lambda row: str(row["relative_edf_path"]))
    if len(records) != EXPECTED_SCANNED_EDFS:
        raise ValueError(
            f"Expected {EXPECTED_SCANNED_EDFS} TUEP EDFs, got {len(records)}"
        )
    if any("deepsoz_content_hash_overlap" in row["exclusion_codes"] for row in records):
        raise ValueError("A TUEP source EDF duplicates mapped DeepSOZ content")
    eligible = [row for row in records if row["eligibility"] == "eligible"]
    eligible_patients = {str(row["patient_id"]) for row in eligible}
    if eligible_patients != set(source_patients):
        raise ValueError("Every DAPT-v2 patient must contribute an eligible continuous window")
    counts = {
        "scanned_edf_count": len(records),
        "eligible_edf_count": len(eligible),
        "excluded_edf_count": len(records) - len(eligible),
        "pretext_train_window_count": sum(
            int(row["window_count"])
            for row in eligible
            if row["pretext_split"] == "pretext_train"
        ),
        "pretext_dev_window_count": sum(
            int(row["window_count"])
            for row in eligible
            if row["pretext_split"] == "pretext_dev"
        ),
        "pretext_qualification_window_count": sum(
            int(row["window_count"])
            for row in eligible
            if row["pretext_split"] == "pretext_qualification"
        ),
    }
    payload: dict[str, object] = {
        "schema_version": TUEP_DAPT_V2_SCHEMA,
        "purpose": TUEP_DAPT_V2_PURPOSE,
        "target_values_loaded": False,
        "diagnostic_directory_labels_used": False,
        "private_data_loaded": False,
        "annotation_sidecars_opened": False,
        "source_root": str(root),
        "source_pool_audit": {
            "tuep_patient_count": len(tuep),
            "tusz_excluded_patient_count": len(tusz),
            "tuep_tusz_overlap_patient_count": len(overlap),
            "source_patient_count": len(source_patients),
            "tusz_patient_roster_sha256": hashlib.sha256(
                _canonical_json_bytes(list(tusz))
            ).hexdigest(),
            "deepsoz_content_roster_source": str(
                Path(audited_source_manifest).resolve(strict=True)
            ),
            "deepsoz_content_count": len(deepsoz_hashes),
        },
        "pretext_split_contract": {
            "policy": TUEP_DAPT_V2_SPLIT_POLICY,
            "seed": int(split_seed),
            "train_patient_ids": list(train),
            "dev_patient_ids": list(dev),
            "qualification_patient_ids": list(qualification),
        },
        "continuous_grid_contract": {
            "policy": TUEP_DAPT_V2_GRID_POLICY,
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "causal_warmup_seconds": CAUSAL_WARMUP_SECONDS,
            "record_zero_anchored": True,
            "annotation_time_used": False,
        },
        "preprocessing_contract": {
            "policy": TUEP_DAPT_V2_PREPROCESSING,
            "allowed_uniform_input_references": ["REF", "LE"],
            "output_reference": "CAR19",
            "highpass_hz": 0.5,
            "lowpass_hz": 45.0,
            "output_sfreq_hz": OUTPUT_SFREQ_HZ,
            "standard_19": list(STANDARD_19),
        },
        "records": records,
        "counts": counts,
    }
    validate_tuep_dapt_v2_manifest(payload)
    return payload


def validate_tuep_dapt_v2_manifest(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("DAPT-v2 manifest must be a mapping")
    if payload.get("schema_version") != TUEP_DAPT_V2_SCHEMA:
        raise ValueError("DAPT-v2 manifest schema changed")
    if payload.get("purpose") != TUEP_DAPT_V2_PURPOSE:
        raise ValueError("DAPT-v2 manifest purpose changed")
    for field in (
        "target_values_loaded",
        "diagnostic_directory_labels_used",
        "private_data_loaded",
        "annotation_sidecars_opened",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"DAPT-v2 safety field changed: {field}")
    split = payload.get("pretext_split_contract")
    if not isinstance(split, Mapping) or split.get("policy") != TUEP_DAPT_V2_SPLIT_POLICY:
        raise ValueError("DAPT-v2 split contract changed")
    train = tuple(_safe_patient(value) for value in split.get("train_patient_ids", ()))
    dev = tuple(_safe_patient(value) for value in split.get("dev_patient_ids", ()))
    qualification = tuple(
        _safe_patient(value) for value in split.get("qualification_patient_ids", ())
    )
    if (
        len(train) != EXPECTED_TRAIN_PATIENTS
        or len(dev) != EXPECTED_DEV_PATIENTS
        or len(qualification) != EXPECTED_QUALIFICATION_PATIENTS
        or len(set(train) | set(dev) | set(qualification)) != EXPECTED_SOURCE_PATIENTS
    ):
        raise ValueError("DAPT-v2 split sizes/overlap changed")
    grid = payload.get("continuous_grid_contract")
    if not isinstance(grid, Mapping) or grid.get("policy") != TUEP_DAPT_V2_GRID_POLICY:
        raise ValueError("DAPT-v2 grid contract changed")
    preprocessing = payload.get("preprocessing_contract")
    if (
        not isinstance(preprocessing, Mapping)
        or preprocessing.get("policy") != TUEP_DAPT_V2_PREPROCESSING
        or tuple(preprocessing.get("standard_19", ())) != STANDARD_19
    ):
        raise ValueError("DAPT-v2 preprocessing contract changed")
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SCANNED_EDFS:
        raise ValueError("DAPT-v2 EDF inventory changed")
    all_patients = set(train) | set(dev) | set(qualification)
    split_lookup = {patient: "pretext_train" for patient in train}
    split_lookup.update({patient: "pretext_dev" for patient in dev})
    split_lookup.update(
        {patient: "pretext_qualification" for patient in qualification}
    )
    seen_paths: set[str] = set()
    seen_uids: set[str] = set()
    eligible_patients: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("DAPT-v2 record must be a mapping")
        patient = _safe_patient(row.get("patient_id"))
        if patient not in all_patients or row.get("pretext_split") != split_lookup[patient]:
            raise ValueError("DAPT-v2 record/patient split mismatch")
        relative = Path(str(row.get("relative_edf_path")))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
            raise ValueError("DAPT-v2 record path is unsafe")
        path_text = relative.as_posix()
        uid = str(row.get("record_uid"))
        if path_text in seen_paths or uid in seen_uids:
            raise ValueError("DAPT-v2 record inventory contains duplicates")
        seen_paths.add(path_text)
        seen_uids.add(uid)
        if _HASH_RE.fullmatch(str(row.get("edf_sha256"))) is None:
            raise ValueError("DAPT-v2 EDF digest is invalid")
        eligibility = row.get("eligibility")
        codes = row.get("exclusion_codes")
        if eligibility not in {"eligible", "excluded"} or not isinstance(codes, list):
            raise ValueError("DAPT-v2 record eligibility is invalid")
        if eligibility == "eligible":
            eligible_patients.add(patient)
            if (
                codes
                or len(row.get("raw_channel_names", ())) != 19
                or len(row.get("raw_units", ())) != 19
                or len(row.get("labram_position_ids", ())) != 19
                or row.get("source_reference") not in {"REF", "LE"}
                or row.get("first_grid_index") != 4
                or not isinstance(row.get("window_count"), int)
                or int(row["window_count"]) < 1
            ):
                raise ValueError("Eligible DAPT-v2 record violates its signal contract")
        elif int(row.get("window_count", -1)) != 0:
            raise ValueError("Excluded DAPT-v2 record exposes windows")
    if eligible_patients != all_patients:
        raise ValueError("Every DAPT-v2 patient must have eligible signal")
    declared = payload.get("counts")
    eligible = [row for row in rows if row["eligibility"] == "eligible"]
    expected_counts = {
        "scanned_edf_count": len(rows),
        "eligible_edf_count": len(eligible),
        "excluded_edf_count": len(rows) - len(eligible),
        "pretext_train_window_count": sum(
            int(row["window_count"])
            for row in eligible
            if row["pretext_split"] == "pretext_train"
        ),
        "pretext_dev_window_count": sum(
            int(row["window_count"])
            for row in eligible
            if row["pretext_split"] == "pretext_dev"
        ),
        "pretext_qualification_window_count": sum(
            int(row["window_count"])
            for row in eligible
            if row["pretext_split"] == "pretext_qualification"
        ),
    }
    if declared != expected_counts:
        raise ValueError("DAPT-v2 aggregate counts changed")


def write_tuep_dapt_v2_manifest(
    payload: Mapping[str, object], path_value: str | Path
) -> str:
    validate_tuep_dapt_v2_manifest(payload)
    path = Path(path_value)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class LoadedTUEPDAPTV2Manifest:
    payload: Mapping[str, object]
    path: Path
    sha256: str
    source_root: Path

    @property
    def records(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self.payload["records"])

    @property
    def tusz_root(self) -> Path:
        """Compatibility alias for the audited continuous-window reader."""

        return self.source_root


def load_tuep_dapt_v2_manifest(
    path_value: str | Path,
    *,
    tuep_root: str | Path | None = None,
    verify_file_inventory: bool = True,
) -> LoadedTUEPDAPTV2Manifest:
    path = Path(path_value).resolve(strict=True)
    content = path.read_bytes()
    payload = json.loads(content)
    validate_tuep_dapt_v2_manifest(payload)
    declared_root = Path(str(payload["source_root"])).resolve(strict=True)
    root = declared_root if tuep_root is None else Path(tuep_root).resolve(strict=True)
    if root != declared_root:
        raise ValueError("Runtime TUEP root differs from the manifest")
    if verify_file_inventory:
        declared_paths = {str(row["relative_edf_path"]) for row in payload["records"]}
        patients = set(payload["pretext_split_contract"]["train_patient_ids"])
        patients.update(payload["pretext_split_contract"]["dev_patient_ids"])
        patients.update(payload["pretext_split_contract"]["qualification_patient_ids"])
        current_paths: set[str] = set()
        for storage_directory in ("00_epilepsy", "01_no_epilepsy"):
            base = root / storage_directory
            for patient in patients:
                patient_root = base / patient
                if not patient_root.is_dir():
                    continue
                for candidate in patient_root.rglob("*.edf"):
                    resolved = candidate.resolve(strict=True)
                    if candidate.is_symlink() or not resolved.is_relative_to(root):
                        raise ValueError("Runtime TUEP inventory contains a symlink/escape")
                    current_paths.add(resolved.relative_to(root).as_posix())
        if current_paths != declared_paths:
            raise ValueError("Runtime TUEP EDF inventory differs from the manifest")
        for row in payload["records"]:
            source = (root / str(row["relative_edf_path"])).resolve(strict=True)
            if source.stat().st_size != int(row["file_size_bytes"]):
                raise ValueError("Runtime TUEP EDF size changed")
    return LoadedTUEPDAPTV2Manifest(
        payload=payload,
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        source_root=root,
    )


class TUEPDAPTV2WindowDataset(Dataset[dict[str, object]]):
    """Map-style annotation-free windows for one DAPT-v2 split."""

    def __init__(
        self,
        manifest: LoadedTUEPDAPTV2Manifest,
        *,
        split: str,
    ) -> None:
        if split not in {"pretext_train", "pretext_dev", "pretext_qualification"}:
            raise ValueError("Unknown DAPT-v2 split")
        self.manifest = manifest
        self.split = split
        self.records = tuple(
            row
            for row in manifest.records
            if row["eligibility"] == "eligible" and row["pretext_split"] == split
        )
        if not self.records:
            raise ValueError(f"DAPT-v2 split has no eligible records: {split}")
        cumulative: list[int] = []
        running = 0
        for row in self.records:
            running += int(row["window_count"])
            cumulative.append(running)
        self._cumulative = tuple(cumulative)
        self._verified_file_identities: dict[str, tuple[int, int, int, int, int]] = {}
        self.patient_to_indices: dict[str, list[int]] = {}
        start = 0
        for row, stop in zip(self.records, self._cumulative):
            self.patient_to_indices.setdefault(str(row["patient_id"]), []).extend(
                range(start, stop)
            )
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
        source = (
            self.manifest.source_root / str(row["relative_edf_path"])
        ).resolve(strict=True)
        if source.is_symlink() or not source.is_relative_to(self.manifest.source_root):
            raise ValueError("Runtime TUEP EDF is a symlink/path escape")
        uid = str(row["record_uid"])
        if uid not in self._verified_file_identities:
            digest, identity = _sha256_file_identity(source)
            if digest != row["edf_sha256"]:
                raise ValueError("Runtime TUEP EDF content changed")
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


__all__ = [
    "EXPECTED_DEV_PATIENTS",
    "EXPECTED_QUALIFICATION_PATIENTS",
    "EXPECTED_SOURCE_PATIENTS",
    "EXPECTED_TRAIN_PATIENTS",
    "LoadedTUEPDAPTV2Manifest",
    "PatientUniformEpochSampler",
    "SPLIT_SEED",
    "TUEPDAPTV2WindowDataset",
    "TUEP_DAPT_V2_SCHEMA",
    "build_tuep_dapt_v2_manifest",
    "deterministic_three_way_split",
    "discover_tuep_patient_roots",
    "discover_tusz_patient_ids",
    "load_tuep_dapt_v2_manifest",
    "validate_tuep_dapt_v2_manifest",
    "write_tuep_dapt_v2_manifest",
]
