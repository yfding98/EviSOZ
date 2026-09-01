"""Build the path-independent TUSZ/DeepSOZ public overlap ledger.

DeepSOZ is an annotation overlay on locally available TUSZ EDF files.  This
builder scans the canonical TUSZ tree, hashes the exact EDF bytes, and emits
both the TUSZ records and the matching DeepSOZ overlay records required by the
global overlap policy.  Absolute paths never enter the returned artifact.

``signal_content_sha256`` currently uses ``exact_edf_bytes_v1`` and therefore
equals ``file_sha256``.  It is deliberately *not* described as a hash of a
normalized, re-referenced, filtered, or resampled signal.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import TypeAlias

import pandas as pd

from .deepsoz import normalize_patient_id
from .overlap import PublicDataRecord, PublicOverlapLedger, build_public_overlap_ledger


PUBLIC_LEDGER_BUILD_SCHEMA = "tusz_deepsoz_public_ledger_build_v1.1.0"
PUBLIC_LEDGER_ARTIFACT_SCHEMA = "tusz_deepsoz_public_ledger_artifact_v1.1.0"
PUBLIC_LEDGER_ARTIFACT_FILENAME = "public_ledger_build.json"
SIGNAL_CONTENT_HASH_POLICY = "exact_edf_bytes_v1"
SIGNAL_CONTENT_HASH_DISCLOSURE = (
    "sha256_of_exact_edf_file_bytes_not_normalized_signal"
)
ALLOWED_TUSZ_SPLITS = ("train", "dev", "eval")
_ALLOWED_MAPPING_STATUSES = frozenset({"unique", "ambiguous", "unmapped"})
_LOCAL_PATIENT_RE = re.compile(r"[a-z0-9]{8}")
_TARGET_PATIENT_RE = re.compile(r"[0-9]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_SPLIT_BY_OFFICIAL = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}
_EDF_SHA_COLUMNS = (
    "exact_edf_sha256",
    "local_edf_sha256",
    "edf_sha256",
    "file_sha256",
)
_LOCAL_PATH_COLUMNS = ("local_edf_path", "local_edf")
_LOCAL_PATIENT_COLUMNS = ("local_patient_id", "local_patient")
_PUBLIC_RECORD_FIELDS = frozenset(
    {
        "dataset",
        "split",
        "patient_key",
        "file_sha256",
        "signal_content_sha256",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "build_sha256",
        "ledger_records",
        "target_patient_to_public_key",
        "builder_receipt",
    }
)
_CROSSWALK_ENTRY_FIELDS = frozenset(
    {"target_patient_id", "public_patient_key"}
)
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

TableInput: TypeAlias = pd.DataFrame | str | Path


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Public-ledger artifact is not canonical JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a 64-character SHA256 digest")
    return text


def _clean_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _strict_flag(value: object, *, field: str, row_number: int) -> bool | None:
    text = _clean_cell(value).lower()
    if not text:
        return None
    if text in {"1", "1.0", "true"}:
        return True
    if text in {"0", "0.0", "false"}:
        return False
    raise ValueError(f"{field} has a non-binary value at input row {row_number}")


def _normalize_target_patient_id(value: object) -> str:
    patient_id = normalize_patient_id(value)
    if not _TARGET_PATIENT_RE.fullmatch(patient_id):
        raise ValueError(
            f"DeepSOZ patient IDs must be numeric, got {patient_id!r}"
        )
    return patient_id


def _normalize_local_patient_id(value: object, *, field: str) -> str:
    raw = _clean_cell(value)
    canonical = raw.lower()
    if raw != canonical or not _LOCAL_PATIENT_RE.fullmatch(canonical):
        raise ValueError(
            f"{field} must be a canonical eight-character lowercase TUSZ ID"
        )
    return canonical


def _normalize_split(value: object, *, field: str) -> str:
    split = _clean_cell(value).lower()
    if split not in ALLOWED_TUSZ_SPLITS:
        raise ValueError(
            f"{field} must be one of {ALLOWED_TUSZ_SPLITS}; got {split!r}"
        )
    return split


def _stable_file_sha256(path: Path, *, display_path: str) -> str:
    """Hash a regular file and reject mutation during the read."""

    if path.is_symlink():
        raise ValueError(f"Symlinked EDF is forbidden: {display_path}")
    before = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected a regular EDF file: {display_path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if fingerprint_before != fingerprint_after:
        raise RuntimeError(f"EDF changed while it was being hashed: {display_path}")
    return digest.hexdigest()


def _canonical_dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()].astype(str)))
        raise ValueError(f"Input table contains duplicate columns: {duplicates}")
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
    ).encode("utf-8")


def _load_frozen_table(source: TableInput, *, name: str) -> tuple[pd.DataFrame, str, str]:
    """Return a defensive copy, its bound digest, and the digest policy."""

    if isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
        payload = _canonical_dataframe_csv_bytes(frame)
        return frame, hashlib.sha256(payload).hexdigest(), "canonical_dataframe_csv_v1"

    path = Path(source)
    if path.is_symlink():
        raise ValueError(f"{name} input cannot be a symlink")
    if not path.is_file():
        raise FileNotFoundError(f"{name} input CSV does not exist")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{name} input CSV changed while it was read")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()].astype(str)))
        raise ValueError(f"{name} input contains duplicate columns: {duplicates}")
    return frame, hashlib.sha256(payload).hexdigest(), "exact_csv_bytes_v1"


@dataclass(frozen=True)
class _ScannedEDF:
    relative_path: str
    split: str
    patient_key: str
    path: Path
    exact_sha256: str

    @property
    def receipt_payload(self) -> tuple[str, str, str, str]:
        return (
            self.relative_path,
            self.split,
            self.patient_key,
            self.exact_sha256,
        )


def _scan_canonical_tusz_edfs(root_input: str | Path) -> tuple[Path, tuple[_ScannedEDF, ...]]:
    root = Path(root_input)
    if root.is_symlink():
        raise ValueError("Canonical TUSZ EDF root cannot be a symlink")
    if not root.is_dir():
        raise FileNotFoundError("Canonical TUSZ EDF root does not exist")
    root = root.resolve(strict=True)

    top_level_directories = tuple(
        sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    )
    extra_splits = sorted(set(top_level_directories) - set(ALLOWED_TUSZ_SPLITS))
    if extra_splits:
        raise ValueError(f"Unexpected top-level TUSZ split directories: {extra_splits}")
    missing_splits = sorted(set(ALLOWED_TUSZ_SPLITS) - set(top_level_directories))
    if missing_splits:
        raise ValueError(f"Missing canonical TUSZ split directories: {missing_splits}")

    scanned: list[_ScannedEDF] = []
    patient_splits: dict[str, str] = {}
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        for name in sorted(directory_names):
            child = current / name
            if child.is_symlink():
                relative = child.relative_to(root).as_posix()
                raise ValueError(f"Symlinked TUSZ directory is forbidden: {relative}")
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f"Symlinked TUSZ file is forbidden: {relative}")
            if path.suffix.lower() != ".edf":
                continue
            if path.suffix != ".edf":
                raise ValueError(f"EDF suffix must be canonical lowercase .edf: {relative}")
            parts = PurePosixPath(relative).parts
            if len(parts) != 5:
                raise ValueError(
                    "Canonical TUSZ EDF paths must be "
                    "split/patient/session/montage/file.edf"
                )
            split = _normalize_split(parts[0], field="EDF path split")
            patient = _normalize_local_patient_id(
                parts[1], field="EDF path patient"
            )
            if not Path(parts[-1]).stem.startswith(f"{patient}_"):
                raise ValueError(
                    f"EDF basename does not agree with path patient: {relative}"
                )
            previous_split = patient_splits.setdefault(patient, split)
            if previous_split != split:
                raise ValueError(
                    f"TUSZ patient {patient} occurs across official splits"
                )
            exact_sha = _stable_file_sha256(path, display_path=relative)
            scanned.append(
                _ScannedEDF(
                    relative_path=relative,
                    split=split,
                    patient_key=patient,
                    path=path,
                    exact_sha256=exact_sha,
                )
            )
    if not scanned:
        raise ValueError("Canonical TUSZ EDF tree contains no EDF files")
    ordered = tuple(sorted(scanned, key=lambda item: item.relative_path))
    relative_paths = tuple(item.relative_path for item in ordered)
    if len(set(relative_paths)) != len(relative_paths):
        raise RuntimeError("TUSZ scan produced duplicate root-relative EDF paths")
    return root, ordered


@dataclass(frozen=True)
class _PatientBinding:
    target_patient_id: str
    local_patient_id: str
    official_split: str


def _build_patient_bindings(
    frame: pd.DataFrame,
    *,
    scanned: tuple[_ScannedEDF, ...],
) -> tuple[_PatientBinding, ...]:
    required = {"deepsoz_patient_id", "local_patient_id", "official_split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Patient/split crosswalk is missing columns: {missing}")

    scanned_split_by_patient: dict[str, str] = {}
    for item in scanned:
        previous = scanned_split_by_patient.setdefault(item.patient_key, item.split)
        if previous != item.split:
            raise RuntimeError("Scanner admitted a cross-split TUSZ patient")

    by_target: dict[str, _PatientBinding] = {}
    by_local: dict[str, str] = {}
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        target = _normalize_target_patient_id(row["deepsoz_patient_id"])
        local = _normalize_local_patient_id(
            row["local_patient_id"], field="local_patient_id"
        )
        split = _normalize_split(row["official_split"], field="official_split")
        model_split = _clean_cell(row.get("model_split")).lower()
        if model_split and model_split not in {
            _MODEL_SPLIT_BY_OFFICIAL[split],
            "quarantine",
        }:
            raise ValueError(
                f"model_split disagrees with official_split at input row {row_number}"
            )
        observed_split = scanned_split_by_patient.get(local)
        if observed_split is None:
            raise ValueError(
                f"Patient/split crosswalk local patient {local} is absent from TUSZ"
            )
        if observed_split != split:
            raise ValueError(
                f"Patient/split crosswalk split drift for local patient {local}"
            )
        binding = _PatientBinding(target, local, split)
        previous_binding = by_target.setdefault(target, binding)
        if previous_binding != binding:
            raise ValueError(
                f"DeepSOZ target patient {target} has conflicting local bindings"
            )
        previous_target = by_local.setdefault(local, target)
        if previous_target != target:
            raise ValueError(
                f"Local TUSZ patient {local} maps from multiple DeepSOZ patients"
            )
    if not by_target:
        raise ValueError("Patient/split crosswalk cannot be empty")
    return tuple(sorted(by_target.values(), key=lambda item: item.target_patient_id))


def _relative_mapping_path(value: object, *, root: Path) -> tuple[str, Path] | None:
    text = _clean_cell(value)
    if not text:
        return None
    raw = Path(text)
    if ".." in raw.parts:
        raise ValueError("Mapped EDF path cannot contain parent traversal")
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError("Mapped EDF path resolves outside the canonical TUSZ root") from exc
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Mapped EDF path traverses a symlink: {relative}")
    return relative, candidate


def _select_consistent_nonempty(
    row: Mapping[str, object],
    *,
    columns: tuple[str, ...],
    field: str,
) -> str:
    values = {_clean_cell(row.get(column)) for column in columns if column in row}
    values.discard("")
    if len(values) > 1:
        raise ValueError(f"Contradictory {field} columns in record crosswalk")
    return next(iter(values), "")


def _declared_edf_sha(row: Mapping[str, object]) -> str | None:
    values = {
        _require_sha256(_clean_cell(row[column]), field=column)
        for column in _EDF_SHA_COLUMNS
        if column in row and _clean_cell(row[column])
    }
    if len(values) > 1:
        raise ValueError("Record crosswalk contains contradictory EDF SHA columns")
    return next(iter(values), None)


class TargetPatientPublicCrosswalk(Mapping[str, str]):
    """Immutable, canonical DeepSOZ-ID to local-TUSZ-ID mapping."""

    def __init__(self, items: Iterable[tuple[object, object]]) -> None:
        normalized: dict[str, str] = {}
        reverse: dict[str, str] = {}
        for raw_target, raw_public in items:
            target = _normalize_target_patient_id(raw_target)
            public = _normalize_local_patient_id(
                raw_public, field="target_patient_to_public_key value"
            )
            if target in normalized:
                raise ValueError(f"Duplicate normalized target patient ID: {target}")
            if public in reverse:
                raise ValueError(
                    f"Target/public crosswalk is not one-to-one at {public}"
                )
            normalized[target] = public
            reverse[public] = target
        if not normalized:
            raise ValueError("Target/public crosswalk cannot be empty")
        self._items = tuple(sorted(normalized.items()))
        self._mapping = dict(self._items)

    def __getitem__(self, key: str) -> str:
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    @property
    def canonical_items(self) -> tuple[tuple[str, str], ...]:
        return self._items

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._items)


@dataclass(frozen=True)
class TUSZDeepSOZPublicLedgerReceipt:
    ledger_sha256: str
    ledger_receipt_sha256: str
    record_crosswalk_input_sha256: str
    record_crosswalk_hash_policy: str
    patient_split_crosswalk_input_sha256: str
    patient_split_crosswalk_hash_policy: str
    tusz_edf_roster: tuple[tuple[str, str, str, str], ...]
    tusz_edf_roster_sha256: str
    tusz_duplicate_alias_roster: tuple[tuple[str, str, str, str, str], ...]
    tusz_duplicate_alias_roster_sha256: str
    target_public_crosswalk: tuple[tuple[str, str], ...]
    target_public_crosswalk_sha256: str
    required_target_patient_ids: tuple[str, ...]
    required_target_patient_roster_sha256: str
    dataset_split_counts: tuple[tuple[str, str, int], ...]
    record_crosswalk_row_count: int
    unique_mapping_row_count: int
    unavailable_unique_mapping_row_count: int
    duplicate_overlay_rows_removed: int
    tusz_scanned_edf_count: int
    tusz_duplicate_alias_count: int
    tusz_record_count: int
    deepsoz_overlay_record_count: int
    signal_content_hash_policy: str = SIGNAL_CONTENT_HASH_POLICY
    signal_content_hash_disclosure: str = SIGNAL_CONTENT_HASH_DISCLOSURE
    schema_version: str = PUBLIC_LEDGER_BUILD_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "ledger_sha256",
            "ledger_receipt_sha256",
            "record_crosswalk_input_sha256",
            "patient_split_crosswalk_input_sha256",
            "tusz_edf_roster_sha256",
            "tusz_duplicate_alias_roster_sha256",
            "target_public_crosswalk_sha256",
            "required_target_patient_roster_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.schema_version != PUBLIC_LEDGER_BUILD_SCHEMA:
            raise ValueError("Unexpected public-ledger build schema")
        if self.signal_content_hash_policy != SIGNAL_CONTENT_HASH_POLICY:
            raise ValueError("Unsupported signal-content hash policy")
        if self.signal_content_hash_disclosure != SIGNAL_CONTENT_HASH_DISCLOSURE:
            raise ValueError("Signal-content hash disclosure cannot be weakened")
        allowed_input_hash_policies = {
            "exact_csv_bytes_v1",
            "canonical_dataframe_csv_v1",
        }
        if self.record_crosswalk_hash_policy not in allowed_input_hash_policies:
            raise ValueError("Unsupported record-crosswalk input hash policy")
        if self.patient_split_crosswalk_hash_policy not in allowed_input_hash_policies:
            raise ValueError("Unsupported patient/split input hash policy")
        if tuple(sorted(self.tusz_edf_roster)) != self.tusz_edf_roster:
            raise ValueError("TUSZ EDF roster must be canonically sorted")
        relative_paths = tuple(row[0] for row in self.tusz_edf_roster)
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("TUSZ EDF roster contains duplicate relative paths")
        if any(Path(path).is_absolute() or ".." in PurePosixPath(path).parts for path in relative_paths):
            raise ValueError("TUSZ EDF roster may contain root-relative paths only")
        for relative_path, split, patient, exact_sha in self.tusz_edf_roster:
            parts = PurePosixPath(relative_path).parts
            if len(parts) != 5 or parts[0] != split or parts[1] != patient:
                raise ValueError("TUSZ EDF roster path metadata is inconsistent")
            _normalize_split(split, field="receipt EDF split")
            _normalize_local_patient_id(patient, field="receipt EDF patient")
            _require_sha256(exact_sha, field="receipt exact EDF SHA")
        if self.tusz_edf_roster_sha256 != _canonical_sha256(self.tusz_edf_roster):
            raise ValueError("tusz_edf_roster_sha256 does not match its roster")
        if tuple(sorted(self.tusz_duplicate_alias_roster)) != (
            self.tusz_duplicate_alias_roster
        ):
            raise ValueError("TUSZ duplicate alias roster must be canonically sorted")
        if self.tusz_duplicate_alias_roster_sha256 != _canonical_sha256(
            self.tusz_duplicate_alias_roster
        ):
            raise ValueError("tusz_duplicate_alias_roster_sha256 does not match")
        roster_by_path = {
            relative_path: (split, patient, exact_sha)
            for relative_path, split, patient, exact_sha in self.tusz_edf_roster
        }
        alias_paths: set[str] = set()
        for alias_path, canonical_path, patient, split, exact_sha in (
            self.tusz_duplicate_alias_roster
        ):
            if alias_path in alias_paths:
                raise ValueError("A duplicate EDF path may appear in the alias roster once")
            alias_paths.add(alias_path)
            if alias_path == canonical_path or canonical_path >= alias_path:
                raise ValueError("Duplicate EDF canonical path must be lexically first")
            expected_role = (split, patient, exact_sha)
            if roster_by_path.get(alias_path) != expected_role:
                raise ValueError("Duplicate EDF alias metadata disagrees with full roster")
            if roster_by_path.get(canonical_path) != expected_role:
                raise ValueError("Duplicate EDF canonical metadata disagrees with full roster")
        roster_groups: dict[str, list[tuple[str, str, str]]] = {}
        for relative_path, split, patient, exact_sha in self.tusz_edf_roster:
            roster_groups.setdefault(exact_sha, []).append(
                (relative_path, split, patient)
            )
        expected_aliases: list[tuple[str, str, str, str, str]] = []
        for exact_sha, group in roster_groups.items():
            roles = {(patient, split) for _, split, patient in group}
            if len(roles) != 1:
                raise ValueError(
                    "Full EDF roster contains identical bytes across patients or splits"
                )
            ordered_group = sorted(group)
            canonical_path, split, patient = ordered_group[0]
            expected_aliases.extend(
                (alias_path, canonical_path, alias_patient, alias_split, exact_sha)
                for alias_path, alias_split, alias_patient in ordered_group[1:]
            )
        if tuple(sorted(expected_aliases)) != self.tusz_duplicate_alias_roster:
            raise ValueError("Duplicate EDF alias roster is not complete and canonical")
        if tuple(sorted(self.target_public_crosswalk)) != self.target_public_crosswalk:
            raise ValueError("Target/public crosswalk must be canonically sorted")
        if self.target_public_crosswalk_sha256 != _canonical_sha256(
            self.target_public_crosswalk
        ):
            raise ValueError("target_public_crosswalk_sha256 does not match")
        TargetPatientPublicCrosswalk(self.target_public_crosswalk)
        if tuple(sorted(set(self.required_target_patient_ids))) != (
            self.required_target_patient_ids
        ):
            raise ValueError("Required target patient IDs must be unique and sorted")
        if self.required_target_patient_roster_sha256 != _canonical_sha256(
            self.required_target_patient_ids
        ):
            raise ValueError("required_target_patient_roster_sha256 does not match")
        if tuple(sorted(self.dataset_split_counts)) != self.dataset_split_counts:
            raise ValueError("dataset_split_counts must be canonically sorted")
        if any(
            dataset not in {"tusz", "deepsoz"}
            or split not in ALLOWED_TUSZ_SPLITS
            or count < 1
            for dataset, split, count in self.dataset_split_counts
        ):
            raise ValueError("dataset_split_counts contains an invalid role or count")
        count_fields = (
            self.record_crosswalk_row_count,
            self.unique_mapping_row_count,
            self.unavailable_unique_mapping_row_count,
            self.duplicate_overlay_rows_removed,
            self.tusz_scanned_edf_count,
            self.tusz_duplicate_alias_count,
            self.tusz_record_count,
            self.deepsoz_overlay_record_count,
        )
        if any(value < 0 for value in count_fields):
            raise ValueError("Receipt counts cannot be negative")
        if self.unique_mapping_row_count > self.record_crosswalk_row_count:
            raise ValueError("Unique mapping rows exceed the input row count")
        if self.unavailable_unique_mapping_row_count > self.unique_mapping_row_count:
            raise ValueError("Unavailable unique rows exceed all unique rows")
        if self.tusz_scanned_edf_count != len(self.tusz_edf_roster):
            raise ValueError("TUSZ scanned EDF count disagrees with the full roster")
        if self.tusz_duplicate_alias_count != len(
            self.tusz_duplicate_alias_roster
        ):
            raise ValueError("TUSZ duplicate alias count disagrees with its roster")
        if self.tusz_record_count + self.tusz_duplicate_alias_count != (
            self.tusz_scanned_edf_count
        ):
            raise ValueError("Unique TUSZ identities plus aliases must cover every EDF path")
        if self.tusz_record_count != len(roster_groups):
            raise ValueError("TUSZ record count must equal unique exact-byte identities")
        by_dataset = Counter()
        for dataset, _, count in self.dataset_split_counts:
            by_dataset[dataset] += count
        if by_dataset["tusz"] != self.tusz_record_count:
            raise ValueError("TUSZ dataset/split counts disagree with receipt")
        if by_dataset["deepsoz"] != self.deepsoz_overlay_record_count:
            raise ValueError("DeepSOZ dataset/split counts disagree with receipt")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class TUSZDeepSOZPublicLedgerBuild:
    ledger: PublicOverlapLedger
    target_patient_to_public_key: TargetPatientPublicCrosswalk
    receipt: TUSZDeepSOZPublicLedgerReceipt

    def __post_init__(self) -> None:
        if self.receipt.ledger_sha256 != self.ledger.receipt.ledger_sha256:
            raise ValueError("Build receipt does not bind the supplied ledger")
        if self.receipt.ledger_receipt_sha256 != self.ledger.receipt.receipt_sha256:
            raise ValueError("Build receipt does not bind the ledger receipt")
        if self.receipt.target_public_crosswalk != (
            self.target_patient_to_public_key.canonical_items
        ):
            raise ValueError("Build crosswalk disagrees with its receipt")
        if self.receipt.dataset_split_counts != (
            self.ledger.receipt.dataset_split_counts
        ):
            raise ValueError("Build receipt dataset/split counts disagree with the ledger")
        tusz_records = self.ledger.records_for(dataset="tusz")
        deepsoz_records = self.ledger.records_for(dataset="deepsoz")
        if len(tusz_records) != self.receipt.tusz_record_count:
            raise ValueError("Build receipt TUSZ count disagrees with the ledger")
        if len(deepsoz_records) != self.receipt.deepsoz_overlay_record_count:
            raise ValueError("Build receipt DeepSOZ count disagrees with the ledger")

        expected_tusz: dict[str, PublicDataRecord] = {}
        for _, split, patient, exact_sha in self.receipt.tusz_edf_roster:
            expected_tusz.setdefault(
                exact_sha,
                PublicDataRecord(
                    dataset="tusz",
                    split=split,
                    patient_key=patient,
                    file_sha256=exact_sha,
                    signal_content_sha256=exact_sha,
                ),
            )
        if tuple(record.canonical_payload for record in tusz_records) != tuple(
            sorted(record.canonical_payload for record in expected_tusz.values())
        ):
            raise ValueError("Ledger TUSZ identities disagree with the full EDF roster")
        tusz_identity = {
            (
                record.split,
                record.patient_key,
                record.file_sha256,
                record.signal_content_sha256,
            )
            for record in tusz_records
        }
        if any(
            (
                record.split,
                record.patient_key,
                record.file_sha256,
                record.signal_content_sha256,
            )
            not in tusz_identity
            for record in deepsoz_records
        ):
            raise ValueError("A DeepSOZ overlay has no identical TUSZ EDF identity")

    @property
    def build_sha256(self) -> str:
        return self.receipt.receipt_sha256


@dataclass(frozen=True)
class TUSZDeepSOZPublicLedgerArtifact:
    """A verified in-memory view of one canonical JSON bundle.

    The object intentionally stores no bundle path.  ``bundle_sha256`` hashes
    the exact canonical JSON bytes, while ``build_sha256`` remains the hash of
    the fully validated builder receipt.
    """

    build: TUSZDeepSOZPublicLedgerBuild
    bundle_sha256: str
    schema_version: str = PUBLIC_LEDGER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.build, TUSZDeepSOZPublicLedgerBuild):
            raise TypeError("Artifact build must be a TUSZDeepSOZPublicLedgerBuild")
        object.__setattr__(
            self,
            "bundle_sha256",
            _require_sha256(self.bundle_sha256, field="bundle_sha256"),
        )
        if self.schema_version != PUBLIC_LEDGER_ARTIFACT_SCHEMA:
            raise ValueError("Unexpected public-ledger artifact schema")

    @property
    def build_sha256(self) -> str:
        return self.build.build_sha256

    @property
    def ledger(self) -> PublicOverlapLedger:
        return self.build.ledger

    @property
    def target_patient_to_public_key(self) -> TargetPatientPublicCrosswalk:
        return self.build.target_patient_to_public_key

    @property
    def receipt(self) -> TUSZDeepSOZPublicLedgerReceipt:
        return self.build.receipt


def build_tusz_deepsoz_public_ledger(
    tusz_edf_root: str | Path,
    record_crosswalk: TableInput,
    patient_split_crosswalk: TableInput,
    *,
    required_target_patient_ids: Iterable[object] | None = None,
) -> TUSZDeepSOZPublicLedgerBuild:
    """Build a complete TUSZ ledger plus DeepSOZ overlay identities.

    ``required_target_patient_ids`` should be the eligible DeepSOZ registry
    roster when the result will be passed to the OOF protocol.  In that mode,
    the returned mapping covers exactly that roster and every required patient
    must have at least one locally available, split-consistent overlay EDF.
    """

    root, scanned = _scan_canonical_tusz_edfs(tusz_edf_root)
    record_frame, record_input_sha, record_hash_policy = _load_frozen_table(
        record_crosswalk, name="record_crosswalk"
    )
    patient_frame, patient_input_sha, patient_hash_policy = _load_frozen_table(
        patient_split_crosswalk, name="patient_split_crosswalk"
    )
    bindings = _build_patient_bindings(patient_frame, scanned=scanned)
    binding_by_target = {item.target_patient_id: item for item in bindings}

    if required_target_patient_ids is None:
        required = tuple(sorted(binding_by_target))
    else:
        normalized_required = tuple(
            sorted(_normalize_target_patient_id(value) for value in required_target_patient_ids)
        )
        if len(set(normalized_required)) != len(normalized_required):
            raise ValueError("required_target_patient_ids contains duplicates")
        if not normalized_required:
            raise ValueError("required_target_patient_ids cannot be empty")
        missing = sorted(set(normalized_required) - set(binding_by_target))
        if missing:
            raise ValueError(
                f"Eligible target patients are missing from the crosswalk: {missing}"
            )
        required = normalized_required
    target_crosswalk = TargetPatientPublicCrosswalk(
        (target, binding_by_target[target].local_patient_id) for target in required
    )

    scanned_by_relative = {item.relative_path: item for item in scanned}
    scanned_by_exact_sha: dict[str, list[_ScannedEDF]] = {}
    for item in scanned:
        scanned_by_exact_sha.setdefault(item.exact_sha256, []).append(item)
    canonical_by_relative: dict[str, _ScannedEDF] = {}
    canonical_scanned: list[_ScannedEDF] = []
    duplicate_aliases: list[tuple[str, str, str, str, str]] = []
    for exact_sha, duplicates in sorted(scanned_by_exact_sha.items()):
        roles = {(item.patient_key, item.split) for item in duplicates}
        if len(roles) != 1:
            raise ValueError(
                "Identical exact EDF bytes occur across TUSZ patients or splits"
            )
        ordered_duplicates = sorted(duplicates, key=lambda item: item.relative_path)
        canonical = ordered_duplicates[0]
        canonical_scanned.append(canonical)
        for item in ordered_duplicates:
            canonical_by_relative[item.relative_path] = canonical
        for alias in ordered_duplicates[1:]:
            duplicate_aliases.append(
                (
                    alias.relative_path,
                    canonical.relative_path,
                    alias.patient_key,
                    alias.split,
                    exact_sha,
                )
            )
    canonical_scanned.sort(key=lambda item: item.relative_path)
    duplicate_aliases.sort()
    tusz_records = tuple(
        PublicDataRecord(
            dataset="tusz",
            split=item.split,
            patient_key=item.patient_key,
            file_sha256=item.exact_sha256,
            signal_content_sha256=item.exact_sha256,
        )
        for item in canonical_scanned
    )

    required_record_columns = {"deepsoz_patient_id", "mapping_status"}
    missing_columns = sorted(required_record_columns - set(record_frame.columns))
    if missing_columns:
        raise ValueError(f"record_crosswalk is missing columns: {missing_columns}")
    if not any(column in record_frame.columns for column in _LOCAL_PATH_COLUMNS):
        raise ValueError("record_crosswalk has no local EDF path column")

    overlay_by_record_sha: dict[str, PublicDataRecord] = {}
    overlay_target_ids: set[str] = set()
    unique_mapping_rows = 0
    unavailable_unique_rows = 0
    duplicate_overlay_rows = 0
    for row_number, row in enumerate(record_frame.to_dict("records"), start=2):
        status = _clean_cell(row.get("mapping_status")).lower()
        if status not in _ALLOWED_MAPPING_STATUSES:
            raise ValueError(
                f"Unsupported mapping_status {status!r} at input row {row_number}"
            )
        selected_path_text = _select_consistent_nonempty(
            row, columns=_LOCAL_PATH_COLUMNS, field="local EDF path"
        )
        mapped = _relative_mapping_path(selected_path_text, root=root)
        if status != "unique":
            continue
        unique_mapping_rows += 1
        if mapped is None:
            unavailable_unique_rows += 1
            continue
        relative_path, candidate_path = mapped
        declared_exists = _strict_flag(
            row.get("local_edf_exists"),
            field="local_edf_exists",
            row_number=row_number,
        )
        actual_exists = candidate_path.is_file()
        if declared_exists is not None and declared_exists != actual_exists:
            raise ValueError(
                f"local_edf_exists drift at record-crosswalk row {row_number}"
            )
        if not actual_exists:
            unavailable_unique_rows += 1
            continue
        scanned_item = scanned_by_relative.get(relative_path)
        if scanned_item is None:
            raise ValueError(
                "A mapped local EDF exists but is absent from the canonical TUSZ scan"
            )
        canonical_scanned_item = canonical_by_relative[relative_path]
        target = _normalize_target_patient_id(row["deepsoz_patient_id"])
        binding = binding_by_target.get(target)
        if binding is None:
            raise ValueError(
                f"Mapped DeepSOZ patient {target} is absent from patient/split crosswalk"
            )
        local_patient_text = _select_consistent_nonempty(
            row, columns=_LOCAL_PATIENT_COLUMNS, field="local patient ID"
        )
        if local_patient_text:
            row_local_patient = _normalize_local_patient_id(
                local_patient_text, field="record-crosswalk local patient"
            )
            if row_local_patient != scanned_item.patient_key:
                raise ValueError(
                    f"Local patient drift at record-crosswalk row {row_number}"
                )
        if binding.local_patient_id != scanned_item.patient_key:
            raise ValueError(
                f"Patient binding drift at record-crosswalk row {row_number}"
            )
        if binding.official_split != scanned_item.split:
            raise ValueError(
                f"Patient split drift at record-crosswalk row {row_number}"
            )
        for split_column in ("local_official_split", "source_official_split"):
            declared_split = _clean_cell(row.get(split_column))
            if declared_split and _normalize_split(
                declared_split, field=split_column
            ) != scanned_item.split:
                raise ValueError(
                    f"{split_column} drift at record-crosswalk row {row_number}"
                )
        split_agreement = _strict_flag(
            row.get("split_agreement"),
            field="split_agreement",
            row_number=row_number,
        )
        if split_agreement is False:
            raise ValueError(
                f"record_crosswalk reports split disagreement at row {row_number}"
            )
        declared_sha = _declared_edf_sha(row)
        if declared_sha is not None and declared_sha != scanned_item.exact_sha256:
            raise ValueError(
                f"Exact EDF SHA drift at record-crosswalk row {row_number}"
            )
        declared_signal_sha = _clean_cell(row.get("signal_content_sha256"))
        if declared_signal_sha:
            signal_policy = _clean_cell(row.get("signal_content_hash_policy"))
            if signal_policy and signal_policy != SIGNAL_CONTENT_HASH_POLICY:
                raise ValueError("Record crosswalk uses an incompatible signal hash policy")
            if _require_sha256(
                declared_signal_sha, field="signal_content_sha256"
            ) != scanned_item.exact_sha256:
                raise ValueError(
                    f"Signal-content SHA drift at record-crosswalk row {row_number}"
                )
        overlay = PublicDataRecord(
            dataset="deepsoz",
            split=canonical_scanned_item.split,
            patient_key=canonical_scanned_item.patient_key,
            file_sha256=canonical_scanned_item.exact_sha256,
            signal_content_sha256=canonical_scanned_item.exact_sha256,
        )
        if overlay.record_sha256 in overlay_by_record_sha:
            duplicate_overlay_rows += 1
        else:
            overlay_by_record_sha[overlay.record_sha256] = overlay
        overlay_target_ids.add(target)

    missing_required_overlays = sorted(set(required) - overlay_target_ids)
    if missing_required_overlays:
        raise ValueError(
            "Eligible target patients lack a locally available, split-consistent "
            f"DeepSOZ overlay EDF: {missing_required_overlays}"
        )
    overlay_records = tuple(
        sorted(overlay_by_record_sha.values(), key=lambda item: item.canonical_payload)
    )
    ledger = build_public_overlap_ledger((*tusz_records, *overlay_records))
    roster = tuple(item.receipt_payload for item in scanned)
    alias_roster = tuple(duplicate_aliases)
    counts = Counter((record.dataset, record.split) for record in ledger)
    dataset_split_counts = tuple(
        (dataset, split, count)
        for (dataset, split), count in sorted(counts.items())
    )
    receipt = TUSZDeepSOZPublicLedgerReceipt(
        ledger_sha256=ledger.receipt.ledger_sha256,
        ledger_receipt_sha256=ledger.receipt.receipt_sha256,
        record_crosswalk_input_sha256=record_input_sha,
        record_crosswalk_hash_policy=record_hash_policy,
        patient_split_crosswalk_input_sha256=patient_input_sha,
        patient_split_crosswalk_hash_policy=patient_hash_policy,
        tusz_edf_roster=roster,
        tusz_edf_roster_sha256=_canonical_sha256(roster),
        tusz_duplicate_alias_roster=alias_roster,
        tusz_duplicate_alias_roster_sha256=_canonical_sha256(alias_roster),
        target_public_crosswalk=target_crosswalk.canonical_items,
        target_public_crosswalk_sha256=target_crosswalk.sha256,
        required_target_patient_ids=required,
        required_target_patient_roster_sha256=_canonical_sha256(required),
        dataset_split_counts=dataset_split_counts,
        record_crosswalk_row_count=len(record_frame),
        unique_mapping_row_count=unique_mapping_rows,
        unavailable_unique_mapping_row_count=unavailable_unique_rows,
        duplicate_overlay_rows_removed=duplicate_overlay_rows,
        tusz_scanned_edf_count=len(scanned),
        tusz_duplicate_alias_count=len(alias_roster),
        tusz_record_count=len(tusz_records),
        deepsoz_overlay_record_count=len(overlay_records),
    )
    return TUSZDeepSOZPublicLedgerBuild(
        ledger=ledger,
        target_patient_to_public_key=target_crosswalk,
        receipt=receipt,
    )


def _artifact_payload(build: TUSZDeepSOZPublicLedgerBuild) -> dict[str, object]:
    if not isinstance(build, TUSZDeepSOZPublicLedgerBuild):
        raise TypeError("build must be a TUSZDeepSOZPublicLedgerBuild")
    return {
        "schema_version": PUBLIC_LEDGER_ARTIFACT_SCHEMA,
        "build_sha256": build.build_sha256,
        "ledger_records": [asdict(record) for record in build.ledger],
        "target_patient_to_public_key": [
            {
                "target_patient_id": target,
                "public_patient_key": public,
            }
            for target, public in build.target_patient_to_public_key.canonical_items
        ],
        "builder_receipt": asdict(build.receipt),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_tusz_deepsoz_public_ledger_build(
    build: TUSZDeepSOZPublicLedgerBuild,
    bundle_directory: str | Path,
) -> TUSZDeepSOZPublicLedgerArtifact:
    """Atomically publish a new, non-overwriting canonical JSON bundle."""

    payload = _artifact_payload(build)
    encoded = _canonical_json_bytes(payload)
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Public-ledger artifact exceeds the closed size limit")
    bundle = Path(bundle_directory)
    if os.path.lexists(bundle):
        raise FileExistsError("Public-ledger bundle destination already exists")
    parent = bundle.parent
    if parent.is_symlink():
        raise ValueError("Public-ledger bundle parent cannot be a symlink")
    if not parent.is_dir():
        raise FileNotFoundError("Public-ledger bundle parent directory does not exist")
    if bundle.name in {"", ".", ".."}:
        raise ValueError("Public-ledger bundle requires a concrete directory name")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{bundle.name}.tmp-", dir=parent)
    )
    temporary_file = temporary / PUBLIC_LEDGER_ARTIFACT_FILENAME
    published = False
    try:
        with temporary_file.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        if os.path.lexists(bundle):
            raise FileExistsError("Public-ledger bundle destination already exists")
        os.rename(temporary, bundle)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            if temporary_file.exists() and not temporary_file.is_symlink():
                temporary_file.unlink()
            temporary.rmdir()

    bundle_sha = hashlib.sha256(encoded).hexdigest()
    return TUSZDeepSOZPublicLedgerArtifact(
        build=build,
        bundle_sha256=bundle_sha,
    )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _require_closed_object(
    value: object,
    *,
    expected_fields: frozenset[str],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    actual = set(value)
    missing = sorted(expected_fields - actual)
    unknown = sorted(actual - expected_fields)
    if missing or unknown:
        raise ValueError(
            f"{field} violates the closed schema; missing={missing}, unknown={unknown}"
        )
    return value


def _require_json_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a JSON string")
    return value


def _require_json_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a JSON integer")
    return value


def _require_json_array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


def _receipt_from_json(value: object) -> TUSZDeepSOZPublicLedgerReceipt:
    expected_fields = frozenset(
        field.name for field in fields(TUSZDeepSOZPublicLedgerReceipt)
    )
    raw = _require_closed_object(
        value,
        expected_fields=expected_fields,
        field="builder_receipt",
    )
    string_fields = {
        "ledger_sha256",
        "ledger_receipt_sha256",
        "record_crosswalk_input_sha256",
        "record_crosswalk_hash_policy",
        "patient_split_crosswalk_input_sha256",
        "patient_split_crosswalk_hash_policy",
        "tusz_edf_roster_sha256",
        "tusz_duplicate_alias_roster_sha256",
        "target_public_crosswalk_sha256",
        "required_target_patient_roster_sha256",
        "signal_content_hash_policy",
        "signal_content_hash_disclosure",
        "schema_version",
    }
    integer_fields = {
        "record_crosswalk_row_count",
        "unique_mapping_row_count",
        "unavailable_unique_mapping_row_count",
        "duplicate_overlay_rows_removed",
        "tusz_scanned_edf_count",
        "tusz_duplicate_alias_count",
        "tusz_record_count",
        "deepsoz_overlay_record_count",
    }
    converted = dict(raw)
    for name in string_fields:
        converted[name] = _require_json_string(raw[name], field=f"builder_receipt.{name}")
    for name in integer_fields:
        converted[name] = _require_json_integer(raw[name], field=f"builder_receipt.{name}")

    roster_rows = _require_json_array(
        raw["tusz_edf_roster"], field="builder_receipt.tusz_edf_roster"
    )
    roster: list[tuple[str, str, str, str]] = []
    for index, row in enumerate(roster_rows):
        values = _require_json_array(
            row, field=f"builder_receipt.tusz_edf_roster[{index}]"
        )
        if len(values) != 4:
            raise ValueError("Each TUSZ EDF roster entry must contain four fields")
        roster.append(
            tuple(
                _require_json_string(
                    item,
                    field=f"builder_receipt.tusz_edf_roster[{index}][{offset}]",
                )
                for offset, item in enumerate(values)
            )
        )
    converted["tusz_edf_roster"] = tuple(roster)

    alias_rows = _require_json_array(
        raw["tusz_duplicate_alias_roster"],
        field="builder_receipt.tusz_duplicate_alias_roster",
    )
    aliases: list[tuple[str, str, str, str, str]] = []
    for index, row in enumerate(alias_rows):
        values = _require_json_array(
            row, field=f"builder_receipt.tusz_duplicate_alias_roster[{index}]"
        )
        if len(values) != 5:
            raise ValueError("Each TUSZ duplicate alias entry must contain five fields")
        aliases.append(
            tuple(
                _require_json_string(
                    item,
                    field=(
                        f"builder_receipt.tusz_duplicate_alias_roster"
                        f"[{index}][{offset}]"
                    ),
                )
                for offset, item in enumerate(values)
            )
        )
    converted["tusz_duplicate_alias_roster"] = tuple(aliases)

    crosswalk_rows = _require_json_array(
        raw["target_public_crosswalk"],
        field="builder_receipt.target_public_crosswalk",
    )
    receipt_crosswalk: list[tuple[str, str]] = []
    for index, row in enumerate(crosswalk_rows):
        values = _require_json_array(
            row, field=f"builder_receipt.target_public_crosswalk[{index}]"
        )
        if len(values) != 2:
            raise ValueError("Each receipt crosswalk entry must contain two fields")
        receipt_crosswalk.append(
            (
                _require_json_string(
                    values[0],
                    field=f"builder_receipt.target_public_crosswalk[{index}][0]",
                ),
                _require_json_string(
                    values[1],
                    field=f"builder_receipt.target_public_crosswalk[{index}][1]",
                ),
            )
        )
    converted["target_public_crosswalk"] = tuple(receipt_crosswalk)

    required_rows = _require_json_array(
        raw["required_target_patient_ids"],
        field="builder_receipt.required_target_patient_ids",
    )
    converted["required_target_patient_ids"] = tuple(
        _require_json_string(
            item, field=f"builder_receipt.required_target_patient_ids[{index}]"
        )
        for index, item in enumerate(required_rows)
    )

    count_rows = _require_json_array(
        raw["dataset_split_counts"],
        field="builder_receipt.dataset_split_counts",
    )
    split_counts: list[tuple[str, str, int]] = []
    for index, row in enumerate(count_rows):
        values = _require_json_array(
            row, field=f"builder_receipt.dataset_split_counts[{index}]"
        )
        if len(values) != 3:
            raise ValueError("Each dataset/split count entry must contain three fields")
        split_counts.append(
            (
                _require_json_string(
                    values[0],
                    field=f"builder_receipt.dataset_split_counts[{index}][0]",
                ),
                _require_json_string(
                    values[1],
                    field=f"builder_receipt.dataset_split_counts[{index}][1]",
                ),
                _require_json_integer(
                    values[2],
                    field=f"builder_receipt.dataset_split_counts[{index}][2]",
                ),
            )
        )
    converted["dataset_split_counts"] = tuple(split_counts)
    return TUSZDeepSOZPublicLedgerReceipt(**converted)


def _build_from_artifact_payload(payload: object) -> TUSZDeepSOZPublicLedgerBuild:
    raw = _require_closed_object(
        payload,
        expected_fields=_ARTIFACT_FIELDS,
        field="public-ledger artifact",
    )
    schema = _require_json_string(raw["schema_version"], field="schema_version")
    if schema != PUBLIC_LEDGER_ARTIFACT_SCHEMA:
        raise ValueError(f"Unsupported public-ledger artifact schema: {schema}")
    declared_build_sha = _require_sha256(
        _require_json_string(raw["build_sha256"], field="build_sha256"),
        field="build_sha256",
    )

    record_rows = _require_json_array(raw["ledger_records"], field="ledger_records")
    records: list[PublicDataRecord] = []
    for index, value in enumerate(record_rows):
        row = _require_closed_object(
            value,
            expected_fields=_PUBLIC_RECORD_FIELDS,
            field=f"ledger_records[{index}]",
        )
        record_kwargs = {
            name: _require_json_string(
                row[name], field=f"ledger_records[{index}].{name}"
            )
            for name in _PUBLIC_RECORD_FIELDS
        }
        records.append(PublicDataRecord(**record_kwargs))
    if not records:
        raise ValueError("Artifact ledger_records cannot be empty")
    canonical_record_payloads = tuple(record.canonical_payload for record in records)
    if canonical_record_payloads != tuple(sorted(canonical_record_payloads)):
        raise ValueError("ledger_records is not in canonical order")
    ledger = build_public_overlap_ledger(records)

    crosswalk_rows = _require_json_array(
        raw["target_patient_to_public_key"],
        field="target_patient_to_public_key",
    )
    crosswalk_items: list[tuple[str, str]] = []
    for index, value in enumerate(crosswalk_rows):
        row = _require_closed_object(
            value,
            expected_fields=_CROSSWALK_ENTRY_FIELDS,
            field=f"target_patient_to_public_key[{index}]",
        )
        crosswalk_items.append(
            (
                _require_json_string(
                    row["target_patient_id"],
                    field=f"target_patient_to_public_key[{index}].target_patient_id",
                ),
                _require_json_string(
                    row["public_patient_key"],
                    field=f"target_patient_to_public_key[{index}].public_patient_key",
                ),
            )
        )
    crosswalk = TargetPatientPublicCrosswalk(crosswalk_items)
    if tuple(crosswalk_items) != crosswalk.canonical_items:
        raise ValueError("target_patient_to_public_key is not in canonical order")
    receipt = _receipt_from_json(raw["builder_receipt"])
    build = TUSZDeepSOZPublicLedgerBuild(
        ledger=ledger,
        target_patient_to_public_key=crosswalk,
        receipt=receipt,
    )
    if declared_build_sha != build.build_sha256:
        raise ValueError("Artifact build_sha256 does not match the rebuilt object")
    return build


def _read_stable_artifact_bytes(path: Path) -> bytes:
    before = path.stat()
    if not path.is_file():
        raise ValueError("Public-ledger artifact entry must be a regular file")
    if before.st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("Public-ledger artifact exceeds the closed size limit")
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("Public-ledger artifact changed while it was read")
    return payload


def load_tusz_deepsoz_public_ledger_build(
    bundle_directory: str | Path,
    *,
    expected_bundle_sha256: str | None = None,
    expected_build_sha256: str | None = None,
) -> TUSZDeepSOZPublicLedgerArtifact:
    """Load a closed-schema bundle and rebuild every validated domain object."""

    bundle = Path(bundle_directory)
    if bundle.is_symlink():
        raise ValueError("Public-ledger bundle directory cannot be a symlink")
    if not bundle.is_dir():
        raise FileNotFoundError("Public-ledger bundle directory does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda item: item.name))
    if len(entries) != 1 or entries[0].name != PUBLIC_LEDGER_ARTIFACT_FILENAME:
        names = [entry.name for entry in entries]
        raise ValueError(
            "Public-ledger bundle must contain exactly the canonical JSON file; "
            f"found={names}"
        )
    artifact_file = entries[0]
    if artifact_file.is_symlink():
        raise ValueError("Public-ledger artifact JSON cannot be a symlink")
    encoded = _read_stable_artifact_bytes(artifact_file)
    bundle_sha = hashlib.sha256(encoded).hexdigest()
    if expected_bundle_sha256 is not None:
        expected_bundle = _require_sha256(
            expected_bundle_sha256, field="expected_bundle_sha256"
        )
        if bundle_sha != expected_bundle:
            raise ValueError("Public-ledger bundle SHA does not match the expected SHA")
    try:
        text = encoded.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Public-ledger artifact is not strict UTF-8 JSON") from exc
    if _canonical_json_bytes(payload) != encoded:
        raise ValueError("Public-ledger artifact bytes are not canonical JSON")
    build = _build_from_artifact_payload(payload)
    if expected_build_sha256 is not None:
        expected_build = _require_sha256(
            expected_build_sha256, field="expected_build_sha256"
        )
        if build.build_sha256 != expected_build:
            raise ValueError("Public-ledger build SHA does not match the expected SHA")
    return TUSZDeepSOZPublicLedgerArtifact(
        build=build,
        bundle_sha256=bundle_sha,
    )


__all__ = [
    "ALLOWED_TUSZ_SPLITS",
    "PUBLIC_LEDGER_ARTIFACT_FILENAME",
    "PUBLIC_LEDGER_ARTIFACT_SCHEMA",
    "PUBLIC_LEDGER_BUILD_SCHEMA",
    "SIGNAL_CONTENT_HASH_DISCLOSURE",
    "SIGNAL_CONTENT_HASH_POLICY",
    "TUSZDeepSOZPublicLedgerBuild",
    "TUSZDeepSOZPublicLedgerArtifact",
    "TUSZDeepSOZPublicLedgerReceipt",
    "TargetPatientPublicCrosswalk",
    "build_tusz_deepsoz_public_ledger",
    "load_tusz_deepsoz_public_ledger_build",
    "save_tusz_deepsoz_public_ledger_build",
]
