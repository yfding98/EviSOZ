"""Formal, annotation-aligned TUEV morphology manifests.

This module is deliberately independent from the fixed-grid adapter in
``src.soz.data.tuev``.  It preserves the native bipolar edge coordinate,
canonicalises one-second REC intervals at 200 Hz, enforces the 30-second
causal warm-up and four-second post-context gates, and binds every record to
its official train-subject or official-evaluation-session parent.

The manifest contains labels and provenance only.  It never stores raw EEG,
LaBraM tokens, SOZ targets, private labels, or endpoint pseudo-labels.

The duplicate gate in this module uses SHA-256 of the complete EDF file bytes,
including the EDF header.  It therefore detects byte-identical EDF containers,
not all possible decoded-sample equivalents.  A re-encoded EDF with different
header bytes but identical decoded samples remains a residual leakage risk for
a future decoded-sample-hash schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence

from ..concept_oof import (
    IctalConceptOOFProtocolArtifact,
    load_ictal_concept_oof_protocol,
)
from .tuev import TUEVInterval, parse_tuev_rec
from .deepsoz import DeepSOZReferenceRegistry
from .overlap import normalize_public_patient_key
from .public_ledger_builder import (
    TUSZDeepSOZPublicLedgerArtifact,
    load_tusz_deepsoz_public_ledger_build,
)
from ..geometry import MORPHOLOGY_CLASSES, STANDARD_19, TCP_20_EDGES


TUEV_MORPHOLOGY_MANIFEST_SCHEMA = "tuev_morphology_manifest_v4.0.0"
TUEV_MORPHOLOGY_BUNDLE_SCHEMA = "tuev_morphology_manifest_bundle_v4.0.0"
TUEV_MORPHOLOGY_POLICY = (
    "annotation_aligned_slot0_warmup30_common20_authorized_content_components_v4"
)
TUEV_MORPHOLOGY_COHORT_AUTHORIZATION_SCHEMA = (
    "tuev_morphology_public_cohort_authorization_v1"
)
TUEV_MORPHOLOGY_GROUP_ASSIGNMENT_POLICY = (
    "all_eligible_deepsoz_public_keys_globally_protected;"
    "official_tuev_train_fit;official_tuev_eval_native_held;"
    "one_shared_morphology_producer_roster_for_all_target_splits_v1"
)
TUEV_MORPHOLOGY_PUBLIC_CONTENT_POLICY = (
    "complete_edf_container_sha256_used_for_file_and_signal_content_overlap;"
    "reencoded_decoded_sample_equivalence_not_evaluated_v1"
)
TUEV_MORPHOLOGY_HOLDING_TARGET_UPPER_BOUND = 58_722
TUEV_MORPHOLOGY_EXTERNAL_METADATA_SCHEMA = (
    "tuev_morphology_external_signal_metadata_v2"
)
TUEV_MORPHOLOGY_PREFLIGHT_SCHEMA = "tuev_morphology_signal_preflight_v2"
TUEV_MORPHOLOGY_PREFLIGHT_BUNDLE_SCHEMA = (
    "tuev_morphology_signal_preflight_bundle_v2"
)
TUEV_MORPHOLOGY_PREFLIGHT_POLICY = (
    "external_qc_exact_source_roster_content_component_ledger_v2"
)
TUEV_MORPHOLOGY_DUPLICATE_LEDGER_SCHEMA = (
    "tuev_morphology_exact_edf_byte_duplicate_ledger_v1"
)
TUEV_MORPHOLOGY_DUPLICATE_POLICY = (
    "exact_edf_byte_components_quarantine_rec_conflict_same_fold_cross_split_closed_v1"
)
TUEV_MORPHOLOGY_DUPLICATE_IDENTITY_BASIS = (
    "sha256_complete_edf_file_bytes_including_header;"
    "decoded_sample_equivalence_not_evaluated"
)

MORPHOLOGY_OUTPUT_SFREQ_HZ = 200
MORPHOLOGY_TARGET_SAMPLES = 200
MORPHOLOGY_CONTEXT_SAMPLES = 800
MORPHOLOGY_WARMUP_SAMPLES = 6_000
MORPHOLOGY_DURATION_TOLERANCE_SEC = 50e-6
MORPHOLOGY_OVERLAP_TOLERANCE_SEC = 50e-6
MORPHOLOGY_ALIGNMENT_TOLERANCE_SEC = 1.0 / (2.0 * MORPHOLOGY_OUTPUT_SFREQ_HZ)

TRAIN_GROUP_KIND = "verified_train_subject"
EVAL_GROUP_KIND = "official_eval_session"
HOLDING_COUNT_SEMANTICS = "holding_upper_bound_pre_fold"
FOLD_COUNT_SEMANTICS = "fold_specific_final"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRAIN_GROUP_RE = re.compile(r"[a-z0-9]{8}")
_EVAL_GROUP_RE = re.compile(r"[0-9]{3}")
_RECORD_RE = re.compile(r"[A-Za-z0-9_]+")
_BUNDLE_MANIFEST_FILE = "manifest.json"
_BUNDLE_RECEIPT_FILE = "receipt.json"
_PREFLIGHT_RECEIPT_FILE = "preflight.json"
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_RECEIPT_BYTES = 128 * 1024 * 1024
_MAX_EXTERNAL_METADATA_BYTES = 128 * 1024 * 1024

_OMISSION_CODES = frozenset(
    {
        "outside_common20",
        "duration_tolerance",
        "sample_alignment",
        "exact_duplicate",
        "same_edge_cross_class_overlap",
        "insufficient_warmup",
        "insufficient_post_context",
        "excluded_by_global_ledger",
        "signal_qc",
        "exact_edf_alias",
        "exact_signal_annotation_conflict",
        "cross_official_split_content_component",
    }
)

_DUPLICATE_RECORD_ACTIONS = frozenset(
    {
        "retain_unique",
        "retain_canonical_duplicate",
        "quarantine_exact_duplicate_alias",
        "quarantine_conflicting_annotation",
        "quarantine_cross_official_split_component",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _parse_canonical_json(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if raw != _canonical_json(value):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} cannot contain control characters")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _relative_path(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a canonical root-relative POSIX path")
    if path.as_posix() != text:
        raise ValueError(f"{field} must use canonical POSIX separators")
    return text


def _file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def _read_stable_regular_file(
    path: str | Path, *, label: str, max_bytes: int
) -> tuple[Path, bytes, str]:
    lexical = Path(path).absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"{label} must be a regular non-symlinked file")
    if lexical.resolve(strict=True) != lexical:
        raise ValueError(f"{label} must use its canonical non-symlink path")
    before = lexical.stat()
    if not 1 <= before.st_size <= max_bytes:
        raise ValueError(f"{label} size is invalid")
    raw = lexical.read_bytes()
    after = lexical.stat()
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
        raise RuntimeError(f"{label} changed while it was read")
    return lexical, raw, hashlib.sha256(raw).hexdigest()


def _round_half_up(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("Sample coordinate must be finite")
    return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _validated_root(path: str | Path) -> Path:
    lexical = Path(path).absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError("TUEV EDF root must be a regular directory, not a symlink")
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ValueError("TUEV EDF root must use its canonical non-symlink path")
    if resolved.name != "edf":
        raise ValueError("TUEV root must be the release's canonical edf directory")
    return resolved


def _require_regular_under(path: Path, *, root: Path, label: str) -> Path:
    lexical = path.absolute()
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} lies outside the TUEV root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} contains a non-canonical path component")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink")
    if not lexical.is_file() or lexical.resolve(strict=True) != lexical:
        raise ValueError(f"{label} must be a canonical regular file")
    return lexical


@dataclass(frozen=True)
class TUEVMorphologySourceRecord:
    """One physical EDF/REC pair bound to its complete official parent group."""

    edf_root: Path
    edf_path: Path
    rec_path: Path
    relative_edf_path: str
    relative_rec_path: str
    official_split: str
    group_id: str
    group_kind: str
    source_subject_id: str | None
    record_id: str
    edf_sha256: str
    rec_sha256: str
    derivative_files: tuple[tuple[str, str], ...]
    parent_group_files: tuple[tuple[str, str], ...]
    group_file_roster_sha256: str

    def __post_init__(self) -> None:
        if not all(path.is_absolute() for path in (self.edf_root, self.edf_path, self.rec_path)):
            raise ValueError("Discovered TUEV paths must be absolute")
        _relative_path(self.relative_edf_path, field="relative_edf_path")
        _relative_path(self.relative_rec_path, field="relative_rec_path")
        if self.official_split not in {"train", "eval"}:
            raise ValueError("official_split must be train or eval")
        _text(self.group_id, field="group_id")
        _text(self.record_id, field="record_id")
        if self.official_split == "train":
            if self.group_kind != TRAIN_GROUP_KIND:
                raise ValueError("TUEV train records must use verified-subject grouping")
            if self.source_subject_id is None or not _TRAIN_GROUP_RE.fullmatch(
                self.source_subject_id
            ):
                raise ValueError("TUEV train subject ID is not canonical")
        else:
            if self.group_kind != EVAL_GROUP_KIND:
                raise ValueError("TUEV eval records must use official-session grouping")
            if self.source_subject_id is not None:
                raise ValueError("Official eval sessions must not be renamed as patients")
        _sha(self.edf_sha256, field="edf_sha256")
        _sha(self.rec_sha256, field="rec_sha256")
        _sha(self.group_file_roster_sha256, field="group_file_roster_sha256")
        for field, roster in (
            ("derivative_files", self.derivative_files),
            ("parent_group_files", self.parent_group_files),
        ):
            paths = tuple(path for path, _ in roster)
            if paths != tuple(sorted(set(paths))):
                raise ValueError(f"{field} paths must be unique and sorted")
            for path, digest in roster:
                _relative_path(path, field=f"{field}.path")
                _sha(digest, field=f"{field}.sha256")
        if self.group_file_roster_sha256 != _canonical_sha256(
            self.parent_group_files
        ):
            raise ValueError("Parent-group file roster SHA is not reproducible")
        parent_paths = {path for path, _ in self.parent_group_files}
        if self.relative_edf_path not in parent_paths or self.relative_rec_path not in parent_paths:
            raise ValueError("Parent-group receipt omits the record EDF/REC")
        if not {path for path, _ in self.derivative_files} <= parent_paths:
            raise ValueError("Record derivatives are absent from the parent-group receipt")


def discover_tuev_morphology_sources(
    edf_root: str | Path,
) -> tuple[TUEVMorphologySourceRecord, ...]:
    """Discover the exact two-level TUEV release tree and hash every sibling.

    Train grouping is the verified subject directory.  Evaluation grouping is
    only the official numeric session/index directory.  No eval group is
    called a subject or patient.
    """

    root = _validated_root(edf_root)
    records: list[TUEVMorphologySourceRecord] = []
    for split in ("train", "eval"):
        split_dir = root / split
        if split_dir.is_symlink() or not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        groups = tuple(sorted(path for path in split_dir.iterdir() if path.is_dir()))
        if not groups:
            raise ValueError(f"TUEV official {split} split contains no groups")
        for group_dir in groups:
            if group_dir.is_symlink() or group_dir.resolve(strict=True) != group_dir.absolute():
                raise ValueError("TUEV group directories may not be symlinks")
            group_name = group_dir.name
            if split == "train":
                if not _TRAIN_GROUP_RE.fullmatch(group_name):
                    raise ValueError(f"Invalid TUEV train subject directory: {group_name!r}")
                group_kind = TRAIN_GROUP_KIND
                group_id = f"train-subject:{group_name}"
                subject_id: str | None = group_name
            else:
                if not _EVAL_GROUP_RE.fullmatch(group_name):
                    raise ValueError(f"Invalid TUEV eval session directory: {group_name!r}")
                group_kind = EVAL_GROUP_KIND
                group_id = f"eval-session:{group_name}"
                subject_id = None

            children = tuple(sorted(group_dir.iterdir()))
            if any(path.is_symlink() for path in children):
                raise ValueError("TUEV parent groups may not contain symlinks")
            if any(not path.is_file() for path in children):
                raise ValueError(
                    "TUEV parent groups must contain regular files only; "
                    "unhashed subdirectories or special files are forbidden"
                )
            files = children
            if any(path.suffix.lower() not in {".edf", ".rec", ".lab", ".htk"} for path in files):
                unknown = [path.name for path in files if path.suffix.lower() not in {".edf", ".rec", ".lab", ".htk"}]
                raise ValueError(f"Unexpected file types in TUEV parent group: {unknown}")
            file_roster = tuple(
                (path.relative_to(root).as_posix(), _file_sha256(_require_regular_under(path, root=root, label="TUEV group file")))
                for path in files
            )
            group_roster_sha = _canonical_sha256(file_roster)
            file_hash = dict(file_roster)
            edfs = tuple(path for path in files if path.suffix.lower() == ".edf")
            if not edfs:
                raise ValueError(f"TUEV parent group {group_id!r} contains no EDF")
            for edf in edfs:
                if not _RECORD_RE.fullmatch(edf.stem):
                    raise ValueError(f"Invalid TUEV record basename: {edf.stem!r}")
                rec = edf.with_suffix(".rec")
                if rec not in files:
                    raise FileNotFoundError(rec)
                edf_relative = edf.relative_to(root).as_posix()
                rec_relative = rec.relative_to(root).as_posix()
                edf_sha = file_hash[edf_relative]
                record_uid = f"{split}/{group_name}/{edf.stem}"
                derivative = tuple(
                    (relative, digest)
                    for relative, digest in file_roster
                    if relative != edf_relative
                    and (
                        relative == rec_relative
                        or PurePosixPath(relative).name.startswith(f"{edf.stem}_ch")
                    )
                )
                records.append(
                    TUEVMorphologySourceRecord(
                        edf_root=root,
                        edf_path=edf,
                        rec_path=rec,
                        relative_edf_path=edf_relative,
                        relative_rec_path=rec_relative,
                        official_split=split,
                        group_id=group_id,
                        group_kind=group_kind,
                        source_subject_id=subject_id,
                        record_id=record_uid,
                        edf_sha256=edf_sha,
                        rec_sha256=file_hash[rec_relative],
                        derivative_files=derivative,
                        parent_group_files=file_roster,
                        group_file_roster_sha256=group_roster_sha,
                    )
                )
    ordered = tuple(sorted(records, key=lambda item: item.relative_edf_path))
    if len({item.record_id for item in ordered}) != len(ordered):
        raise RuntimeError("TUEV discovery produced duplicate record IDs")
    return ordered


@dataclass(frozen=True)
class TUEVExactSignalDuplicateClass:
    """Records sharing one SHA-256 over complete, byte-identical EDF files."""

    edf_sha256: str
    record_descriptors: tuple[tuple[str, str, str, str], ...]
    group_ids: tuple[str, ...]
    official_splits: tuple[str, ...]
    annotation_status: str
    canonical_record_id: str | None

    def __post_init__(self) -> None:
        _sha(self.edf_sha256, field="duplicate_class.edf_sha256")
        if len(self.record_descriptors) < 2:
            raise ValueError("An exact EDF-byte duplicate class requires >=2 records")
        record_ids = tuple(row[0] for row in self.record_descriptors)
        if record_ids != tuple(sorted(set(record_ids))):
            raise ValueError("Duplicate-class records must be unique and sorted")
        for record_id, group_id, official_split, rec_sha256 in (
            self.record_descriptors
        ):
            _text(record_id, field="duplicate_class.record_id")
            _text(group_id, field="duplicate_class.group_id")
            if official_split not in {"train", "eval"}:
                raise ValueError("Duplicate-class official split is invalid")
            _sha(rec_sha256, field="duplicate_class.rec_sha256")
        expected_groups = tuple(sorted({row[1] for row in self.record_descriptors}))
        expected_splits = tuple(sorted({row[2] for row in self.record_descriptors}))
        if self.group_ids != expected_groups:
            raise ValueError("Duplicate-class group roster is not reproducible")
        if self.official_splits != expected_splits:
            raise ValueError("Duplicate-class split roster is not reproducible")
        rec_hashes = {row[3] for row in self.record_descriptors}
        expected_status = (
            "identical_rec_bytes"
            if len(rec_hashes) == 1
            else "conflicting_rec_bytes"
        )
        if self.annotation_status != expected_status:
            raise ValueError("Duplicate-class REC conflict status is incorrect")
        expected_canonical = min(record_ids) if len(rec_hashes) == 1 else None
        if self.canonical_record_id != expected_canonical:
            raise ValueError(
                "Only an exact EDF+REC duplicate class has a canonical record"
            )

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "edf_sha256": self.edf_sha256,
            "record_descriptors": [list(row) for row in self.record_descriptors],
            "group_ids": list(self.group_ids),
            "official_splits": list(self.official_splits),
            "annotation_status": self.annotation_status,
            "canonical_record_id": self.canonical_record_id,
        }


@dataclass(frozen=True)
class TUEVContentGroupComponent:
    """Parent groups connected transitively by one or more exact EDF hashes."""

    component_id: str
    group_ids: tuple[str, ...]
    official_splits: tuple[str, ...]
    connecting_edf_sha256s: tuple[str, ...]
    policy_action: str

    def __post_init__(self) -> None:
        _text(self.component_id, field="content_component.component_id")
        if not self.group_ids or self.group_ids != tuple(
            sorted(set(self.group_ids))
        ):
            raise ValueError("Content-component groups must be non-empty and sorted")
        if self.official_splits != tuple(sorted(set(self.official_splits))) or not set(
            self.official_splits
        ) <= {"train", "eval"}:
            raise ValueError("Content-component official splits are invalid")
        if self.connecting_edf_sha256s != tuple(
            sorted(set(self.connecting_edf_sha256s))
        ):
            raise ValueError("Content-component EDF hashes must be unique and sorted")
        for digest in self.connecting_edf_sha256s:
            _sha(digest, field="content_component.edf_sha256")
        expected_action = (
            "quarantine_cross_official_split_component"
            if len(self.official_splits) > 1
            else "same_fold_required"
        )
        if self.policy_action != expected_action:
            raise ValueError("Content-component split policy is not fail-closed")
        expected_id = "content-component:" + _canonical_sha256(
            (
                "exact_edf_group_component",
                self.group_ids,
                self.connecting_edf_sha256s,
            )
        )
        if self.component_id != expected_id:
            raise ValueError("Content-component ID is not reproducible")

    @property
    def crosses_official_split(self) -> bool:
        return len(self.official_splits) > 1

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "group_ids": list(self.group_ids),
            "official_splits": list(self.official_splits),
            "connecting_edf_sha256s": list(self.connecting_edf_sha256s),
            "policy_action": self.policy_action,
        }


@dataclass(frozen=True)
class TUEVDuplicateRecordDecision:
    """Explicit keep/quarantine decision for every discovered source record."""

    record_id: str
    group_id: str
    official_split: str
    edf_sha256: str
    rec_sha256: str
    component_id: str
    action: str
    canonical_record_id: str | None

    def __post_init__(self) -> None:
        for field in ("record_id", "group_id", "component_id"):
            _text(getattr(self, field), field=f"record_decision.{field}")
        if self.official_split not in {"train", "eval"}:
            raise ValueError("Record-decision official split is invalid")
        _sha(self.edf_sha256, field="record_decision.edf_sha256")
        _sha(self.rec_sha256, field="record_decision.rec_sha256")
        if self.action not in _DUPLICATE_RECORD_ACTIONS:
            raise ValueError(f"Unknown exact EDF-byte record action: {self.action!r}")
        if self.canonical_record_id is not None:
            _text(
                self.canonical_record_id,
                field="record_decision.canonical_record_id",
            )
        if self.action == "retain_canonical_duplicate":
            if self.canonical_record_id != self.record_id:
                raise ValueError("Retained duplicate must identify itself as canonical")
        elif self.action == "quarantine_exact_duplicate_alias":
            if self.canonical_record_id in {None, self.record_id}:
                raise ValueError("An exact alias must name another canonical record")
        elif self.canonical_record_id is not None:
            raise ValueError("Only identical EDF+REC duplicates name a canonical record")

    @property
    def quarantined(self) -> bool:
        return self.action.startswith("quarantine_")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "group_id": self.group_id,
            "official_split": self.official_split,
            "edf_sha256": self.edf_sha256,
            "rec_sha256": self.rec_sha256,
            "component_id": self.component_id,
            "action": self.action,
            "canonical_record_id": self.canonical_record_id,
        }


@dataclass(frozen=True)
class TUEVExactSignalDuplicateLedger:
    """Replayable exact-EDF-byte graph, conflicts, and record decisions.

    Despite the legacy ``ExactSignal`` API name, identity here is deliberately
    limited to the complete EDF container bytes.  Decoded sample equivalence
    is not claimed by this schema.
    """

    source_roster_sha256: str
    duplicate_classes: tuple[TUEVExactSignalDuplicateClass, ...]
    group_components: tuple[TUEVContentGroupComponent, ...]
    record_decisions: tuple[TUEVDuplicateRecordDecision, ...]
    identity_basis: str = TUEV_MORPHOLOGY_DUPLICATE_IDENTITY_BASIS
    schema_version: str = TUEV_MORPHOLOGY_DUPLICATE_LEDGER_SCHEMA
    policy_version: str = TUEV_MORPHOLOGY_DUPLICATE_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != TUEV_MORPHOLOGY_DUPLICATE_LEDGER_SCHEMA:
            raise ValueError("Unsupported TUEV exact EDF-byte ledger schema")
        if self.policy_version != TUEV_MORPHOLOGY_DUPLICATE_POLICY:
            raise ValueError("TUEV exact EDF-byte duplicate policy changed")
        if self.identity_basis != TUEV_MORPHOLOGY_DUPLICATE_IDENTITY_BASIS:
            raise ValueError("TUEV duplicate identity must be complete EDF-file bytes")
        _sha(self.source_roster_sha256, field="duplicate_ledger.source_roster_sha256")
        class_hashes = tuple(item.edf_sha256 for item in self.duplicate_classes)
        if class_hashes != tuple(sorted(set(class_hashes))):
            raise ValueError("Duplicate classes must be unique and content-sorted")
        component_ids = tuple(item.component_id for item in self.group_components)
        if not component_ids or component_ids != tuple(sorted(set(component_ids))):
            raise ValueError("Content components must be non-empty and ID-sorted")
        record_ids = tuple(item.record_id for item in self.record_decisions)
        if not record_ids or record_ids != tuple(sorted(set(record_ids))):
            raise ValueError("Record decisions must be non-empty and record-sorted")

        component_by_id = {
            component.component_id: component
            for component in self.group_components
        }
        group_to_component: dict[str, TUEVContentGroupComponent] = {}
        for component in self.group_components:
            for group_id in component.group_ids:
                if group_id in group_to_component:
                    raise ValueError("A parent group occurs in multiple content components")
                group_to_component[group_id] = component
        decisions = {item.record_id: item for item in self.record_decisions}
        decision_groups = {item.group_id for item in self.record_decisions}
        if set(group_to_component) != decision_groups:
            raise ValueError("Content components do not partition the source groups")
        group_splits: dict[str, str] = {}
        for decision in self.record_decisions:
            component = component_by_id.get(decision.component_id)
            if component is None or decision.group_id not in component.group_ids:
                raise ValueError("Record decision points to another content component")
            previous_split = group_splits.setdefault(
                decision.group_id, decision.official_split
            )
            if previous_split != decision.official_split:
                raise ValueError("One parent group spans contradictory official splits")

        duplicate_by_record: dict[str, TUEVExactSignalDuplicateClass] = {}
        for duplicate_class in self.duplicate_classes:
            for record_id, group_id, official_split, rec_sha256 in (
                duplicate_class.record_descriptors
            ):
                decision = decisions.get(record_id)
                if decision is None:
                    raise ValueError("Duplicate class names an unknown source record")
                if record_id in duplicate_by_record:
                    raise ValueError("One source record occurs in multiple EDF classes")
                duplicate_by_record[record_id] = duplicate_class
                if (
                    decision.group_id != group_id
                    or decision.official_split != official_split
                    or decision.rec_sha256 != rec_sha256
                    or decision.edf_sha256 != duplicate_class.edf_sha256
                ):
                    raise ValueError("Duplicate class and record decision disagree")

        for component in self.group_components:
            expected_splits = tuple(
                sorted({group_splits[group_id] for group_id in component.group_ids})
            )
            expected_hashes = tuple(
                duplicate_class.edf_sha256
                for duplicate_class in self.duplicate_classes
                if set(duplicate_class.group_ids) <= set(component.group_ids)
            )
            if component.official_splits != expected_splits:
                raise ValueError("Content-component split roster is not reproducible")
            if component.connecting_edf_sha256s != expected_hashes:
                raise ValueError("Content-component EDF roster is not reproducible")

        for decision in self.record_decisions:
            component = component_by_id[decision.component_id]
            duplicate_class = duplicate_by_record.get(decision.record_id)
            canonical: str | None = None
            if component.crosses_official_split:
                expected_action = "quarantine_cross_official_split_component"
            elif duplicate_class is None:
                expected_action = "retain_unique"
            elif duplicate_class.annotation_status == "conflicting_rec_bytes":
                expected_action = "quarantine_conflicting_annotation"
            else:
                canonical = duplicate_class.canonical_record_id
                expected_action = (
                    "retain_canonical_duplicate"
                    if decision.record_id == canonical
                    else "quarantine_exact_duplicate_alias"
                )
            if decision.action != expected_action:
                raise ValueError("Exact EDF-byte record decision is not reproducible")
            if decision.canonical_record_id != canonical:
                raise ValueError("Record-decision canonical identity changed")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "identity_basis": self.identity_basis,
            "source_roster_sha256": self.source_roster_sha256,
            "duplicate_classes": [
                item.canonical_payload for item in self.duplicate_classes
            ],
            "group_components": [
                item.canonical_payload for item in self.group_components
            ],
            "record_decisions": [
                item.canonical_payload for item in self.record_decisions
            ],
        }

    @property
    def ledger_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)

    @property
    def decision_by_record_id(self) -> dict[str, TUEVDuplicateRecordDecision]:
        return {item.record_id: item for item in self.record_decisions}

    @property
    def component_by_group_id(self) -> dict[str, TUEVContentGroupComponent]:
        return {
            group_id: component
            for component in self.group_components
            for group_id in component.group_ids
        }

    @property
    def cross_split_quarantined_group_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                group_id
                for component in self.group_components
                if component.crosses_official_split
                for group_id in component.group_ids
            )
        )


def build_tuev_exact_signal_duplicate_ledger(
    sources: Sequence[TUEVMorphologySourceRecord],
) -> TUEVExactSignalDuplicateLedger:
    """Build an exact EDF-file-byte graph without merging conflicting REC.

    This intentionally hashes the complete on-disk EDF container.  It does not
    claim to detect separately encoded files with equal decoded sample arrays.
    """

    ordered = tuple(sorted(sources, key=lambda item: item.record_id))
    if not ordered or tuple(item.record_id for item in ordered) != tuple(
        sorted({item.record_id for item in ordered})
    ):
        raise ValueError("Duplicate-ledger sources must be non-empty and unique")
    groups = tuple(sorted({item.group_id for item in ordered}))
    group_index = {group_id: index for index, group_id in enumerate(groups)}
    dsu = _DisjointSet(len(groups))
    by_edf: dict[str, list[TUEVMorphologySourceRecord]] = {}
    for source in ordered:
        by_edf.setdefault(source.edf_sha256, []).append(source)
    duplicate_sources = {
        digest: tuple(sorted(records, key=lambda item: item.record_id))
        for digest, records in by_edf.items()
        if len(records) > 1
    }
    for records in duplicate_sources.values():
        first = group_index[records[0].group_id]
        for record in records[1:]:
            dsu.union(first, group_index[record.group_id])

    duplicate_classes = tuple(
        TUEVExactSignalDuplicateClass(
            edf_sha256=digest,
            record_descriptors=tuple(
                (
                    source.record_id,
                    source.group_id,
                    source.official_split,
                    source.rec_sha256,
                )
                for source in records
            ),
            group_ids=tuple(sorted({source.group_id for source in records})),
            official_splits=tuple(
                sorted({source.official_split for source in records})
            ),
            annotation_status=(
                "identical_rec_bytes"
                if len({source.rec_sha256 for source in records}) == 1
                else "conflicting_rec_bytes"
            ),
            canonical_record_id=(
                min(source.record_id for source in records)
                if len({source.rec_sha256 for source in records}) == 1
                else None
            ),
        )
        for digest, records in sorted(duplicate_sources.items())
    )
    groups_by_root: dict[int, list[str]] = {}
    for group_id in groups:
        groups_by_root.setdefault(dsu.find(group_index[group_id]), []).append(
            group_id
        )
    split_by_group = {source.group_id: source.official_split for source in ordered}
    components: list[TUEVContentGroupComponent] = []
    for member_groups in groups_by_root.values():
        group_ids = tuple(sorted(member_groups))
        connecting_hashes = tuple(
            duplicate_class.edf_sha256
            for duplicate_class in duplicate_classes
            if set(duplicate_class.group_ids) <= set(group_ids)
        )
        official_splits = tuple(
            sorted({split_by_group[group_id] for group_id in group_ids})
        )
        component_id = "content-component:" + _canonical_sha256(
            ("exact_edf_group_component", group_ids, connecting_hashes)
        )
        components.append(
            TUEVContentGroupComponent(
                component_id=component_id,
                group_ids=group_ids,
                official_splits=official_splits,
                connecting_edf_sha256s=connecting_hashes,
                policy_action=(
                    "quarantine_cross_official_split_component"
                    if len(official_splits) > 1
                    else "same_fold_required"
                ),
            )
        )
    components_tuple = tuple(sorted(components, key=lambda item: item.component_id))
    component_by_group = {
        group_id: component
        for component in components_tuple
        for group_id in component.group_ids
    }
    duplicate_by_record = {
        record_id: duplicate_class
        for duplicate_class in duplicate_classes
        for record_id, _, _, _ in duplicate_class.record_descriptors
    }
    decisions: list[TUEVDuplicateRecordDecision] = []
    for source in ordered:
        component = component_by_group[source.group_id]
        duplicate_class = duplicate_by_record.get(source.record_id)
        canonical_record_id: str | None = None
        if component.crosses_official_split:
            action = "quarantine_cross_official_split_component"
        elif duplicate_class is None:
            action = "retain_unique"
        elif duplicate_class.annotation_status == "conflicting_rec_bytes":
            action = "quarantine_conflicting_annotation"
        else:
            canonical_record_id = duplicate_class.canonical_record_id
            action = (
                "retain_canonical_duplicate"
                if source.record_id == canonical_record_id
                else "quarantine_exact_duplicate_alias"
            )
        decisions.append(
            TUEVDuplicateRecordDecision(
                record_id=source.record_id,
                group_id=source.group_id,
                official_split=source.official_split,
                edf_sha256=source.edf_sha256,
                rec_sha256=source.rec_sha256,
                component_id=component.component_id,
                action=action,
                canonical_record_id=canonical_record_id,
            )
        )
    return TUEVExactSignalDuplicateLedger(
        source_roster_sha256=_canonical_sha256(_source_roster_payload(ordered)),
        duplicate_classes=duplicate_classes,
        group_components=components_tuple,
        record_decisions=tuple(decisions),
    )


def _require_content_component_closed_roles(
    ledger: TUEVExactSignalDuplicateLedger,
    role_by_group: Mapping[str, str],
) -> None:
    if set(role_by_group) != set(ledger.component_by_group_id):
        raise ValueError("Fold roles do not cover the exact content-component roster")
    for component in ledger.group_components:
        roles = {role_by_group[group_id] for group_id in component.group_ids}
        if len(roles) != 1:
            raise ValueError(
                "Every exact-EDF-byte-connected parent-group component must "
                "remain wholly in one fit/held/excluded role"
            )


@dataclass(frozen=True)
class TUEVMorphologyRecordMetadata:
    """Strict signal-preflight output supplied to manifest construction."""

    relative_edf_path: str
    edf_sha256: str
    source_sfreq_hz: float
    source_sample_count: int
    output_sample_count: int
    direct_standard19: bool
    standard19_mapping_sha256: str
    preprocessing_receipt_sha256: str
    signal_qc_passed: bool
    signal_qc_receipt_sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_edf_path, field="relative_edf_path")
        for field in (
            "edf_sha256",
            "standard19_mapping_sha256",
            "preprocessing_receipt_sha256",
            "signal_qc_receipt_sha256",
        ):
            _sha(getattr(self, field), field=field)
        if not math.isfinite(float(self.source_sfreq_hz)) or self.source_sfreq_hz <= 0:
            raise ValueError("source_sfreq_hz must be finite and positive")
        _integer(self.source_sample_count, field="source_sample_count", minimum=1)
        _integer(self.output_sample_count, field="output_sample_count", minimum=1)
        expected_output_samples = int(
            math.ceil(
                self.source_sample_count
                * MORPHOLOGY_OUTPUT_SFREQ_HZ
                / float(self.source_sfreq_hz)
            )
        )
        if self.output_sample_count != expected_output_samples:
            raise ValueError(
                "output_sample_count must equal causal full-record resampling "
                "length ceil(source_samples*200/source_sfreq)"
            )
        if not isinstance(self.direct_standard19, bool) or not isinstance(
            self.signal_qc_passed, bool
        ):
            raise TypeError("Signal geometry/QC flags must be bool")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "relative_edf_path": self.relative_edf_path,
            "edf_sha256": self.edf_sha256,
            "source_sfreq_hz": float(self.source_sfreq_hz),
            "source_sample_count": self.source_sample_count,
            "output_sample_count": self.output_sample_count,
            "direct_standard19": self.direct_standard19,
            "standard19_mapping_sha256": self.standard19_mapping_sha256,
            "preprocessing_receipt_sha256": self.preprocessing_receipt_sha256,
            "signal_qc_passed": self.signal_qc_passed,
            "signal_qc_receipt_sha256": self.signal_qc_receipt_sha256,
        }


@dataclass(frozen=True)
class TUEVMorphologyPreflightArtifact:
    """Hashes returned after atomically publishing an external preflight gate."""

    path: Path
    bundle_manifest_sha256: str
    preflight_receipt_sha256: str
    external_metadata_sha256: str
    source_roster_sha256: str
    duplicate_ledger_sha256: str
    record_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Preflight artifact path must be absolute")
        for field in (
            "bundle_manifest_sha256",
            "preflight_receipt_sha256",
            "external_metadata_sha256",
            "source_roster_sha256",
            "duplicate_ledger_sha256",
        ):
            _sha(getattr(self, field), field=field)
        _integer(self.record_count, field="record_count", minimum=1)


_VERIFIED_TUEV_MORPHOLOGY_PREFLIGHT_MARKER = object()


@dataclass(frozen=True, init=False)
class VerifiedTUEVMorphologyPreflight:
    """Opaque signal-metadata verification issued only by the strict loader.

    Current first-party metadata are independently regenerated from the real
    EDF/REC/LAB/HTK tree before this object can be issued.  Self-reported or
    legacy external summaries cannot issue this token.
    """

    path: Path
    bundle_manifest_sha256: str
    preflight_receipt_sha256: str
    external_metadata_sha256: str
    source_roster_sha256: str
    producer_source_sha256: str
    preprocessing_policy_sha256: str
    standard19_mapping_policy_sha256: str
    duplicate_ledger_sha256: str
    duplicate_ledger: TUEVExactSignalDuplicateLedger
    records: tuple[TUEVMorphologyRecordMetadata, ...]

    def __init__(
        self,
        *,
        _marker: object,
        path: Path,
        bundle_manifest_sha256: str,
        preflight_receipt_sha256: str,
        external_metadata_sha256: str,
        source_roster_sha256: str,
        producer_source_sha256: str,
        preprocessing_policy_sha256: str,
        standard19_mapping_policy_sha256: str,
        duplicate_ledger_sha256: str,
        duplicate_ledger: TUEVExactSignalDuplicateLedger,
        records: Sequence[TUEVMorphologyRecordMetadata],
    ) -> None:
        if _marker is not _VERIFIED_TUEV_MORPHOLOGY_PREFLIGHT_MARKER:
            raise TypeError(
                "VerifiedTUEVMorphologyPreflight can only be issued by the "
                "strict signal-preflight loader"
            )
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Verified preflight path must be absolute")
        values: dict[str, object] = {
            "path": path,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "preflight_receipt_sha256": preflight_receipt_sha256,
            "external_metadata_sha256": external_metadata_sha256,
            "source_roster_sha256": source_roster_sha256,
            "producer_source_sha256": producer_source_sha256,
            "preprocessing_policy_sha256": preprocessing_policy_sha256,
            "standard19_mapping_policy_sha256": standard19_mapping_policy_sha256,
            "duplicate_ledger_sha256": duplicate_ledger_sha256,
            "duplicate_ledger": duplicate_ledger,
            "records": tuple(records),
        }
        for field in (
            "bundle_manifest_sha256",
            "preflight_receipt_sha256",
            "external_metadata_sha256",
            "source_roster_sha256",
            "producer_source_sha256",
            "preprocessing_policy_sha256",
            "standard19_mapping_policy_sha256",
            "duplicate_ledger_sha256",
        ):
            _sha(values[field], field=field)
        record_values = values["records"]
        ledger = values["duplicate_ledger"]
        if not isinstance(ledger, TUEVExactSignalDuplicateLedger):
            raise TypeError("Verified preflight duplicate ledger is not typed")
        if values["duplicate_ledger_sha256"] != ledger.ledger_sha256:
            raise ValueError("Verified preflight duplicate-ledger SHA changed")
        if values["source_roster_sha256"] != ledger.source_roster_sha256:
            raise ValueError("Verified preflight ledger belongs to another roster")
        if not isinstance(record_values, tuple) or not record_values:
            raise ValueError("Verified preflight records cannot be empty")
        if any(
            not isinstance(item, TUEVMorphologyRecordMetadata)
            for item in record_values
        ):
            raise TypeError("Verified preflight contains an invalid metadata row")
        paths = tuple(item.relative_edf_path for item in record_values)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("Verified preflight metadata paths must be unique and sorted")
        for field, value in values.items():
            object.__setattr__(self, field, value)

    @property
    def metadata_by_relative_edf(self) -> dict[str, TUEVMorphologyRecordMetadata]:
        return {record.relative_edf_path: record for record in self.records}


@dataclass(frozen=True)
class TUEVMorphologyTarget:
    target_id: str
    record_id: str
    edge_index: int
    label_index: int
    label_name: str
    start_sample: int
    stop_sample: int
    source_line: int
    overlap_component_id: str
    overlap_component_size: int

    def __post_init__(self) -> None:
        _text(self.target_id, field="target_id")
        _text(self.record_id, field="record_id")
        if not 0 <= self.edge_index < len(TCP_20_EDGES):
            raise ValueError("edge_index must identify common TCP20")
        if not 0 <= self.label_index < len(MORPHOLOGY_CLASSES):
            raise ValueError("label_index must identify native CE6")
        if self.label_name != MORPHOLOGY_CLASSES[self.label_index]:
            raise ValueError("label_name disagrees with native CE6 order")
        _integer(self.start_sample, field="start_sample")
        if self.stop_sample != self.start_sample + MORPHOLOGY_TARGET_SAMPLES:
            raise ValueError("Every canonical TUEV target must span exactly 200 samples")
        _integer(self.source_line, field="source_line", minimum=1)
        _text(self.overlap_component_id, field="overlap_component_id")
        _integer(self.overlap_component_size, field="overlap_component_size", minimum=1)

    @property
    def component_weight(self) -> float:
        return 1.0 / self.overlap_component_size

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "record_id": self.record_id,
            "edge_index": self.edge_index,
            "label_index": self.label_index,
            "label_name": self.label_name,
            "start_sample": self.start_sample,
            "stop_sample": self.stop_sample,
            "source_line": self.source_line,
            "overlap_component_id": self.overlap_component_id,
            "overlap_component_size": self.overlap_component_size,
        }


@dataclass(frozen=True)
class TUEVMorphologyIntervalGroup:
    crop_id: str
    record_id: str
    parent_group_id: str
    edf_sha256: str
    start_sample: int
    stop_sample: int
    targets: tuple[TUEVMorphologyTarget, ...]
    source_target_mask_sha256: str

    def __post_init__(self) -> None:
        for field in ("crop_id", "record_id", "parent_group_id"):
            _text(getattr(self, field), field=field)
        _sha(self.edf_sha256, field="edf_sha256")
        _integer(self.start_sample, field="start_sample")
        if self.stop_sample != self.start_sample + MORPHOLOGY_CONTEXT_SAMPLES:
            raise ValueError("Morphology source crops must contain exactly four seconds")
        if not self.targets:
            raise ValueError("An interval group must contain at least one target")
        ordered = tuple(sorted(self.targets, key=lambda item: item.edge_index))
        if ordered != self.targets or len({item.edge_index for item in ordered}) != len(ordered):
            raise ValueError("Interval-group targets must have unique sorted edges")
        if any(
            target.record_id != self.record_id or target.start_sample != self.start_sample
            for target in self.targets
        ):
            raise ValueError("Interval-group targets disagree with their signal crop")
        expected_mask_sha = _canonical_sha256(
            tuple((target.edge_index, 0) for target in self.targets)
        )
        if self.source_target_mask_sha256 != expected_mask_sha:
            raise ValueError("source_target_mask_sha256 disagrees with slot-0 targets")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "crop_id": self.crop_id,
            "record_id": self.record_id,
            "parent_group_id": self.parent_group_id,
            "edf_sha256": self.edf_sha256,
            "start_sample": self.start_sample,
            "stop_sample": self.stop_sample,
            "targets": [target.canonical_payload for target in self.targets],
            "source_target_mask_sha256": self.source_target_mask_sha256,
        }


@dataclass(frozen=True)
class TUEVMorphologyOmission:
    record_id: str
    reason_code: str
    source_line: int | None
    edge_index: int | None
    label_index: int | None

    def __post_init__(self) -> None:
        _text(self.record_id, field="record_id")
        if self.reason_code not in _OMISSION_CODES:
            raise ValueError(f"Unknown morphology omission code: {self.reason_code!r}")
        if self.source_line is not None:
            _integer(self.source_line, field="source_line", minimum=1)
        if self.edge_index is not None and not 0 <= self.edge_index < len(TCP_20_EDGES):
            raise ValueError("Omission edge_index is invalid")
        if self.label_index is not None and not 0 <= self.label_index < len(MORPHOLOGY_CLASSES):
            raise ValueError("Omission label_index is invalid")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "reason_code": self.reason_code,
            "source_line": self.source_line,
            "edge_index": self.edge_index,
            "label_index": self.label_index,
        }


@dataclass(frozen=True)
class TUEVMorphologyRecordReceipt:
    record_id: str
    relative_edf_path: str
    relative_rec_path: str
    official_split: str
    parent_group_id: str
    group_kind: str
    source_subject_id: str | None
    edf_sha256: str
    rec_sha256: str
    derivative_files: tuple[tuple[str, str], ...]
    parent_group_files: tuple[tuple[str, str], ...]
    group_file_roster_sha256: str
    metadata: TUEVMorphologyRecordMetadata

    def __post_init__(self) -> None:
        _text(self.record_id, field="record_id")
        _relative_path(self.relative_edf_path, field="relative_edf_path")
        _relative_path(self.relative_rec_path, field="relative_rec_path")
        if self.official_split == "train":
            if self.group_kind != TRAIN_GROUP_KIND or self.source_subject_id is None:
                raise ValueError("Train receipt lost verified-subject grouping")
        elif self.official_split == "eval":
            if self.group_kind != EVAL_GROUP_KIND or self.source_subject_id is not None:
                raise ValueError("Eval receipt must remain an official session/index")
        else:
            raise ValueError("official_split must be train or eval")
        _text(self.parent_group_id, field="parent_group_id")
        for field in ("edf_sha256", "rec_sha256", "group_file_roster_sha256"):
            _sha(getattr(self, field), field=field)
        if self.metadata.relative_edf_path != self.relative_edf_path:
            raise ValueError("Record metadata is attached to a different EDF path")
        if self.metadata.edf_sha256 != self.edf_sha256:
            raise ValueError("Record metadata is attached to different EDF bytes")
        for field, roster in (
            ("derivative_files", self.derivative_files),
            ("parent_group_files", self.parent_group_files),
        ):
            paths = tuple(path for path, _ in roster)
            if paths != tuple(sorted(set(paths))):
                raise ValueError(f"{field} receipt paths must be unique and sorted")
            for path, digest in roster:
                _relative_path(path, field=f"{field}.path")
                _sha(digest, field=f"{field}.sha256")
        if self.group_file_roster_sha256 != _canonical_sha256(
            self.parent_group_files
        ):
            raise ValueError("Parent-group receipt SHA is not reproducible")
        parent_paths = {path for path, _ in self.parent_group_files}
        if self.relative_edf_path not in parent_paths or self.relative_rec_path not in parent_paths:
            raise ValueError("Parent-group receipt omits the record EDF/REC")
        if not {path for path, _ in self.derivative_files} <= parent_paths:
            raise ValueError("Record derivatives are absent from its parent receipt")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "relative_edf_path": self.relative_edf_path,
            "relative_rec_path": self.relative_rec_path,
            "official_split": self.official_split,
            "parent_group_id": self.parent_group_id,
            "group_kind": self.group_kind,
            "source_subject_id": self.source_subject_id,
            "edf_sha256": self.edf_sha256,
            "rec_sha256": self.rec_sha256,
            "derivative_files": [list(item) for item in self.derivative_files],
            "parent_group_files": [list(item) for item in self.parent_group_files],
            "group_file_roster_sha256": self.group_file_roster_sha256,
            "metadata": self.metadata.canonical_payload,
        }


@dataclass(frozen=True)
class TUEVMorphologyManifest(Sequence[TUEVMorphologyIntervalGroup]):
    records: tuple[TUEVMorphologyRecordReceipt, ...]
    interval_groups: tuple[TUEVMorphologyIntervalGroup, ...]
    omissions: tuple[TUEVMorphologyOmission, ...]
    duplicate_ledger: TUEVExactSignalDuplicateLedger
    duplicate_ledger_sha256: str
    cohort_authorization_sha256: str
    global_ledger_sha256: str
    preprocessing_policy_sha256: str
    signal_preflight_receipt_sha256: str
    external_metadata_sha256: str
    preflight_producer_source_sha256: str
    standard19_mapping_policy_sha256: str
    source_roster_sha256: str
    count_semantics: str
    derived_from_manifest_sha256: str | None
    eligible_group_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    held_group_ids: tuple[str, ...]
    excluded_group_ids: tuple[str, ...]
    holding_reference_target_count: int | None = None
    schema_version: str = TUEV_MORPHOLOGY_MANIFEST_SCHEMA
    policy_version: str = TUEV_MORPHOLOGY_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != TUEV_MORPHOLOGY_MANIFEST_SCHEMA:
            raise ValueError("Unexpected TUEV morphology manifest schema")
        if self.policy_version != TUEV_MORPHOLOGY_POLICY:
            raise ValueError("TUEV morphology policy cannot be changed")
        _sha(
            self.cohort_authorization_sha256,
            field="cohort_authorization_sha256",
        )
        _sha(self.global_ledger_sha256, field="global_ledger_sha256")
        _sha(self.preprocessing_policy_sha256, field="preprocessing_policy_sha256")
        for field in (
            "duplicate_ledger_sha256",
            "signal_preflight_receipt_sha256",
            "external_metadata_sha256",
            "preflight_producer_source_sha256",
            "standard19_mapping_policy_sha256",
            "source_roster_sha256",
        ):
            _sha(getattr(self, field), field=field)
        if not isinstance(
            self.duplicate_ledger, TUEVExactSignalDuplicateLedger
        ):
            raise TypeError("Manifest duplicate ledger must be typed")
        if self.duplicate_ledger_sha256 != self.duplicate_ledger.ledger_sha256:
            raise ValueError("Manifest duplicate-ledger SHA is not reproducible")
        if self.source_roster_sha256 != self.duplicate_ledger.source_roster_sha256:
            raise ValueError("Manifest duplicate ledger belongs to another source roster")
        if self.derived_from_manifest_sha256 is not None:
            _sha(self.derived_from_manifest_sha256, field="derived_from_manifest_sha256")
        record_ids = tuple(record.record_id for record in self.records)
        if record_ids != tuple(sorted(set(record_ids))):
            raise ValueError("Manifest records must be unique and sorted")
        crop_ids = tuple(group.crop_id for group in self.interval_groups)
        if crop_ids != tuple(sorted(set(crop_ids))):
            raise ValueError("Interval groups must be unique and sorted")
        record_by_id = {record.record_id: record for record in self.records}
        decisions = self.duplicate_ledger.decision_by_record_id
        if set(decisions) != set(record_by_id):
            raise ValueError("Manifest records differ from duplicate-ledger records")
        for record_id, record in record_by_id.items():
            decision = decisions[record_id]
            if (
                decision.group_id != record.parent_group_id
                or decision.official_split != record.official_split
                or decision.edf_sha256 != record.edf_sha256
                or decision.rec_sha256 != record.rec_sha256
            ):
                raise ValueError("Manifest record and duplicate-ledger identity disagree")
        if any(group.record_id not in record_by_id for group in self.interval_groups):
            raise ValueError("An interval group refers to an unknown record")
        for group in self.interval_groups:
            record = record_by_id[group.record_id]
            if (
                group.parent_group_id != record.parent_group_id
                or group.edf_sha256 != record.edf_sha256
            ):
                raise ValueError("An interval group was swapped across record/parent receipts")
        if any(omission.record_id not in record_by_id for omission in self.omissions):
            raise ValueError("An omission refers to an unknown record")
        all_parent_groups = {record.parent_group_id for record in self.records}
        rosters = (
            self.eligible_group_ids,
            self.fit_group_ids,
            self.held_group_ids,
            self.excluded_group_ids,
        )
        if any(roster != tuple(sorted(set(roster))) for roster in rosters):
            raise ValueError("All group rosters must be sorted and duplicate-free")
        eligible = set(self.eligible_group_ids)
        fit = set(self.fit_group_ids)
        held = set(self.held_group_ids)
        excluded = set(self.excluded_group_ids)
        if eligible & excluded or fit & held or fit & excluded or held & excluded:
            raise ValueError("Fit/held/eligible/excluded group roles overlap")
        if eligible | excluded != all_parent_groups:
            raise ValueError("Eligible/excluded rosters must account for every parent group")
        role_by_group = {
            group_id: (
                "excluded"
                if group_id in excluded
                else "fit"
                if group_id in fit
                else "held"
                if group_id in held
                else "eligible"
            )
            for group_id in all_parent_groups
        }
        _require_content_component_closed_roles(
            self.duplicate_ledger, role_by_group
        )
        required_quarantine = set(
            self.duplicate_ledger.cross_split_quarantined_group_ids
        )
        if not required_quarantine <= excluded:
            raise ValueError(
                "A train/eval exact-EDF-byte component must be wholly excluded"
            )
        eval_groups = {
            record.parent_group_id
            for record in self.records
            if record.official_split == "eval"
        }
        if fit & eval_groups:
            raise ValueError("Official TUEV evaluation sessions can never enter fitting")
        parent_receipts: dict[
            str, tuple[str, str, str | None, tuple[tuple[str, str], ...], str]
        ] = {}
        for record in self.records:
            identity = (
                record.official_split,
                record.group_kind,
                record.source_subject_id,
                record.parent_group_files,
                record.group_file_roster_sha256,
            )
            previous = parent_receipts.setdefault(record.parent_group_id, identity)
            if previous != identity:
                raise ValueError("One parent group has contradictory file/split receipts")
        groups_with_targets = {group.parent_group_id for group in self.interval_groups}
        if groups_with_targets & excluded:
            raise ValueError("An excluded parent group produced a morphology target")
        quarantined_records = {
            record_id
            for record_id, decision in decisions.items()
            if decision.quarantined
        }
        if any(
            group.record_id in quarantined_records
            for group in self.interval_groups
        ):
            raise ValueError("A quarantined exact-EDF-byte record produced a target")
        if self.count_semantics == HOLDING_COUNT_SEMANTICS:
            if fit or held or self.derived_from_manifest_sha256 is not None:
                raise ValueError("Holding audit manifests cannot claim fold fit/held rosters")
            if self.holding_reference_target_count is not None:
                _integer(
                    self.holding_reference_target_count,
                    field="holding_reference_target_count",
                    minimum=1,
                )
                if self.holding_reference_target_count != (
                    TUEV_MORPHOLOGY_HOLDING_TARGET_UPPER_BOUND
                ):
                    raise ValueError("Holding reference count disagrees with the frozen audit")
        elif self.count_semantics == FOLD_COUNT_SEMANTICS:
            if self.derived_from_manifest_sha256 is None:
                raise ValueError("Fold-final manifests require their holding-manifest parent SHA")
            if fit | held != eligible:
                raise ValueError("Fold fit/held rosters must partition eligible groups")
            if not fit or not held:
                raise ValueError("Fold-final manifests require non-empty fit and held rosters")
            if self.holding_reference_target_count is not None:
                raise ValueError(
                    "The global 58,722 holding count cannot be claimed as a fold-final count"
                )
        else:
            raise ValueError("Unknown morphology count semantics")

    def __len__(self) -> int:
        return len(self.interval_groups)

    def __getitem__(self, index: int) -> TUEVMorphologyIntervalGroup:
        return self.interval_groups[index]

    def __iter__(self) -> Iterator[TUEVMorphologyIntervalGroup]:
        return iter(self.interval_groups)

    @property
    def target_count(self) -> int:
        return sum(len(group.targets) for group in self.interval_groups)

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "records": [record.canonical_payload for record in self.records],
            "interval_groups": [group.canonical_payload for group in self.interval_groups],
            "omissions": [item.canonical_payload for item in self.omissions],
            "duplicate_ledger": self.duplicate_ledger.canonical_payload,
            "duplicate_ledger_sha256": self.duplicate_ledger_sha256,
            "cohort_authorization_sha256": (
                self.cohort_authorization_sha256
            ),
            "global_ledger_sha256": self.global_ledger_sha256,
            "preprocessing_policy_sha256": self.preprocessing_policy_sha256,
            "signal_preflight_receipt_sha256": (
                self.signal_preflight_receipt_sha256
            ),
            "external_metadata_sha256": self.external_metadata_sha256,
            "preflight_producer_source_sha256": (
                self.preflight_producer_source_sha256
            ),
            "standard19_mapping_policy_sha256": (
                self.standard19_mapping_policy_sha256
            ),
            "source_roster_sha256": self.source_roster_sha256,
            "count_semantics": self.count_semantics,
            "derived_from_manifest_sha256": self.derived_from_manifest_sha256,
            "eligible_group_ids": list(self.eligible_group_ids),
            "fit_group_ids": list(self.fit_group_ids),
            "held_group_ids": list(self.held_group_ids),
            "excluded_group_ids": list(self.excluded_group_ids),
            "holding_reference_target_count": self.holding_reference_target_count,
        }

    @property
    def manifest_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


@dataclass(frozen=True)
class _Candidate:
    interval: TUEVInterval
    start_sample: int
    stop_sample: int


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _omission(
    record_id: str,
    reason: str,
    interval: TUEVInterval | None = None,
) -> TUEVMorphologyOmission:
    return TUEVMorphologyOmission(
        record_id=record_id,
        reason_code=reason,
        source_line=None if interval is None else interval.source_line,
        edge_index=None if interval is None else interval.modern_edge_index,
        label_index=None if interval is None else interval.label_index,
    )


def _canonical_candidates(
    source: TUEVMorphologySourceRecord,
    metadata: TUEVMorphologyRecordMetadata,
) -> tuple[list[_Candidate], list[TUEVMorphologyOmission]]:
    annotation = parse_tuev_rec(source.rec_path)
    if annotation.receipt.rec_sha256 != source.rec_sha256:
        raise ValueError("TUEV REC bytes changed after discovery")
    candidates: list[_Candidate] = []
    omissions: list[TUEVMorphologyOmission] = []
    for interval in annotation.intervals:
        if interval.modern_edge_index is None:
            omissions.append(_omission(source.record_id, "outside_common20", interval))
            continue
        duration = interval.stop_sec - interval.start_sec
        if abs(duration - 1.0) > MORPHOLOGY_DURATION_TOLERANCE_SEC:
            omissions.append(_omission(source.record_id, "duration_tolerance", interval))
            continue
        start_sample = _round_half_up(interval.start_sec * MORPHOLOGY_OUTPUT_SFREQ_HZ)
        aligned = start_sample / MORPHOLOGY_OUTPUT_SFREQ_HZ
        if abs(aligned - interval.start_sec) > MORPHOLOGY_ALIGNMENT_TOLERANCE_SEC + 1e-12:
            omissions.append(_omission(source.record_id, "sample_alignment", interval))
            continue
        candidates.append(
            _Candidate(
                interval=interval,
                start_sample=start_sample,
                stop_sample=start_sample + MORPHOLOGY_TARGET_SAMPLES,
            )
        )

    unique: dict[tuple[int, int, int, int], _Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.interval.source_line):
        key = (
            int(candidate.interval.modern_edge_index),
            candidate.start_sample,
            candidate.stop_sample,
            candidate.interval.label_index,
        )
        if key in unique:
            omissions.append(
                _omission(source.record_id, "exact_duplicate", candidate.interval)
            )
        else:
            unique[key] = candidate
    candidates = list(unique.values())

    conflicted: set[int] = set()
    for left in range(len(candidates)):
        left_interval = candidates[left].interval
        for right in range(left + 1, len(candidates)):
            right_interval = candidates[right].interval
            if left_interval.modern_edge_index != right_interval.modern_edge_index:
                continue
            if left_interval.label_index == right_interval.label_index:
                continue
            overlap = min(left_interval.stop_sec, right_interval.stop_sec) - max(
                left_interval.start_sec, right_interval.start_sec
            )
            if overlap > MORPHOLOGY_OVERLAP_TOLERANCE_SEC:
                conflicted.update((left, right))
    clean: list[_Candidate] = []
    for index, candidate in enumerate(candidates):
        if index in conflicted:
            omissions.append(
                _omission(
                    source.record_id,
                    "same_edge_cross_class_overlap",
                    candidate.interval,
                )
            )
        elif candidate.start_sample < MORPHOLOGY_WARMUP_SAMPLES:
            omissions.append(
                _omission(source.record_id, "insufficient_warmup", candidate.interval)
            )
        elif candidate.start_sample + MORPHOLOGY_CONTEXT_SAMPLES > metadata.output_sample_count:
            omissions.append(
                _omission(
                    source.record_id,
                    "insufficient_post_context",
                    candidate.interval,
                )
            )
        else:
            clean.append(candidate)
    return clean, omissions


def _component_assignments(
    record_id: str, candidates: Sequence[_Candidate]
) -> tuple[tuple[str, int], ...]:
    _text(record_id, field="record_id")
    dsu = _DisjointSet(len(candidates))
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            a, b = candidates[left], candidates[right]
            if (
                a.interval.modern_edge_index != b.interval.modern_edge_index
                or a.interval.label_index != b.interval.label_index
            ):
                continue
            overlap_samples = min(a.stop_sample, b.stop_sample) - max(
                a.start_sample, b.start_sample
            )
            if overlap_samples / MORPHOLOGY_OUTPUT_SFREQ_HZ > (
                MORPHOLOGY_OVERLAP_TOLERANCE_SEC
            ):
                dsu.union(left, right)
    members: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        members.setdefault(dsu.find(index), []).append(index)
    assignments: list[tuple[str, int] | None] = [None] * len(candidates)
    for indices in members.values():
        descriptor = (
            record_id,
            tuple(
                sorted(
                    (
                        int(candidates[index].interval.modern_edge_index),
                        candidates[index].interval.label_index,
                        candidates[index].start_sample,
                    )
                    for index in indices
                )
            ),
        )
        component_id = f"overlap:{_canonical_sha256(descriptor)}"
        for index in indices:
            assignments[index] = (component_id, len(indices))
    if any(value is None for value in assignments):
        raise RuntimeError("Overlap-component assignment is incomplete")
    return tuple(value for value in assignments if value is not None)


def _source_roster_payload(
    sources: Sequence[TUEVMorphologySourceRecord],
) -> tuple[tuple[object, ...], ...]:
    """Canonical source/group identity used by the external preflight gate."""

    ordered = tuple(sorted(sources, key=lambda item: item.record_id))
    return tuple(
        (
            source.record_id,
            source.relative_edf_path,
            source.relative_rec_path,
            source.official_split,
            source.group_id,
            source.group_kind,
            source.source_subject_id,
            source.edf_sha256,
            source.rec_sha256,
            source.group_file_roster_sha256,
        )
        for source in ordered
    )


_PUBLIC_PROTOCOL_MARKER = object()
_COHORT_AUTHORIZATION_MARKER = object()


@dataclass(frozen=True, init=False)
class VerifiedTUEVMorphologyPublicProtocol:
    """Opaque binding issued only by the strict public-artifact loaders.

    A bare :class:`TUSZDeepSOZPublicLedgerArtifact` or
    :class:`IctalConceptOOFProtocolArtifact` is intentionally insufficient at
    the morphology formal boundary: both legacy dataclasses can be constructed
    in memory.  This capability is issued only after the canonical public
    ledger and OOF bundles have been reloaded and cross-checked together.
    """

    public_ledger_artifact: TUSZDeepSOZPublicLedgerArtifact
    oof_protocol_artifact: IctalConceptOOFProtocolArtifact

    def __init__(
        self,
        *,
        _marker: object,
        public_ledger_artifact: TUSZDeepSOZPublicLedgerArtifact,
        oof_protocol_artifact: IctalConceptOOFProtocolArtifact,
    ) -> None:
        if _marker is not _PUBLIC_PROTOCOL_MARKER:
            raise PermissionError(
                "TUEV public-protocol binding must come from the strict loader"
            )
        if not isinstance(
            public_ledger_artifact, TUSZDeepSOZPublicLedgerArtifact
        ):
            raise TypeError("public_ledger_artifact has the wrong type")
        if not isinstance(
            oof_protocol_artifact, IctalConceptOOFProtocolArtifact
        ):
            raise TypeError("oof_protocol_artifact has the wrong type")
        if (
            oof_protocol_artifact.public_ledger_build_sha256
            != public_ledger_artifact.build_sha256
        ):
            raise ValueError(
                "OOF protocol was not rebuilt from the supplied public ledger"
            )
        object.__setattr__(
            self, "public_ledger_artifact", public_ledger_artifact
        )
        object.__setattr__(
            self, "oof_protocol_artifact", oof_protocol_artifact
        )


def load_tuev_morphology_public_protocol(
    public_ledger_bundle: str | Path,
    oof_protocol_bundle: str | Path,
    registry: DeepSOZReferenceRegistry,
    *,
    expected_public_ledger_bundle_sha256: str,
    expected_public_ledger_build_sha256: str,
    expected_oof_protocol_artifact_sha256: str,
    expected_oof_protocol_sha256: str,
) -> VerifiedTUEVMorphologyPublicProtocol:
    """Strictly load the public ledger and its reconstructed OOF protocol."""

    if not isinstance(registry, DeepSOZReferenceRegistry):
        raise TypeError("registry must be a DeepSOZReferenceRegistry")
    public_artifact = load_tusz_deepsoz_public_ledger_build(
        public_ledger_bundle,
        expected_bundle_sha256=_sha(
            expected_public_ledger_bundle_sha256,
            field="expected_public_ledger_bundle_sha256",
        ),
        expected_build_sha256=_sha(
            expected_public_ledger_build_sha256,
            field="expected_public_ledger_build_sha256",
        ),
    )
    protocol_artifact = load_ictal_concept_oof_protocol(
        oof_protocol_bundle,
        registry,
        public_artifact,
        expected_artifact_sha256=_sha(
            expected_oof_protocol_artifact_sha256,
            field="expected_oof_protocol_artifact_sha256",
        ),
        expected_protocol_sha256=_sha(
            expected_oof_protocol_sha256,
            field="expected_oof_protocol_sha256",
        ),
    )
    return VerifiedTUEVMorphologyPublicProtocol(
        _marker=_PUBLIC_PROTOCOL_MARKER,
        public_ledger_artifact=public_artifact,
        oof_protocol_artifact=protocol_artifact,
    )


def issue_tuev_morphology_public_protocol_for_testing(
    public_ledger_artifact: TUSZDeepSOZPublicLedgerArtifact,
    oof_protocol_artifact: IctalConceptOOFProtocolArtifact,
) -> VerifiedTUEVMorphologyPublicProtocol:
    """Explicit test-only issuer; production CLIs must never call this helper."""

    return VerifiedTUEVMorphologyPublicProtocol(
        _marker=_PUBLIC_PROTOCOL_MARKER,
        public_ledger_artifact=public_ledger_artifact,
        oof_protocol_artifact=oof_protocol_artifact,
    )


@dataclass(frozen=True, init=False)
class VerifiedTUEVMorphologyCohortAuthorization:
    """Shared fit/held/excluded group authority for morphology production.

    The capability contains no free caller roster.  It is derived from one
    strictly loaded public ledger/OOF protocol pair and a live replay of the
    complete TUEV source tree.  Exact-EDF-connected groups always share one
    role, and official TUEV evaluation sessions can only be held or excluded.
    """

    public_ledger_bundle_sha256: str
    public_ledger_build_sha256: str
    public_ledger_sha256: str
    public_ledger_receipt_sha256: str
    oof_protocol_artifact_sha256: str
    oof_protocol_sha256: str
    oof_plan_union_sha256: str
    cohort_bindings: tuple[tuple[str, str], ...]
    source_roster_sha256: str
    duplicate_ledger_sha256: str
    fit_group_ids: tuple[str, ...]
    held_group_ids: tuple[str, ...]
    excluded_group_ids: tuple[str, ...]
    exclusion_reasons: tuple[tuple[str, tuple[str, ...]], ...]
    assignment_policy: str = TUEV_MORPHOLOGY_GROUP_ASSIGNMENT_POLICY
    public_content_policy: str = TUEV_MORPHOLOGY_PUBLIC_CONTENT_POLICY
    schema_version: str = TUEV_MORPHOLOGY_COHORT_AUTHORIZATION_SCHEMA

    def __init__(
        self,
        *,
        _marker: object,
        public_ledger_bundle_sha256: str,
        public_ledger_build_sha256: str,
        public_ledger_sha256: str,
        public_ledger_receipt_sha256: str,
        oof_protocol_artifact_sha256: str,
        oof_protocol_sha256: str,
        oof_plan_union_sha256: str,
        cohort_bindings: tuple[tuple[str, str], ...],
        source_roster_sha256: str,
        duplicate_ledger_sha256: str,
        fit_group_ids: tuple[str, ...],
        held_group_ids: tuple[str, ...],
        excluded_group_ids: tuple[str, ...],
        exclusion_reasons: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> None:
        if _marker is not _COHORT_AUTHORIZATION_MARKER:
            raise PermissionError(
                "TUEV morphology cohort authorization must be derived"
            )
        digest_fields = {
            "public_ledger_bundle_sha256": public_ledger_bundle_sha256,
            "public_ledger_build_sha256": public_ledger_build_sha256,
            "public_ledger_sha256": public_ledger_sha256,
            "public_ledger_receipt_sha256": public_ledger_receipt_sha256,
            "oof_protocol_artifact_sha256": oof_protocol_artifact_sha256,
            "oof_protocol_sha256": oof_protocol_sha256,
            "oof_plan_union_sha256": oof_plan_union_sha256,
            "source_roster_sha256": source_roster_sha256,
            "duplicate_ledger_sha256": duplicate_ledger_sha256,
        }
        for field_name, value in digest_fields.items():
            digest_fields[field_name] = _sha(value, field=field_name)
        bindings = tuple(
            (
                _text(split, field="cohort_binding.split"),
                _sha(digest, field="cohort_binding.receipt_sha256"),
            )
            for split, digest in cohort_bindings
        )
        if not bindings or bindings != tuple(sorted(set(bindings))):
            raise ValueError("cohort_bindings must be non-empty, unique and sorted")
        rosters = tuple(
            tuple(sorted(set(values)))
            for values in (fit_group_ids, held_group_ids, excluded_group_ids)
        )
        fit, held, excluded = map(set, rosters)
        if fit & held or fit & excluded or held & excluded:
            raise ValueError("Authorized group roles must be disjoint")
        if not fit or not held:
            raise ValueError("Authorized fit and held roles must both be non-empty")
        normalized_reasons = tuple(
            (
                _text(group_id, field="exclusion_reasons.group_id"),
                tuple(sorted(set(str(reason) for reason in reasons))),
            )
            for group_id, reasons in exclusion_reasons
        )
        if tuple(group for group, _ in normalized_reasons) != tuple(
            sorted(set(group for group, _ in normalized_reasons))
        ):
            raise ValueError("Exclusion-reason groups must be unique and sorted")
        if {group for group, _ in normalized_reasons} != excluded:
            raise ValueError("Every excluded group requires an exact reason receipt")
        if any(not reasons or any(not reason for reason in reasons) for _, reasons in normalized_reasons):
            raise ValueError("Every excluded group requires non-empty reasons")
        for field_name, value in digest_fields.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "cohort_bindings", bindings)
        object.__setattr__(self, "fit_group_ids", rosters[0])
        object.__setattr__(self, "held_group_ids", rosters[1])
        object.__setattr__(self, "excluded_group_ids", rosters[2])
        object.__setattr__(self, "exclusion_reasons", normalized_reasons)
        object.__setattr__(
            self, "assignment_policy", TUEV_MORPHOLOGY_GROUP_ASSIGNMENT_POLICY
        )
        object.__setattr__(
            self, "public_content_policy", TUEV_MORPHOLOGY_PUBLIC_CONTENT_POLICY
        )
        object.__setattr__(
            self, "schema_version", TUEV_MORPHOLOGY_COHORT_AUTHORIZATION_SCHEMA
        )

    @property
    def eligible_group_ids(self) -> tuple[str, ...]:
        return tuple(sorted((*self.fit_group_ids, *self.held_group_ids)))

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "public_ledger_bundle_sha256": self.public_ledger_bundle_sha256,
            "public_ledger_build_sha256": self.public_ledger_build_sha256,
            "public_ledger_sha256": self.public_ledger_sha256,
            "public_ledger_receipt_sha256": self.public_ledger_receipt_sha256,
            "oof_protocol_artifact_sha256": self.oof_protocol_artifact_sha256,
            "oof_protocol_sha256": self.oof_protocol_sha256,
            "oof_plan_union_sha256": self.oof_plan_union_sha256,
            "cohort_bindings": [list(item) for item in self.cohort_bindings],
            "source_roster_sha256": self.source_roster_sha256,
            "duplicate_ledger_sha256": self.duplicate_ledger_sha256,
            "fit_group_ids": list(self.fit_group_ids),
            "held_group_ids": list(self.held_group_ids),
            "excluded_group_ids": list(self.excluded_group_ids),
            "exclusion_reasons": [
                [group_id, list(reasons)]
                for group_id, reasons in self.exclusion_reasons
            ],
            "assignment_policy": self.assignment_policy,
            "public_content_policy": self.public_content_policy,
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


def _protected_public_identity_sets(
    context: VerifiedTUEVMorphologyPublicProtocol,
) -> tuple[set[str], set[str], set[str]]:
    """Protect the global union of every eligible DeepSOZ public identity."""

    ledger = context.public_ledger_artifact.ledger
    protocol = context.oof_protocol_artifact.protocol
    eligible_public_keys = {
        public_key
        for _, public_key in protocol.receipt.target_public_crosswalk
    }
    protected_records = {
        record
        for record in ledger
        if record.split in {"dev", "eval"}
        or (
            record.dataset == "deepsoz"
            and record.patient_key in eligible_public_keys
        )
    }
    return (
        eligible_public_keys
        | {record.patient_key for record in protected_records},
        {record.file_sha256 for record in protected_records},
        {record.signal_content_sha256 for record in protected_records},
    )


def authorize_tuev_morphology_cohort(
    sources: Sequence[TUEVMorphologySourceRecord],
    public_protocol: VerifiedTUEVMorphologyPublicProtocol,
    *,
    replay_live_source: bool = True,
) -> VerifiedTUEVMorphologyCohortAuthorization:
    """Derive one shared TUEV producer roster for every target OOF selection."""

    if not isinstance(public_protocol, VerifiedTUEVMorphologyPublicProtocol):
        raise TypeError(
            "public_protocol must be issued by the strict public-protocol loader"
        )
    if not isinstance(replay_live_source, bool):
        raise TypeError("replay_live_source must be bool")
    ordered_sources = tuple(sorted(sources, key=lambda item: item.record_id))
    if not ordered_sources:
        raise ValueError("TUEV source roster cannot be empty")
    roots = {source.edf_root for source in ordered_sources}
    if len(roots) != 1:
        raise ValueError("TUEV sources must belong to one canonical root")
    if replay_live_source:
        replayed = discover_tuev_morphology_sources(next(iter(roots)))
        if _source_roster_payload(replayed) != _source_roster_payload(ordered_sources):
            raise ValueError("Live TUEV source roster changed before authorization")
    source_roster_sha = _canonical_sha256(_source_roster_payload(ordered_sources))
    duplicate_ledger = build_tuev_exact_signal_duplicate_ledger(ordered_sources)
    protected_patients, protected_files, protected_content = (
        _protected_public_identity_sets(public_protocol)
    )

    reasons_by_group: dict[str, set[str]] = {}
    all_groups = {source.group_id for source in ordered_sources}
    for group_id in duplicate_ledger.cross_split_quarantined_group_ids:
        reasons_by_group.setdefault(group_id, set()).add(
            "cross_official_split_exact_edf_component"
        )
    for source in ordered_sources:
        reasons: set[str] = set()
        if source.source_subject_id is not None:
            patient_key = normalize_public_patient_key(source.source_subject_id)
            if patient_key in protected_patients:
                reasons.add("protected_target_patient_overlap")
        if source.edf_sha256 in protected_files:
            reasons.add("protected_file_sha_overlap")
        # The current strict TUSZ/DeepSOZ ledger defines signal-content SHA as
        # the complete EDF-container SHA.  Use the identical policy here and
        # disclose the residual re-encoding risk in the authorization receipt.
        if source.edf_sha256 in protected_content:
            reasons.add("protected_signal_content_sha_overlap")
        if reasons:
            reasons_by_group.setdefault(source.group_id, set()).update(reasons)

    # A protected member excludes its complete transitive exact-EDF component.
    for component in duplicate_ledger.group_components:
        component_reasons = set().union(
            *(reasons_by_group.get(group_id, set()) for group_id in component.group_ids)
        )
        if component_reasons:
            component_reasons.add("exact_edf_component_contains_excluded_group")
            for group_id in component.group_ids:
                reasons_by_group.setdefault(group_id, set()).update(
                    component_reasons
                )

    excluded = set(reasons_by_group)
    eligible = all_groups - excluded
    component_by_group = duplicate_ledger.component_by_group_id
    fit: set[str] = set()
    held: set[str] = set()
    for component in duplicate_ledger.group_components:
        component_groups = set(component.group_ids)
        if component_groups <= excluded:
            continue
        if component_groups & excluded:
            raise RuntimeError("Exact-EDF component exclusion propagation failed")
        if not component_groups <= eligible:
            raise RuntimeError("TUEV component escaped the authorized group partition")
        if component.official_splits == ("eval",):
            held.update(component_groups)
            continue
        if component.official_splits != ("train",):
            raise RuntimeError("A cross-split TUEV component was not quarantined")
        fit.update(component_groups)

    role_by_group = {
        group_id: (
            "excluded"
            if group_id in excluded
            else "fit"
            if group_id in fit
            else "held"
            if group_id in held
            else "missing"
        )
        for group_id in all_groups
    }
    if "missing" in role_by_group.values():
        raise RuntimeError("Authorization did not assign every TUEV parent group")
    _require_content_component_closed_roles(duplicate_ledger, role_by_group)
    official_eval_groups = {
        source.group_id
        for source in ordered_sources
        if source.official_split == "eval"
    }
    if fit & official_eval_groups:
        raise RuntimeError("Official TUEV evaluation sessions entered fitting")
    if not fit or not held:
        raise ValueError(
            "Authorization requires non-empty official-train fit and "
            "official-eval held group rosters"
        )
    if fit | held != eligible:
        raise RuntimeError("Fit/held roles do not partition eligible TUEV groups")
    if any(component_by_group[group].crosses_official_split for group in fit | held):
        raise RuntimeError("A cross-split component escaped quarantine")

    public_artifact = public_protocol.public_ledger_artifact
    protocol_artifact = public_protocol.oof_protocol_artifact
    protocol = protocol_artifact.protocol
    plans = (*protocol.fold_plans, protocol.final_plan)
    plan_receipts = tuple(plan.receipt.receipt_sha256 for plan in plans)
    cohort_bindings = tuple(
        sorted(
            {
                binding
                for plan in plans
                for binding in plan.receipt.cohort_bindings
            }
        )
    )
    return VerifiedTUEVMorphologyCohortAuthorization(
        _marker=_COHORT_AUTHORIZATION_MARKER,
        public_ledger_bundle_sha256=public_artifact.bundle_sha256,
        public_ledger_build_sha256=public_artifact.build_sha256,
        public_ledger_sha256=public_artifact.ledger.receipt.ledger_sha256,
        public_ledger_receipt_sha256=(
            public_artifact.ledger.receipt.receipt_sha256
        ),
        oof_protocol_artifact_sha256=protocol_artifact.artifact_sha256,
        oof_protocol_sha256=protocol_artifact.protocol_sha256,
        oof_plan_union_sha256=_canonical_sha256(plan_receipts),
        cohort_bindings=cohort_bindings,
        source_roster_sha256=source_roster_sha,
        duplicate_ledger_sha256=duplicate_ledger.ledger_sha256,
        fit_group_ids=tuple(sorted(fit)),
        held_group_ids=tuple(sorted(held)),
        excluded_group_ids=tuple(sorted(excluded)),
        exclusion_reasons=tuple(
            (group_id, tuple(sorted(reasons_by_group[group_id])))
            for group_id in sorted(excluded)
        ),
    )


def _build_tuev_morphology_manifest_from_roles(
    sources: Sequence[TUEVMorphologySourceRecord],
    preflight: VerifiedTUEVMorphologyPreflight,
    *,
    global_ledger_sha256: str,
    preprocessing_policy_sha256: str,
    cohort_authorization_sha256: str,
    excluded_group_ids: Iterable[str] = (),
    holding_reference_target_count: int | None = None,
) -> TUEVMorphologyManifest:
    """Internal/test-only holding builder after roles were independently issued.

    ``preflight`` must be issued by :func:`load_tuev_morphology_preflight`.
    Bare caller-created metadata mappings are intentionally rejected.  The
    formal first-party producer is independently replayed by the default
    loader; its exact bytes, code/policy hashes, source roster, and EDF signal
    checks are fail-closed inputs.  Geometry or QC failures are omissions,
    never guessed inputs.
    """

    _sha(global_ledger_sha256, field="global_ledger_sha256")
    _sha(preprocessing_policy_sha256, field="preprocessing_policy_sha256")
    _sha(
        cohort_authorization_sha256,
        field="cohort_authorization_sha256",
    )
    if not isinstance(preflight, VerifiedTUEVMorphologyPreflight):
        raise TypeError(
            "preflight must be issued by the strict TUEV morphology loader"
        )
    if preflight.preprocessing_policy_sha256 != preprocessing_policy_sha256:
        raise ValueError(
            "Manifest preprocessing policy differs from the verified preflight"
        )
    ordered_sources = tuple(sorted(sources, key=lambda item: item.record_id))
    if not ordered_sources or len({item.record_id for item in ordered_sources}) != len(
        ordered_sources
    ):
        raise ValueError("sources must be non-empty with unique record IDs")
    roots = {source.edf_root for source in ordered_sources}
    if len(roots) != 1:
        raise ValueError("All morphology sources must belong to one canonical TUEV root")
    replayed_sources = discover_tuev_morphology_sources(next(iter(roots)))
    if _source_roster_payload(replayed_sources) != _source_roster_payload(
        ordered_sources
    ):
        raise ValueError("TUEV source or parent-group roster changed before manifest build")
    source_roster_sha256 = _canonical_sha256(_source_roster_payload(ordered_sources))
    if source_roster_sha256 != preflight.source_roster_sha256:
        raise ValueError("Verified preflight belongs to another TUEV source roster")
    duplicate_ledger = build_tuev_exact_signal_duplicate_ledger(ordered_sources)
    if duplicate_ledger != preflight.duplicate_ledger or (
        duplicate_ledger.ledger_sha256 != preflight.duplicate_ledger_sha256
    ):
        raise ValueError(
            "Verified preflight exact-EDF-byte duplicate ledger changed"
        )
    metadata_by_relative_edf = preflight.metadata_by_relative_edf
    expected_paths = {source.relative_edf_path for source in ordered_sources}
    if set(metadata_by_relative_edf) != expected_paths:
        raise ValueError("Signal metadata must cover exactly the discovered EDF roster")

    all_groups = {source.group_id for source in ordered_sources}
    caller_excluded = set(str(value) for value in excluded_group_ids)
    if not caller_excluded <= all_groups:
        raise ValueError("excluded_group_ids contains an unknown parent group")
    required_quarantine = set(
        duplicate_ledger.cross_split_quarantined_group_ids
    )
    excluded_set = caller_excluded | required_quarantine
    _require_content_component_closed_roles(
        duplicate_ledger,
        {
            group_id: "excluded" if group_id in excluded_set else "eligible"
            for group_id in all_groups
        },
    )
    excluded = tuple(sorted(excluded_set))
    eligible = tuple(sorted(all_groups - excluded_set))
    fit: tuple[str, ...] = ()
    held: tuple[str, ...] = ()

    records: list[TUEVMorphologyRecordReceipt] = []
    groups: list[TUEVMorphologyIntervalGroup] = []
    omissions: list[TUEVMorphologyOmission] = []
    decision_by_record = duplicate_ledger.decision_by_record_id
    for source in ordered_sources:
        if _file_sha256(source.edf_path) != source.edf_sha256:
            raise ValueError("TUEV EDF bytes changed after discovery")
        if _file_sha256(source.rec_path) != source.rec_sha256:
            raise ValueError("TUEV REC bytes changed after discovery")
        metadata = metadata_by_relative_edf[source.relative_edf_path]
        if metadata.relative_edf_path != source.relative_edf_path:
            raise ValueError("Signal metadata path was swapped across records")
        if metadata.edf_sha256 != source.edf_sha256:
            raise ValueError("Signal metadata EDF SHA was swapped across records")
        receipt = TUEVMorphologyRecordReceipt(
            record_id=source.record_id,
            relative_edf_path=source.relative_edf_path,
            relative_rec_path=source.relative_rec_path,
            official_split=source.official_split,
            parent_group_id=source.group_id,
            group_kind=source.group_kind,
            source_subject_id=source.source_subject_id,
            edf_sha256=source.edf_sha256,
            rec_sha256=source.rec_sha256,
            derivative_files=source.derivative_files,
            parent_group_files=source.parent_group_files,
            group_file_roster_sha256=source.group_file_roster_sha256,
            metadata=metadata,
        )
        records.append(receipt)
        decision = decision_by_record[source.record_id]
        if decision.action == "quarantine_cross_official_split_component":
            omissions.append(
                _omission(
                    source.record_id,
                    "cross_official_split_content_component",
                )
            )
            continue
        if decision.action == "quarantine_conflicting_annotation":
            omissions.append(
                _omission(source.record_id, "exact_signal_annotation_conflict")
            )
            continue
        if decision.action == "quarantine_exact_duplicate_alias":
            omissions.append(_omission(source.record_id, "exact_edf_alias"))
            continue
        if decision.action not in {
            "retain_unique",
            "retain_canonical_duplicate",
        }:
            raise RuntimeError("Unhandled exact-EDF-byte duplicate-ledger action")
        if source.group_id in excluded_set:
            omissions.append(_omission(source.record_id, "excluded_by_global_ledger"))
            continue
        if not metadata.direct_standard19 or not metadata.signal_qc_passed:
            omissions.append(_omission(source.record_id, "signal_qc"))
            continue

        candidates, record_omissions = _canonical_candidates(source, metadata)
        omissions.extend(record_omissions)
        assignments = _component_assignments(source.record_id, candidates)
        by_start: dict[int, list[TUEVMorphologyTarget]] = {}
        for candidate, (component_id, component_size) in zip(candidates, assignments):
            edge_index = int(candidate.interval.modern_edge_index)
            target_id = (
                f"{source.record_id}:e{edge_index:02d}:"
                f"s{candidate.start_sample:010d}:c{candidate.interval.label_index}"
            )
            target = TUEVMorphologyTarget(
                target_id=target_id,
                record_id=source.record_id,
                edge_index=edge_index,
                label_index=candidate.interval.label_index,
                label_name=candidate.interval.label_name,
                start_sample=candidate.start_sample,
                stop_sample=candidate.stop_sample,
                source_line=candidate.interval.source_line,
                overlap_component_id=component_id,
                overlap_component_size=component_size,
            )
            by_start.setdefault(candidate.start_sample, []).append(target)
        for start_sample, targets in by_start.items():
            ordered_targets = tuple(sorted(targets, key=lambda item: item.edge_index))
            crop_id = f"{source.record_id}:crop:{start_sample:010d}"
            groups.append(
                TUEVMorphologyIntervalGroup(
                    crop_id=crop_id,
                    record_id=source.record_id,
                    parent_group_id=source.group_id,
                    edf_sha256=source.edf_sha256,
                    start_sample=start_sample,
                    stop_sample=start_sample + MORPHOLOGY_CONTEXT_SAMPLES,
                    targets=ordered_targets,
                    source_target_mask_sha256=_canonical_sha256(
                        tuple((target.edge_index, 0) for target in ordered_targets)
                    ),
                )
            )
    return TUEVMorphologyManifest(
        records=tuple(sorted(records, key=lambda item: item.record_id)),
        interval_groups=tuple(sorted(groups, key=lambda item: item.crop_id)),
        omissions=tuple(
            sorted(
                omissions,
                key=lambda item: (
                    item.record_id,
                    -1 if item.source_line is None else item.source_line,
                    item.reason_code,
                ),
            )
        ),
        duplicate_ledger=duplicate_ledger,
        duplicate_ledger_sha256=duplicate_ledger.ledger_sha256,
        cohort_authorization_sha256=cohort_authorization_sha256,
        global_ledger_sha256=global_ledger_sha256,
        preprocessing_policy_sha256=preprocessing_policy_sha256,
        signal_preflight_receipt_sha256=preflight.preflight_receipt_sha256,
        external_metadata_sha256=preflight.external_metadata_sha256,
        preflight_producer_source_sha256=preflight.producer_source_sha256,
        standard19_mapping_policy_sha256=(
            preflight.standard19_mapping_policy_sha256
        ),
        source_roster_sha256=preflight.source_roster_sha256,
        count_semantics=HOLDING_COUNT_SEMANTICS,
        derived_from_manifest_sha256=None,
        eligible_group_ids=eligible,
        fit_group_ids=fit,
        held_group_ids=held,
        excluded_group_ids=excluded,
        holding_reference_target_count=holding_reference_target_count,
    )


def build_tuev_morphology_manifest(
    sources: Sequence[TUEVMorphologySourceRecord],
    preflight: VerifiedTUEVMorphologyPreflight,
    authorization: VerifiedTUEVMorphologyCohortAuthorization,
    *,
    preprocessing_policy_sha256: str,
    holding_reference_target_count: int | None = None,
) -> TUEVMorphologyManifest:
    """Formal holding builder with no caller-supplied ledger or group roster."""

    if not isinstance(
        authorization, VerifiedTUEVMorphologyCohortAuthorization
    ):
        raise TypeError(
            "authorization must be a derived TUEV morphology cohort capability"
        )
    source_roster_sha = _canonical_sha256(_source_roster_payload(sources))
    if source_roster_sha != authorization.source_roster_sha256:
        raise ValueError("Authorization belongs to another live TUEV source roster")
    duplicate_ledger = build_tuev_exact_signal_duplicate_ledger(sources)
    if duplicate_ledger.ledger_sha256 != authorization.duplicate_ledger_sha256:
        raise ValueError("Authorization belongs to another exact-EDF ledger")
    manifest = _build_tuev_morphology_manifest_from_roles(
        sources,
        preflight,
        global_ledger_sha256=authorization.public_ledger_sha256,
        preprocessing_policy_sha256=preprocessing_policy_sha256,
        cohort_authorization_sha256=authorization.receipt_sha256,
        excluded_group_ids=authorization.excluded_group_ids,
        holding_reference_target_count=holding_reference_target_count,
    )
    if manifest.eligible_group_ids != authorization.eligible_group_ids:
        raise RuntimeError("Holding manifest lost its authorized eligible roster")
    return manifest


def build_tuev_morphology_manifest_for_testing(
    sources: Sequence[TUEVMorphologySourceRecord],
    preflight: VerifiedTUEVMorphologyPreflight,
    *,
    global_ledger_sha256: str,
    preprocessing_policy_sha256: str,
    excluded_group_ids: Iterable[str] = (),
    holding_reference_target_count: int | None = None,
) -> TUEVMorphologyManifest:
    """Explicit self-attested helper for synthetic unit tests only."""

    excluded = tuple(sorted(set(str(value) for value in excluded_group_ids)))
    test_authorization_sha = _canonical_sha256(
        (
            "test_only_self_attested_tuev_holding",
            global_ledger_sha256,
            excluded,
        )
    )
    return _build_tuev_morphology_manifest_from_roles(
        sources,
        preflight,
        global_ledger_sha256=global_ledger_sha256,
        preprocessing_policy_sha256=preprocessing_policy_sha256,
        cohort_authorization_sha256=test_authorization_sha,
        excluded_group_ids=excluded,
        holding_reference_target_count=holding_reference_target_count,
    )


def _derive_tuev_morphology_fold_manifest_from_roles(
    master_manifest: TUEVMorphologyManifest,
    *,
    fit_group_ids: Iterable[str],
    held_group_ids: Iterable[str],
    excluded_group_ids: Iterable[str] = (),
) -> TUEVMorphologyManifest:
    """Deterministically project a verified holding manifest into one fold.

    The fold cannot introduce records, crops, labels, metadata, or provenance.
    Its only degrees of freedom are parent-group roles.  Final exclusions must
    include every exclusion already frozen in the holding manifest.
    """

    if (
        not isinstance(master_manifest, TUEVMorphologyManifest)
        or master_manifest.count_semantics != HOLDING_COUNT_SEMANTICS
    ):
        raise ValueError("Morphology folds require their exact holding manifest")
    all_groups = {record.parent_group_id for record in master_manifest.records}
    base_excluded = set(master_manifest.excluded_group_ids)
    excluded = tuple(sorted(set(str(value) for value in excluded_group_ids)))
    if not base_excluded <= set(excluded):
        raise ValueError("Fold exclusions cannot restore a holding-excluded group")
    if not set(excluded) <= all_groups:
        raise ValueError("Fold exclusions contain an unknown parent group")
    fit = tuple(sorted(set(str(value) for value in fit_group_ids)))
    held = tuple(sorted(set(str(value) for value in held_group_ids)))
    eligible = tuple(sorted(all_groups - set(excluded)))
    if set(fit) & set(held) or set(fit) | set(held) != set(eligible):
        raise ValueError("Fold fit/held rosters must partition eligible parent groups")
    newly_excluded = set(excluded) - base_excluded
    extra_omissions = tuple(
        _omission(record.record_id, "excluded_by_global_ledger")
        for record in master_manifest.records
        if record.parent_group_id in newly_excluded
    )
    omissions = tuple(
        sorted(
            (*master_manifest.omissions, *extra_omissions),
            key=lambda item: (
                item.record_id,
                -1 if item.source_line is None else item.source_line,
                item.reason_code,
            ),
        )
    )
    return TUEVMorphologyManifest(
        records=master_manifest.records,
        interval_groups=tuple(
            group
            for group in master_manifest.interval_groups
            if group.parent_group_id not in set(excluded)
        ),
        omissions=omissions,
        duplicate_ledger=master_manifest.duplicate_ledger,
        duplicate_ledger_sha256=master_manifest.duplicate_ledger_sha256,
        cohort_authorization_sha256=(
            master_manifest.cohort_authorization_sha256
        ),
        global_ledger_sha256=master_manifest.global_ledger_sha256,
        preprocessing_policy_sha256=master_manifest.preprocessing_policy_sha256,
        signal_preflight_receipt_sha256=(
            master_manifest.signal_preflight_receipt_sha256
        ),
        external_metadata_sha256=master_manifest.external_metadata_sha256,
        preflight_producer_source_sha256=(
            master_manifest.preflight_producer_source_sha256
        ),
        standard19_mapping_policy_sha256=(
            master_manifest.standard19_mapping_policy_sha256
        ),
        source_roster_sha256=master_manifest.source_roster_sha256,
        count_semantics=FOLD_COUNT_SEMANTICS,
        derived_from_manifest_sha256=master_manifest.manifest_sha256,
        eligible_group_ids=eligible,
        fit_group_ids=fit,
        held_group_ids=held,
        excluded_group_ids=excluded,
        holding_reference_target_count=None,
    )


def derive_tuev_morphology_fold_manifest(
    master_manifest: TUEVMorphologyManifest,
    authorization: VerifiedTUEVMorphologyCohortAuthorization,
) -> TUEVMorphologyManifest:
    """Project a holding manifest using only its replay-derived authority.

    Morphology is an independent auxiliary source, so this is one shared
    producer roster rather than five target-fold-specific rosters.  Every
    eligible official-train component is fit, every eligible official-eval
    component is natively held, and protected or cross-split components stay
    excluded.  Callers cannot supply or modify any of those roles.
    """

    if not isinstance(
        authorization, VerifiedTUEVMorphologyCohortAuthorization
    ):
        raise TypeError(
            "authorization must be a derived TUEV morphology cohort capability"
        )
    if (
        not isinstance(master_manifest, TUEVMorphologyManifest)
        or master_manifest.count_semantics != HOLDING_COUNT_SEMANTICS
    ):
        raise ValueError("Morphology folds require their exact holding manifest")
    expected_bindings = (
        master_manifest.cohort_authorization_sha256,
        master_manifest.global_ledger_sha256,
        master_manifest.source_roster_sha256,
        master_manifest.duplicate_ledger_sha256,
        master_manifest.eligible_group_ids,
        master_manifest.excluded_group_ids,
    )
    authorized_bindings = (
        authorization.receipt_sha256,
        authorization.public_ledger_sha256,
        authorization.source_roster_sha256,
        authorization.duplicate_ledger_sha256,
        authorization.eligible_group_ids,
        authorization.excluded_group_ids,
    )
    if expected_bindings != authorized_bindings:
        raise ValueError(
            "Holding manifest does not match the replay-derived cohort "
            "authorization"
        )
    manifest = _derive_tuev_morphology_fold_manifest_from_roles(
        master_manifest,
        fit_group_ids=authorization.fit_group_ids,
        held_group_ids=authorization.held_group_ids,
        excluded_group_ids=authorization.excluded_group_ids,
    )
    if (
        manifest.fit_group_ids != authorization.fit_group_ids
        or manifest.held_group_ids != authorization.held_group_ids
        or manifest.excluded_group_ids != authorization.excluded_group_ids
        or manifest.eligible_group_ids != authorization.eligible_group_ids
    ):
        raise RuntimeError("Authorized morphology roles changed during projection")
    return manifest


def derive_tuev_morphology_fold_manifest_for_testing(
    master_manifest: TUEVMorphologyManifest,
    *,
    fit_group_ids: Iterable[str],
    held_group_ids: Iterable[str],
    excluded_group_ids: Iterable[str] = (),
) -> TUEVMorphologyManifest:
    """Explicit self-attested fold projection for synthetic unit tests only."""

    return _derive_tuev_morphology_fold_manifest_from_roles(
        master_manifest,
        fit_group_ids=fit_group_ids,
        held_group_ids=held_group_ids,
        excluded_group_ids=excluded_group_ids,
    )


_EXTERNAL_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "producer_source_sha256",
        "preprocessing_policy_sha256",
        "standard19_mapping_policy_sha256",
        "duplicate_ledger",
        "duplicate_ledger_sha256",
        "records",
    }
)
_PREFLIGHT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy_version",
        "producer_source_sha256",
        "preprocessing_policy_sha256",
        "standard19_mapping_policy_sha256",
        "external_metadata_sha256",
        "source_roster_sha256",
        "metadata_roster_sha256",
        "duplicate_ledger",
        "duplicate_ledger_sha256",
        "records",
    }
)
_PREFLIGHT_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "receipt_file",
        "receipt_sha256",
        "receipt_size_bytes",
        "external_metadata_sha256",
        "source_roster_sha256",
        "duplicate_ledger_sha256",
        "record_count",
    }
)
_METADATA_FIELDS = frozenset(
    {
        "relative_edf_path",
        "edf_sha256",
        "source_sfreq_hz",
        "source_sample_count",
        "output_sample_count",
        "direct_standard19",
        "standard19_mapping_sha256",
        "preprocessing_receipt_sha256",
        "signal_qc_passed",
        "signal_qc_receipt_sha256",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "relative_edf_path",
        "relative_rec_path",
        "official_split",
        "parent_group_id",
        "group_kind",
        "source_subject_id",
        "edf_sha256",
        "rec_sha256",
        "derivative_files",
        "parent_group_files",
        "group_file_roster_sha256",
        "metadata",
    }
)
_TARGET_FIELDS = frozenset(
    {
        "target_id",
        "record_id",
        "edge_index",
        "label_index",
        "label_name",
        "start_sample",
        "stop_sample",
        "source_line",
        "overlap_component_id",
        "overlap_component_size",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "crop_id",
        "record_id",
        "parent_group_id",
        "edf_sha256",
        "start_sample",
        "stop_sample",
        "targets",
        "source_target_mask_sha256",
    }
)
_OMISSION_FIELDS = frozenset(
    {"record_id", "reason_code", "source_line", "edge_index", "label_index"}
)
_DUPLICATE_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "policy_version",
        "identity_basis",
        "source_roster_sha256",
        "duplicate_classes",
        "group_components",
        "record_decisions",
    }
)
_DUPLICATE_CLASS_FIELDS = frozenset(
    {
        "edf_sha256",
        "record_descriptors",
        "group_ids",
        "official_splits",
        "annotation_status",
        "canonical_record_id",
    }
)
_CONTENT_COMPONENT_FIELDS = frozenset(
    {
        "component_id",
        "group_ids",
        "official_splits",
        "connecting_edf_sha256s",
        "policy_action",
    }
)
_RECORD_DECISION_FIELDS = frozenset(
    {
        "record_id",
        "group_id",
        "official_split",
        "edf_sha256",
        "rec_sha256",
        "component_id",
        "action",
        "canonical_record_id",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy_version",
        "records",
        "interval_groups",
        "omissions",
        "duplicate_ledger",
        "duplicate_ledger_sha256",
        "cohort_authorization_sha256",
        "global_ledger_sha256",
        "preprocessing_policy_sha256",
        "signal_preflight_receipt_sha256",
        "external_metadata_sha256",
        "preflight_producer_source_sha256",
        "standard19_mapping_policy_sha256",
        "source_roster_sha256",
        "count_semantics",
        "derived_from_manifest_sha256",
        "eligible_group_ids",
        "fit_group_ids",
        "held_group_ids",
        "excluded_group_ids",
        "holding_reference_target_count",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "receipt_file",
        "receipt_sha256",
        "receipt_size_bytes",
        "source_manifest_sha256",
        "record_count",
        "interval_group_count",
        "target_count",
        "omission_count",
    }
)


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    return value


def _string_array(value: object, *, field: str) -> tuple[str, ...]:
    items = _array(value, field=field)
    if any(not isinstance(item, str) for item in items):
        raise TypeError(f"{field} must contain strings")
    return tuple(items)


def _metadata_from_payload(value: object) -> TUEVMorphologyRecordMetadata:
    payload = _object(value, field="metadata")
    _require_exact_fields(payload, _METADATA_FIELDS, label="metadata")
    sfreq = payload["source_sfreq_hz"]
    if isinstance(sfreq, bool) or not isinstance(sfreq, (int, float)):
        raise TypeError("metadata.source_sfreq_hz must be numeric")
    return TUEVMorphologyRecordMetadata(
        relative_edf_path=_relative_path(payload["relative_edf_path"], field="metadata.relative_edf_path"),
        edf_sha256=_sha(payload["edf_sha256"], field="metadata.edf_sha256"),
        source_sfreq_hz=float(sfreq),
        source_sample_count=_integer(payload["source_sample_count"], field="metadata.source_sample_count", minimum=1),
        output_sample_count=_integer(payload["output_sample_count"], field="metadata.output_sample_count", minimum=1),
        direct_standard19=payload["direct_standard19"],
        standard19_mapping_sha256=_sha(payload["standard19_mapping_sha256"], field="metadata.standard19_mapping_sha256"),
        preprocessing_receipt_sha256=_sha(payload["preprocessing_receipt_sha256"], field="metadata.preprocessing_receipt_sha256"),
        signal_qc_passed=payload["signal_qc_passed"],
        signal_qc_receipt_sha256=_sha(payload["signal_qc_receipt_sha256"], field="metadata.signal_qc_receipt_sha256"),
    )


def _duplicate_ledger_from_payload(
    value: object,
) -> TUEVExactSignalDuplicateLedger:
    payload = _object(value, field="duplicate_ledger")
    _require_exact_fields(
        payload, _DUPLICATE_LEDGER_FIELDS, label="duplicate_ledger"
    )
    duplicate_classes: list[TUEVExactSignalDuplicateClass] = []
    for index, raw_item in enumerate(
        _array(payload["duplicate_classes"], field="duplicate_classes")
    ):
        item = _object(raw_item, field=f"duplicate_classes[{index}]")
        _require_exact_fields(
            item,
            _DUPLICATE_CLASS_FIELDS,
            label=f"duplicate_classes[{index}]",
        )
        descriptors: list[tuple[str, str, str, str]] = []
        for row_index, raw_row in enumerate(
            _array(
                item["record_descriptors"],
                field=f"duplicate_classes[{index}].record_descriptors",
            )
        ):
            if not isinstance(raw_row, list) or len(raw_row) != 4 or any(
                not isinstance(value, str) for value in raw_row
            ):
                raise TypeError(
                    "Duplicate-class record descriptors must be four-string rows"
                )
            descriptors.append(tuple(raw_row))
        canonical = item["canonical_record_id"]
        if canonical is not None and not isinstance(canonical, str):
            raise TypeError("Duplicate-class canonical_record_id must be string/null")
        duplicate_classes.append(
            TUEVExactSignalDuplicateClass(
                edf_sha256=_sha(
                    item["edf_sha256"],
                    field=f"duplicate_classes[{index}].edf_sha256",
                ),
                record_descriptors=tuple(descriptors),
                group_ids=_string_array(
                    item["group_ids"],
                    field=f"duplicate_classes[{index}].group_ids",
                ),
                official_splits=_string_array(
                    item["official_splits"],
                    field=f"duplicate_classes[{index}].official_splits",
                ),
                annotation_status=_text(
                    item["annotation_status"],
                    field=f"duplicate_classes[{index}].annotation_status",
                ),
                canonical_record_id=canonical,
            )
        )
    components: list[TUEVContentGroupComponent] = []
    for index, raw_item in enumerate(
        _array(payload["group_components"], field="group_components")
    ):
        item = _object(raw_item, field=f"group_components[{index}]")
        _require_exact_fields(
            item,
            _CONTENT_COMPONENT_FIELDS,
            label=f"group_components[{index}]",
        )
        components.append(
            TUEVContentGroupComponent(
                component_id=_text(
                    item["component_id"],
                    field=f"group_components[{index}].component_id",
                ),
                group_ids=_string_array(
                    item["group_ids"],
                    field=f"group_components[{index}].group_ids",
                ),
                official_splits=_string_array(
                    item["official_splits"],
                    field=f"group_components[{index}].official_splits",
                ),
                connecting_edf_sha256s=_string_array(
                    item["connecting_edf_sha256s"],
                    field=(
                        f"group_components[{index}].connecting_edf_sha256s"
                    ),
                ),
                policy_action=_text(
                    item["policy_action"],
                    field=f"group_components[{index}].policy_action",
                ),
            )
        )
    decisions: list[TUEVDuplicateRecordDecision] = []
    for index, raw_item in enumerate(
        _array(payload["record_decisions"], field="record_decisions")
    ):
        item = _object(raw_item, field=f"record_decisions[{index}]")
        _require_exact_fields(
            item,
            _RECORD_DECISION_FIELDS,
            label=f"record_decisions[{index}]",
        )
        canonical = item["canonical_record_id"]
        if canonical is not None and not isinstance(canonical, str):
            raise TypeError("Record-decision canonical_record_id must be string/null")
        decisions.append(
            TUEVDuplicateRecordDecision(
                record_id=_text(
                    item["record_id"],
                    field=f"record_decisions[{index}].record_id",
                ),
                group_id=_text(
                    item["group_id"],
                    field=f"record_decisions[{index}].group_id",
                ),
                official_split=_text(
                    item["official_split"],
                    field=f"record_decisions[{index}].official_split",
                ),
                edf_sha256=_sha(
                    item["edf_sha256"],
                    field=f"record_decisions[{index}].edf_sha256",
                ),
                rec_sha256=_sha(
                    item["rec_sha256"],
                    field=f"record_decisions[{index}].rec_sha256",
                ),
                component_id=_text(
                    item["component_id"],
                    field=f"record_decisions[{index}].component_id",
                ),
                action=_text(
                    item["action"],
                    field=f"record_decisions[{index}].action",
                ),
                canonical_record_id=canonical,
            )
        )
    return TUEVExactSignalDuplicateLedger(
        source_roster_sha256=_sha(
            payload["source_roster_sha256"],
            field="duplicate_ledger.source_roster_sha256",
        ),
        duplicate_classes=tuple(duplicate_classes),
        group_components=tuple(components),
        record_decisions=tuple(decisions),
        identity_basis=_text(
            payload["identity_basis"],
            field="duplicate_ledger.identity_basis",
        ),
        schema_version=_text(
            payload["schema_version"], field="duplicate_ledger.schema_version"
        ),
        policy_version=_text(
            payload["policy_version"], field="duplicate_ledger.policy_version"
        ),
    )


def _external_metadata_records(
    payload: Mapping[str, object],
    *,
    expected_producer_source_sha256: str,
    expected_preprocessing_policy_sha256: str,
    expected_standard19_mapping_policy_sha256: str,
) -> tuple[
    tuple[TUEVMorphologyRecordMetadata, ...],
    TUEVExactSignalDuplicateLedger,
]:
    _require_exact_fields(
        payload, _EXTERNAL_METADATA_FIELDS, label="external metadata"
    )
    if payload["schema_version"] != TUEV_MORPHOLOGY_EXTERNAL_METADATA_SCHEMA:
        raise ValueError("Unsupported TUEV morphology external-metadata schema")
    expected_bindings = {
        "producer_source_sha256": _sha(
            expected_producer_source_sha256,
            field="expected_producer_source_sha256",
        ),
        "preprocessing_policy_sha256": _sha(
            expected_preprocessing_policy_sha256,
            field="expected_preprocessing_policy_sha256",
        ),
        "standard19_mapping_policy_sha256": _sha(
            expected_standard19_mapping_policy_sha256,
            field="expected_standard19_mapping_policy_sha256",
        ),
    }
    for field, expected in expected_bindings.items():
        if payload[field] != expected:
            raise ValueError(f"External metadata binding mismatch: {field}")
    records = tuple(
        _metadata_from_payload(item)
        for item in _array(payload["records"], field="external metadata.records")
    )
    paths = tuple(record.relative_edf_path for record in records)
    if not records or paths != tuple(sorted(set(paths))):
        raise ValueError(
            "External metadata records must be non-empty, unique, and path-sorted"
        )
    duplicate_ledger = _duplicate_ledger_from_payload(
        payload["duplicate_ledger"]
    )
    if payload["duplicate_ledger_sha256"] != duplicate_ledger.ledger_sha256:
        raise ValueError("External metadata duplicate-ledger SHA mismatch")
    return records, duplicate_ledger


def _replay_current_first_party_external_metadata(
    *,
    edf_root: str | Path,
    payload: Mapping[str, object],
    producer_source_sha256: str,
    preprocessing_policy_sha256: str,
    standard19_mapping_policy_sha256: str,
) -> None:
    """Require current bindings and independently replay every real source."""

    from .tuev_morphology_signal_preflight import (
        replay_tuev_morphology_first_party_metadata,
        require_first_party_tuev_morphology_bindings,
    )

    require_first_party_tuev_morphology_bindings(
        producer_source_sha256=producer_source_sha256,
        preprocessing_policy_sha256=preprocessing_policy_sha256,
        standard19_mapping_policy_sha256=standard19_mapping_policy_sha256,
    )
    replay_tuev_morphology_first_party_metadata(edf_root, payload)


def _validate_preflight_records_against_sources(
    records: Sequence[TUEVMorphologyRecordMetadata],
    sources: Sequence[TUEVMorphologySourceRecord],
) -> None:
    by_path = {source.relative_edf_path: source for source in sources}
    record_paths = {record.relative_edf_path for record in records}
    if record_paths != set(by_path):
        raise ValueError(
            "External preflight metadata must cover exactly the discovered EDF roster"
        )
    for record in records:
        source = by_path[record.relative_edf_path]
        if record.edf_sha256 != source.edf_sha256:
            raise ValueError(
                f"External preflight EDF SHA mismatch: {record.relative_edf_path}"
            )


def materialize_tuev_morphology_preflight(
    path: str | Path,
    *,
    edf_root: str | Path,
    external_metadata_path: str | Path,
    expected_external_metadata_sha256: str,
    expected_producer_source_sha256: str,
    expected_preprocessing_policy_sha256: str,
    expected_standard19_mapping_policy_sha256: str,
) -> TUEVMorphologyPreflightArtifact:
    """Publish a safe gate around canonical header/QC metadata.

    Current first-party bindings are mandatory and the claimed rows are
    regenerated from the real signals before publication.  There is no legacy
    or caller-supplied-summary path that can issue a verified token.
    """

    expected_external_sha = _sha(
        expected_external_metadata_sha256,
        field="expected_external_metadata_sha256",
    )
    _, external_raw, external_sha = _read_stable_regular_file(
        external_metadata_path,
        label="external morphology metadata",
        max_bytes=_MAX_EXTERNAL_METADATA_BYTES,
    )
    if external_sha != expected_external_sha:
        raise ValueError("External morphology metadata SHA-256 mismatch")
    external_payload = _parse_canonical_json(
        external_raw, label="external morphology metadata"
    )
    records, external_duplicate_ledger = _external_metadata_records(
        external_payload,
        expected_producer_source_sha256=expected_producer_source_sha256,
        expected_preprocessing_policy_sha256=(
            expected_preprocessing_policy_sha256
        ),
        expected_standard19_mapping_policy_sha256=(
            expected_standard19_mapping_policy_sha256
        ),
    )
    _replay_current_first_party_external_metadata(
        edf_root=edf_root,
        payload=external_payload,
        producer_source_sha256=expected_producer_source_sha256,
        preprocessing_policy_sha256=expected_preprocessing_policy_sha256,
        standard19_mapping_policy_sha256=(
            expected_standard19_mapping_policy_sha256
        ),
    )
    sources = discover_tuev_morphology_sources(edf_root)
    _validate_preflight_records_against_sources(records, sources)
    source_roster_sha = _canonical_sha256(_source_roster_payload(sources))
    duplicate_ledger = build_tuev_exact_signal_duplicate_ledger(sources)
    if duplicate_ledger != external_duplicate_ledger:
        raise ValueError(
            "External metadata exact-EDF-byte ledger differs from source replay"
        )
    metadata_roster_sha = _canonical_sha256(
        tuple(record.canonical_payload for record in records)
    )
    receipt = {
        "schema_version": TUEV_MORPHOLOGY_PREFLIGHT_SCHEMA,
        "policy_version": TUEV_MORPHOLOGY_PREFLIGHT_POLICY,
        "producer_source_sha256": _sha(
            expected_producer_source_sha256,
            field="expected_producer_source_sha256",
        ),
        "preprocessing_policy_sha256": _sha(
            expected_preprocessing_policy_sha256,
            field="expected_preprocessing_policy_sha256",
        ),
        "standard19_mapping_policy_sha256": _sha(
            expected_standard19_mapping_policy_sha256,
            field="expected_standard19_mapping_policy_sha256",
        ),
        "external_metadata_sha256": external_sha,
        "source_roster_sha256": source_roster_sha,
        "metadata_roster_sha256": metadata_roster_sha,
        "duplicate_ledger": duplicate_ledger.canonical_payload,
        "duplicate_ledger_sha256": duplicate_ledger.ledger_sha256,
        "records": [record.canonical_payload for record in records],
    }
    _require_exact_fields(
        receipt, _PREFLIGHT_RECEIPT_FIELDS, label="preflight receipt"
    )
    receipt_bytes = _canonical_json(receipt)
    if not 1 <= len(receipt_bytes) <= _MAX_RECEIPT_BYTES:
        raise ValueError("TUEV morphology preflight receipt size is invalid")
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    bundle = {
        "schema_version": TUEV_MORPHOLOGY_PREFLIGHT_BUNDLE_SCHEMA,
        "serialization": "canonical_json_no_pickle_external_bytes_required",
        "receipt_file": _PREFLIGHT_RECEIPT_FILE,
        "receipt_sha256": receipt_sha,
        "receipt_size_bytes": len(receipt_bytes),
        "external_metadata_sha256": external_sha,
        "source_roster_sha256": source_roster_sha,
        "duplicate_ledger_sha256": duplicate_ledger.ledger_sha256,
        "record_count": len(records),
    }
    _require_exact_fields(
        bundle, _PREFLIGHT_BUNDLE_FIELDS, label="preflight manifest"
    )
    bundle_bytes = _canonical_json(bundle)
    if not 1 <= len(bundle_bytes) <= _MAX_MANIFEST_BYTES:
        raise ValueError("TUEV morphology preflight manifest size is invalid")

    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Morphology preflight already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        (temporary / _PREFLIGHT_RECEIPT_FILE).write_bytes(receipt_bytes)
        (temporary / _BUNDLE_MANIFEST_FILE).write_bytes(bundle_bytes)
        _fsync_file(temporary / _PREFLIGHT_RECEIPT_FILE)
        _fsync_file(temporary / _BUNDLE_MANIFEST_FILE)
        _fsync_directory(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Morphology preflight already exists: {target}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return TUEVMorphologyPreflightArtifact(
        path=target,
        bundle_manifest_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        preflight_receipt_sha256=receipt_sha,
        external_metadata_sha256=external_sha,
        source_roster_sha256=source_roster_sha,
        duplicate_ledger_sha256=duplicate_ledger.ledger_sha256,
        record_count=len(records),
    )


def load_tuev_morphology_preflight(
    path: str | Path,
    *,
    edf_root: str | Path,
    external_metadata_path: str | Path,
    expected_bundle_manifest_sha256: str,
    expected_preflight_receipt_sha256: str,
    expected_external_metadata_sha256: str,
    expected_producer_source_sha256: str,
    expected_preprocessing_policy_sha256: str,
    expected_standard19_mapping_policy_sha256: str,
    replay_live_source: bool = True,
) -> VerifiedTUEVMorphologyPreflight:
    """Strictly replay a preflight bundle, its external bytes, and TUEV tree."""

    if not isinstance(replay_live_source, bool):
        raise TypeError("replay_live_source must be bool")
    expected_bundle_sha = _sha(
        expected_bundle_manifest_sha256,
        field="expected_bundle_manifest_sha256",
    )
    expected_receipt_sha = _sha(
        expected_preflight_receipt_sha256,
        field="expected_preflight_receipt_sha256",
    )
    expected_external_sha = _sha(
        expected_external_metadata_sha256,
        field="expected_external_metadata_sha256",
    )
    producer_sha = _sha(
        expected_producer_source_sha256,
        field="expected_producer_source_sha256",
    )
    preprocessing_sha = _sha(
        expected_preprocessing_policy_sha256,
        field="expected_preprocessing_policy_sha256",
    )
    mapping_sha = _sha(
        expected_standard19_mapping_policy_sha256,
        field="expected_standard19_mapping_policy_sha256",
    )
    _, external_raw, external_sha = _read_stable_regular_file(
        external_metadata_path,
        label="external morphology metadata",
        max_bytes=_MAX_EXTERNAL_METADATA_BYTES,
    )
    if external_sha != expected_external_sha:
        raise ValueError("External morphology metadata SHA-256 mismatch")
    external_payload = _parse_canonical_json(
        external_raw, label="external morphology metadata"
    )
    external_records, external_duplicate_ledger = _external_metadata_records(
        external_payload,
        expected_producer_source_sha256=producer_sha,
        expected_preprocessing_policy_sha256=preprocessing_sha,
        expected_standard19_mapping_policy_sha256=mapping_sha,
    )
    if replay_live_source:
        _replay_current_first_party_external_metadata(
            edf_root=edf_root,
            payload=external_payload,
            producer_source_sha256=producer_sha,
            preprocessing_policy_sha256=preprocessing_sha,
            standard19_mapping_policy_sha256=mapping_sha,
        )

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("Morphology preflight bundle must be a canonical directory")
    expected_files = {_BUNDLE_MANIFEST_FILE, _PREFLIGHT_RECEIPT_FILE}
    if {item.name for item in source.iterdir()} != expected_files:
        raise ValueError("Morphology preflight bundle contains missing or unknown files")
    manifest_path = source / _BUNDLE_MANIFEST_FILE
    receipt_path = source / _PREFLIGHT_RECEIPT_FILE
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, receipt_path)):
        raise ValueError("Morphology preflight members must be regular files")
    manifest_bytes = manifest_path.read_bytes()
    if not 1 <= len(manifest_bytes) <= _MAX_MANIFEST_BYTES:
        raise ValueError("Morphology preflight manifest size is invalid")
    bundle_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if bundle_sha != expected_bundle_sha:
        raise ValueError("Morphology preflight bundle SHA-256 mismatch")
    bundle = _parse_canonical_json(manifest_bytes, label="preflight manifest")
    _require_exact_fields(
        bundle, _PREFLIGHT_BUNDLE_FIELDS, label="preflight manifest"
    )
    if (
        bundle["schema_version"] != TUEV_MORPHOLOGY_PREFLIGHT_BUNDLE_SCHEMA
        or bundle["serialization"]
        != "canonical_json_no_pickle_external_bytes_required"
        or bundle["receipt_file"] != _PREFLIGHT_RECEIPT_FILE
    ):
        raise ValueError("Morphology preflight bundle schema/serialization drifted")
    receipt_size = _integer(
        bundle["receipt_size_bytes"], field="receipt_size_bytes", minimum=1
    )
    if receipt_size > _MAX_RECEIPT_BYTES or receipt_path.stat().st_size != receipt_size:
        raise ValueError("Morphology preflight receipt size mismatch")
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha != expected_receipt_sha or receipt_sha != _sha(
        bundle["receipt_sha256"], field="receipt_sha256"
    ):
        raise ValueError("Morphology preflight receipt SHA-256 mismatch")
    receipt = _parse_canonical_json(receipt_bytes, label="preflight receipt")
    _require_exact_fields(
        receipt, _PREFLIGHT_RECEIPT_FIELDS, label="preflight receipt"
    )
    if (
        receipt["schema_version"] != TUEV_MORPHOLOGY_PREFLIGHT_SCHEMA
        or receipt["policy_version"] != TUEV_MORPHOLOGY_PREFLIGHT_POLICY
    ):
        raise ValueError("Morphology preflight receipt schema/policy drifted")
    expected_bindings = {
        "producer_source_sha256": producer_sha,
        "preprocessing_policy_sha256": preprocessing_sha,
        "standard19_mapping_policy_sha256": mapping_sha,
        "external_metadata_sha256": external_sha,
    }
    for field, expected in expected_bindings.items():
        if receipt[field] != expected:
            raise ValueError(f"Morphology preflight receipt binding mismatch: {field}")
    receipt_records = tuple(
        _metadata_from_payload(item)
        for item in _array(receipt["records"], field="preflight receipt.records")
    )
    if receipt_records != external_records:
        raise ValueError("Preflight receipt records differ from external metadata bytes")
    receipt_duplicate_ledger = _duplicate_ledger_from_payload(
        receipt["duplicate_ledger"]
    )
    if (
        receipt_duplicate_ledger != external_duplicate_ledger
        or receipt["duplicate_ledger_sha256"]
        != receipt_duplicate_ledger.ledger_sha256
    ):
        raise ValueError(
            "Preflight receipt duplicate ledger differs from external metadata"
        )
    metadata_roster_sha = _canonical_sha256(
        tuple(record.canonical_payload for record in receipt_records)
    )
    if metadata_roster_sha != _sha(
        receipt["metadata_roster_sha256"], field="metadata_roster_sha256"
    ):
        raise ValueError("Morphology preflight metadata-roster SHA mismatch")

    sources = discover_tuev_morphology_sources(edf_root)
    _validate_preflight_records_against_sources(receipt_records, sources)
    source_roster_sha = _canonical_sha256(_source_roster_payload(sources))
    replayed_duplicate_ledger = build_tuev_exact_signal_duplicate_ledger(sources)
    if replayed_duplicate_ledger != receipt_duplicate_ledger:
        raise ValueError("Morphology preflight duplicate ledger changed")
    if source_roster_sha != _sha(
        receipt["source_roster_sha256"], field="source_roster_sha256"
    ):
        raise ValueError("Morphology preflight source roster changed")
    if (
        bundle["external_metadata_sha256"] != external_sha
        or bundle["source_roster_sha256"] != source_roster_sha
        or bundle["duplicate_ledger_sha256"]
        != replayed_duplicate_ledger.ledger_sha256
        or _integer(bundle["record_count"], field="record_count", minimum=1)
        != len(receipt_records)
    ):
        raise ValueError("Morphology preflight outer bundle disagrees with its receipt")
    return VerifiedTUEVMorphologyPreflight(
        _marker=_VERIFIED_TUEV_MORPHOLOGY_PREFLIGHT_MARKER,
        path=source,
        bundle_manifest_sha256=bundle_sha,
        preflight_receipt_sha256=receipt_sha,
        external_metadata_sha256=external_sha,
        source_roster_sha256=source_roster_sha,
        producer_source_sha256=producer_sha,
        preprocessing_policy_sha256=preprocessing_sha,
        standard19_mapping_policy_sha256=mapping_sha,
        duplicate_ledger_sha256=replayed_duplicate_ledger.ledger_sha256,
        duplicate_ledger=replayed_duplicate_ledger,
        records=receipt_records,
    )


def _record_from_payload(value: object) -> TUEVMorphologyRecordReceipt:
    payload = _object(value, field="record")
    _require_exact_fields(payload, _RECORD_FIELDS, label="record")
    def parse_roster(field: str) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = []
        for index, item in enumerate(_array(payload[field], field=f"record.{field}")):
            if not isinstance(item, list) or len(item) != 2:
                raise TypeError(f"record.{field}[{index}] must be [path,sha]")
            rows.append(
                (
                    _relative_path(item[0], field=f"record.{field}[{index}].path"),
                    _sha(item[1], field=f"record.{field}[{index}].sha256"),
                )
            )
        return tuple(rows)
    derivatives = parse_roster("derivative_files")
    parent_group_files = parse_roster("parent_group_files")
    subject = payload["source_subject_id"]
    if subject is not None and not isinstance(subject, str):
        raise TypeError("record.source_subject_id must be string or null")
    return TUEVMorphologyRecordReceipt(
        record_id=_text(payload["record_id"], field="record.record_id"),
        relative_edf_path=_relative_path(payload["relative_edf_path"], field="record.relative_edf_path"),
        relative_rec_path=_relative_path(payload["relative_rec_path"], field="record.relative_rec_path"),
        official_split=_text(payload["official_split"], field="record.official_split"),
        parent_group_id=_text(payload["parent_group_id"], field="record.parent_group_id"),
        group_kind=_text(payload["group_kind"], field="record.group_kind"),
        source_subject_id=subject,
        edf_sha256=_sha(payload["edf_sha256"], field="record.edf_sha256"),
        rec_sha256=_sha(payload["rec_sha256"], field="record.rec_sha256"),
        derivative_files=derivatives,
        parent_group_files=parent_group_files,
        group_file_roster_sha256=_sha(payload["group_file_roster_sha256"], field="record.group_file_roster_sha256"),
        metadata=_metadata_from_payload(payload["metadata"]),
    )


def _target_from_payload(value: object) -> TUEVMorphologyTarget:
    payload = _object(value, field="target")
    _require_exact_fields(payload, _TARGET_FIELDS, label="target")
    return TUEVMorphologyTarget(
        target_id=_text(payload["target_id"], field="target.target_id"),
        record_id=_text(payload["record_id"], field="target.record_id"),
        edge_index=_integer(payload["edge_index"], field="target.edge_index"),
        label_index=_integer(payload["label_index"], field="target.label_index"),
        label_name=_text(payload["label_name"], field="target.label_name"),
        start_sample=_integer(payload["start_sample"], field="target.start_sample"),
        stop_sample=_integer(payload["stop_sample"], field="target.stop_sample"),
        source_line=_integer(payload["source_line"], field="target.source_line", minimum=1),
        overlap_component_id=_text(payload["overlap_component_id"], field="target.overlap_component_id"),
        overlap_component_size=_integer(payload["overlap_component_size"], field="target.overlap_component_size", minimum=1),
    )


def _group_from_payload(value: object) -> TUEVMorphologyIntervalGroup:
    payload = _object(value, field="interval_group")
    _require_exact_fields(payload, _GROUP_FIELDS, label="interval_group")
    return TUEVMorphologyIntervalGroup(
        crop_id=_text(payload["crop_id"], field="interval_group.crop_id"),
        record_id=_text(payload["record_id"], field="interval_group.record_id"),
        parent_group_id=_text(payload["parent_group_id"], field="interval_group.parent_group_id"),
        edf_sha256=_sha(payload["edf_sha256"], field="interval_group.edf_sha256"),
        start_sample=_integer(payload["start_sample"], field="interval_group.start_sample"),
        stop_sample=_integer(payload["stop_sample"], field="interval_group.stop_sample"),
        targets=tuple(_target_from_payload(item) for item in _array(payload["targets"], field="interval_group.targets")),
        source_target_mask_sha256=_sha(payload["source_target_mask_sha256"], field="interval_group.source_target_mask_sha256"),
    )


def _omission_from_payload(value: object) -> TUEVMorphologyOmission:
    payload = _object(value, field="omission")
    _require_exact_fields(payload, _OMISSION_FIELDS, label="omission")
    for field in ("source_line", "edge_index", "label_index"):
        if payload[field] is not None and (isinstance(payload[field], bool) or not isinstance(payload[field], int)):
            raise TypeError(f"omission.{field} must be integer or null")
    return TUEVMorphologyOmission(
        record_id=_text(payload["record_id"], field="omission.record_id"),
        reason_code=_text(payload["reason_code"], field="omission.reason_code"),
        source_line=payload["source_line"],
        edge_index=payload["edge_index"],
        label_index=payload["label_index"],
    )


def _manifest_from_payload(value: Mapping[str, object]) -> TUEVMorphologyManifest:
    _require_exact_fields(value, _RECEIPT_FIELDS, label="receipt.json")
    derived = value["derived_from_manifest_sha256"]
    if derived is not None:
        derived = _sha(derived, field="derived_from_manifest_sha256")
    reference = value["holding_reference_target_count"]
    if reference is not None:
        reference = _integer(reference, field="holding_reference_target_count", minimum=1)
    duplicate_ledger = _duplicate_ledger_from_payload(
        value["duplicate_ledger"]
    )
    return TUEVMorphologyManifest(
        records=tuple(_record_from_payload(item) for item in _array(value["records"], field="records")),
        interval_groups=tuple(_group_from_payload(item) for item in _array(value["interval_groups"], field="interval_groups")),
        omissions=tuple(_omission_from_payload(item) for item in _array(value["omissions"], field="omissions")),
        duplicate_ledger=duplicate_ledger,
        duplicate_ledger_sha256=_sha(
            value["duplicate_ledger_sha256"],
            field="duplicate_ledger_sha256",
        ),
        cohort_authorization_sha256=_sha(
            value["cohort_authorization_sha256"],
            field="cohort_authorization_sha256",
        ),
        global_ledger_sha256=_sha(value["global_ledger_sha256"], field="global_ledger_sha256"),
        preprocessing_policy_sha256=_sha(value["preprocessing_policy_sha256"], field="preprocessing_policy_sha256"),
        signal_preflight_receipt_sha256=_sha(
            value["signal_preflight_receipt_sha256"],
            field="signal_preflight_receipt_sha256",
        ),
        external_metadata_sha256=_sha(
            value["external_metadata_sha256"], field="external_metadata_sha256"
        ),
        preflight_producer_source_sha256=_sha(
            value["preflight_producer_source_sha256"],
            field="preflight_producer_source_sha256",
        ),
        standard19_mapping_policy_sha256=_sha(
            value["standard19_mapping_policy_sha256"],
            field="standard19_mapping_policy_sha256",
        ),
        source_roster_sha256=_sha(
            value["source_roster_sha256"], field="source_roster_sha256"
        ),
        count_semantics=_text(value["count_semantics"], field="count_semantics"),
        derived_from_manifest_sha256=derived,
        eligible_group_ids=_string_array(value["eligible_group_ids"], field="eligible_group_ids"),
        fit_group_ids=_string_array(value["fit_group_ids"], field="fit_group_ids"),
        held_group_ids=_string_array(value["held_group_ids"], field="held_group_ids"),
        excluded_group_ids=_string_array(value["excluded_group_ids"], field="excluded_group_ids"),
        holding_reference_target_count=reference,
        schema_version=_text(value["schema_version"], field="schema_version"),
        policy_version=_text(value["policy_version"], field="policy_version"),
    )


@dataclass(frozen=True)
class TUEVMorphologyManifestArtifact:
    path: Path
    bundle_manifest_sha256: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Manifest artifact path must be absolute")
        _sha(self.bundle_manifest_sha256, field="bundle_manifest_sha256")
        _sha(self.source_manifest_sha256, field="source_manifest_sha256")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_tuev_morphology_manifest(
    path: str | Path, manifest: TUEVMorphologyManifest
) -> TUEVMorphologyManifestArtifact:
    if not isinstance(manifest, TUEVMorphologyManifest):
        raise TypeError("manifest must be TUEVMorphologyManifest")
    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"TUEV morphology manifest already exists: {target}")
    receipt_bytes = _canonical_json(manifest.canonical_payload)
    if not 1 <= len(receipt_bytes) <= _MAX_RECEIPT_BYTES:
        raise ValueError("Morphology receipt size is invalid")
    reconstructed = _manifest_from_payload(
        _parse_canonical_json(receipt_bytes, label="receipt.json")
    )
    if reconstructed != manifest:
        raise ValueError("Morphology manifest is not stable under safe reconstruction")
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha != manifest.manifest_sha256:
        raise RuntimeError("Canonical receipt SHA disagrees with manifest_sha256")
    bundle = {
        "schema_version": TUEV_MORPHOLOGY_BUNDLE_SCHEMA,
        "serialization": "canonical_json_no_pickle",
        "receipt_file": _BUNDLE_RECEIPT_FILE,
        "receipt_sha256": receipt_sha,
        "receipt_size_bytes": len(receipt_bytes),
        "source_manifest_sha256": manifest.manifest_sha256,
        "record_count": len(manifest.records),
        "interval_group_count": len(manifest.interval_groups),
        "target_count": manifest.target_count,
        "omission_count": len(manifest.omissions),
    }
    _require_exact_fields(bundle, _BUNDLE_FIELDS, label="manifest.json")
    bundle_bytes = _canonical_json(bundle)
    if len(bundle_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("Morphology bundle manifest is unexpectedly large")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        (temporary / _BUNDLE_RECEIPT_FILE).write_bytes(receipt_bytes)
        (temporary / _BUNDLE_MANIFEST_FILE).write_bytes(bundle_bytes)
        _fsync_file(temporary / _BUNDLE_RECEIPT_FILE)
        _fsync_file(temporary / _BUNDLE_MANIFEST_FILE)
        _fsync_directory(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"TUEV morphology manifest already exists: {target}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return TUEVMorphologyManifestArtifact(
        path=target,
        bundle_manifest_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        source_manifest_sha256=receipt_sha,
    )


def load_tuev_morphology_manifest(
    path: str | Path,
    *,
    expected_bundle_manifest_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> TUEVMorphologyManifest:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("Morphology manifest bundle must be a canonical regular directory")
    expected_files = {_BUNDLE_MANIFEST_FILE, _BUNDLE_RECEIPT_FILE}
    actual_files = {item.name for item in source.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            "Morphology manifest bundle contains missing or unknown files; "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    bundle_path = source / _BUNDLE_MANIFEST_FILE
    receipt_path = source / _BUNDLE_RECEIPT_FILE
    if any(path.is_symlink() or not path.is_file() for path in (bundle_path, receipt_path)):
        raise ValueError("Morphology manifest members must be regular files")
    bundle_bytes = bundle_path.read_bytes()
    if not 1 <= len(bundle_bytes) <= _MAX_MANIFEST_BYTES:
        raise ValueError("Morphology bundle manifest size is invalid")
    bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
    if expected_bundle_manifest_sha256 is not None and bundle_sha != _sha(
        expected_bundle_manifest_sha256, field="expected_bundle_manifest_sha256"
    ):
        raise ValueError("Morphology bundle manifest SHA-256 mismatch")
    bundle = _parse_canonical_json(bundle_bytes, label="manifest.json")
    _require_exact_fields(bundle, _BUNDLE_FIELDS, label="manifest.json")
    if bundle["schema_version"] != TUEV_MORPHOLOGY_BUNDLE_SCHEMA:
        raise ValueError("Unsupported morphology bundle schema")
    if bundle["serialization"] != "canonical_json_no_pickle" or bundle["receipt_file"] != _BUNDLE_RECEIPT_FILE:
        raise ValueError("Morphology bundle violates its safe serialization contract")
    receipt_size = _integer(bundle["receipt_size_bytes"], field="receipt_size_bytes", minimum=1)
    if receipt_size > _MAX_RECEIPT_BYTES or receipt_path.stat().st_size != receipt_size:
        raise ValueError("Morphology receipt size mismatch")
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha != _sha(bundle["receipt_sha256"], field="receipt_sha256"):
        raise ValueError("Morphology receipt SHA-256 mismatch")
    manifest = _manifest_from_payload(
        _parse_canonical_json(receipt_bytes, label="receipt.json")
    )
    if receipt_sha != manifest.manifest_sha256 or receipt_sha != _sha(
        bundle["source_manifest_sha256"], field="source_manifest_sha256"
    ):
        raise ValueError("Reconstructed morphology source manifest SHA mismatch")
    if expected_source_manifest_sha256 is not None and receipt_sha != _sha(
        expected_source_manifest_sha256, field="expected_source_manifest_sha256"
    ):
        raise ValueError("Morphology source manifest SHA-256 mismatch")
    expected_counts = (
        len(manifest.records),
        len(manifest.interval_groups),
        manifest.target_count,
        len(manifest.omissions),
    )
    observed_counts = tuple(
        _integer(bundle[field], field=field)
        for field in ("record_count", "interval_group_count", "target_count", "omission_count")
    )
    if observed_counts != expected_counts:
        raise ValueError("Morphology bundle counts disagree with its receipt")
    return manifest


def load_authorized_tuev_morphology_manifest(
    path: str | Path,
    authorization: VerifiedTUEVMorphologyCohortAuthorization,
    *,
    expected_bundle_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    expected_count_semantics: str = FOLD_COUNT_SEMANTICS,
) -> TUEVMorphologyManifest:
    """Load a hash-pinned manifest and verify its replay-derived group roles.

    Canonical JSON and recomputed outer hashes prove only that a bundle is
    internally self-consistent.  This boundary additionally requires the
    opaque authorization derived from the strict public ledger/OOF loaders and
    the live TUEV roster.  A rewritten bundle therefore cannot authorize a
    caller-selected fit, held, or excluded roster.
    """

    if not isinstance(
        authorization, VerifiedTUEVMorphologyCohortAuthorization
    ):
        raise TypeError(
            "authorization must be a derived TUEV morphology cohort capability"
        )
    if expected_count_semantics not in {
        HOLDING_COUNT_SEMANTICS,
        FOLD_COUNT_SEMANTICS,
    }:
        raise ValueError("expected_count_semantics is not a formal manifest role")
    manifest = load_tuev_morphology_manifest(
        path,
        expected_bundle_manifest_sha256=_sha(
            expected_bundle_manifest_sha256,
            field="expected_bundle_manifest_sha256",
        ),
        expected_source_manifest_sha256=_sha(
            expected_source_manifest_sha256,
            field="expected_source_manifest_sha256",
        ),
    )
    common_observed = (
        manifest.cohort_authorization_sha256,
        manifest.global_ledger_sha256,
        manifest.source_roster_sha256,
        manifest.duplicate_ledger_sha256,
        manifest.eligible_group_ids,
        manifest.excluded_group_ids,
    )
    common_authorized = (
        authorization.receipt_sha256,
        authorization.public_ledger_sha256,
        authorization.source_roster_sha256,
        authorization.duplicate_ledger_sha256,
        authorization.eligible_group_ids,
        authorization.excluded_group_ids,
    )
    if common_observed != common_authorized:
        raise ValueError(
            "Morphology manifest does not match the replay-derived cohort "
            "authorization"
        )
    if manifest.count_semantics != expected_count_semantics:
        raise ValueError("Morphology manifest count/role semantics are unexpected")
    if expected_count_semantics == HOLDING_COUNT_SEMANTICS:
        expected_roles = ((), ())
    else:
        expected_roles = (
            authorization.fit_group_ids,
            authorization.held_group_ids,
        )
    if (manifest.fit_group_ids, manifest.held_group_ids) != expected_roles:
        raise ValueError(
            "Morphology manifest fit/held roles were not issued by the "
            "authorization"
        )
    return manifest


def replay_tuev_morphology_source_bindings(
    manifest: TUEVMorphologyManifest, edf_root: str | Path
) -> None:
    """Re-hash source/group files and reject path, parent, or byte substitution."""

    if not isinstance(manifest, TUEVMorphologyManifest):
        raise TypeError("manifest must be TUEVMorphologyManifest")
    discovered = discover_tuev_morphology_sources(edf_root)
    replayed_duplicate_ledger = build_tuev_exact_signal_duplicate_ledger(
        discovered
    )
    if (
        replayed_duplicate_ledger != manifest.duplicate_ledger
        or replayed_duplicate_ledger.ledger_sha256
        != manifest.duplicate_ledger_sha256
    ):
        raise ValueError("Replayed TUEV exact-EDF-byte duplicate ledger changed")
    by_record = {source.record_id: source for source in discovered}
    if set(by_record) != {record.record_id for record in manifest.records}:
        raise ValueError("Replayed TUEV record roster differs from the formal manifest")
    for record in manifest.records:
        source = by_record[record.record_id]
        expected = (
            source.relative_edf_path,
            source.relative_rec_path,
            source.official_split,
            source.group_id,
            source.group_kind,
            source.source_subject_id,
            source.edf_sha256,
            source.rec_sha256,
            source.derivative_files,
            source.parent_group_files,
            source.group_file_roster_sha256,
        )
        observed = (
            record.relative_edf_path,
            record.relative_rec_path,
            record.official_split,
            record.parent_group_id,
            record.group_kind,
            record.source_subject_id,
            record.edf_sha256,
            record.rec_sha256,
            record.derivative_files,
            record.parent_group_files,
            record.group_file_roster_sha256,
        )
        if observed != expected:
            raise ValueError(f"TUEV source/group binding drifted for {record.record_id}")


__all__ = [
    "EVAL_GROUP_KIND",
    "FOLD_COUNT_SEMANTICS",
    "HOLDING_COUNT_SEMANTICS",
    "MORPHOLOGY_ALIGNMENT_TOLERANCE_SEC",
    "MORPHOLOGY_CONTEXT_SAMPLES",
    "MORPHOLOGY_DURATION_TOLERANCE_SEC",
    "MORPHOLOGY_OUTPUT_SFREQ_HZ",
    "MORPHOLOGY_OVERLAP_TOLERANCE_SEC",
    "MORPHOLOGY_TARGET_SAMPLES",
    "MORPHOLOGY_WARMUP_SAMPLES",
    "TRAIN_GROUP_KIND",
    "TUEVContentGroupComponent",
    "TUEVDuplicateRecordDecision",
    "TUEVExactSignalDuplicateClass",
    "TUEVExactSignalDuplicateLedger",
    "TUEVMorphologyIntervalGroup",
    "TUEVMorphologyManifest",
    "TUEVMorphologyManifestArtifact",
    "TUEVMorphologyOmission",
    "TUEVMorphologyPreflightArtifact",
    "TUEVMorphologyRecordMetadata",
    "TUEVMorphologyRecordReceipt",
    "TUEVMorphologySourceRecord",
    "TUEVMorphologyTarget",
    "VerifiedTUEVMorphologyCohortAuthorization",
    "VerifiedTUEVMorphologyPreflight",
    "VerifiedTUEVMorphologyPublicProtocol",
    "TUEV_MORPHOLOGY_BUNDLE_SCHEMA",
    "TUEV_MORPHOLOGY_COHORT_AUTHORIZATION_SCHEMA",
    "TUEV_MORPHOLOGY_DUPLICATE_IDENTITY_BASIS",
    "TUEV_MORPHOLOGY_DUPLICATE_LEDGER_SCHEMA",
    "TUEV_MORPHOLOGY_DUPLICATE_POLICY",
    "TUEV_MORPHOLOGY_EXTERNAL_METADATA_SCHEMA",
    "TUEV_MORPHOLOGY_HOLDING_TARGET_UPPER_BOUND",
    "TUEV_MORPHOLOGY_MANIFEST_SCHEMA",
    "TUEV_MORPHOLOGY_POLICY",
    "TUEV_MORPHOLOGY_GROUP_ASSIGNMENT_POLICY",
    "TUEV_MORPHOLOGY_PUBLIC_CONTENT_POLICY",
    "TUEV_MORPHOLOGY_PREFLIGHT_BUNDLE_SCHEMA",
    "TUEV_MORPHOLOGY_PREFLIGHT_POLICY",
    "TUEV_MORPHOLOGY_PREFLIGHT_SCHEMA",
    "build_tuev_exact_signal_duplicate_ledger",
    "build_tuev_morphology_manifest",
    "build_tuev_morphology_manifest_for_testing",
    "authorize_tuev_morphology_cohort",
    "derive_tuev_morphology_fold_manifest",
    "derive_tuev_morphology_fold_manifest_for_testing",
    "discover_tuev_morphology_sources",
    "issue_tuev_morphology_public_protocol_for_testing",
    "load_authorized_tuev_morphology_manifest",
    "load_tuev_morphology_preflight",
    "load_tuev_morphology_manifest",
    "load_tuev_morphology_public_protocol",
    "materialize_tuev_morphology_preflight",
    "replay_tuev_morphology_source_bindings",
    "save_tuev_morphology_manifest",
]
