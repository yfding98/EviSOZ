"""Frozen official-train TUSZ manifest and complete patient-bag preparation.

This module is intentionally CLI-free.  Discovery is restricted to the
canonical ``edf/train/<patient>/<session>/<montage>/<record>.edf`` tree.  A
caller must provide a :class:`~src.soz.data.overlap.ConceptTrainingCohort`;
the cohort ledger, rather than path heuristics repeated here, decides which
official-train source records are protected by DeepSOZ patient/file/content
overlap.

Only official global ``TERM`` rows whose labels belong to the frozen TUSZ
seizure-type vocabulary start anchor events. TUSZ edge annotations remain
ictal-involvement targets and are never expanded into endpoint or SOZ labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Callable, Iterator, Sequence

import torch

from ..concept_training import IctalPatientBag
from .edf import (
    CausalEDFConfig,
    EDFEventEligibilityError,
    LoadedEDFEvent,
    load_standard19_edf_event,
)
from .overlap import (
    ConceptCohortReceipt,
    ConceptTrainingCohort,
    PublicDataRecord,
    normalize_public_patient_key,
)
from .tusz import (
    TUSZAnnotationPairSummary,
    TUSZ_EVENT_ANCHOR_SEMANTICS,
    TUSZ_SEIZURE_TYPE_LABELS,
    TUSZIctalInvolvementTarget,
    inspect_tusz_annotation_pair,
    load_tusz_ictal_involvement_target,
)


TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA = "tusz_ictal_training_manifest_v4.0.0"
TUSZ_ICTAL_TRAINING_BUNDLE_SCHEMA = "tusz_ictal_training_bundle_v4.0.0"
TUSZ_DUPLICATE_TASK_SEMANTICS_POLICY = (
    "official_global_ictal_events_types_targets_masks_bin_states_v2"
)
TUSZ_OFFICIAL_SPLIT = "train"
_PATIENT_RE = re.compile(r"[a-z0-9]{8}")
_SESSION_RE = re.compile(r"s[0-9]{3}_[0-9]{4}")
_RECORD_RE = re.compile(r"[a-z0-9]{8}_s[0-9]{3}_t[0-9]{3}")
_MONTAGES = frozenset({"01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
ReaderFactory = Callable[[str], object]
_BUNDLE_MANIFEST_FILE = "manifest.json"
_BUNDLE_RECEIPT_FILE = "receipt.json"
_MAX_BUNDLE_MANIFEST_BYTES = 64 * 1024
_MAX_BUNDLE_RECEIPT_BYTES = 128 * 1024 * 1024

_BUNDLE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "receipt_file",
        "receipt_sha256",
        "receipt_size_bytes",
        "source_manifest_sha256",
        "event_count",
        "patient_count",
        "omission_count",
        "duplicate_edf_alias_count",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "derived_from_manifest_sha256",
        "cohort_receipt_sha256",
        "cohort_receipt",
        "preprocess_config",
        "preflight_performed",
        "discovered_source_count",
        "duplicate_edf_aliases",
        "authorized_source_record_sha256s",
        "excluded_source_record_sha256s",
        "events",
        "omissions",
    }
)
_COHORT_RECEIPT_FIELDS = frozenset(
    {
        "ledger_sha256",
        "ledger_receipt_sha256",
        "target_split",
        "concept_datasets",
        "heldout_target_patient_keys",
        "heldout_target_roster_sha256",
        "allowed_record_sha256s",
        "must_exclude_record_sha256s",
        "allowed_roster_sha256",
        "must_exclude_roster_sha256",
        "exclusion_reason_roster_sha256",
        "schema_version",
    }
)
_PREPROCESS_CONFIG_FIELDS = frozenset(
    {
        "output_sfreq_hz",
        "highpass_hz",
        "lowpass_hz",
        "butterworth_order",
        "warmup_sec",
        "pre_onset_sec",
        "post_onset_sec",
        "fir_half_length_per_rate",
        "flatline_run_sec",
        "clipping_run_sec",
        "qc_tolerance_volts",
        "reference_policy",
        "sensitivity_reference",
        "apply_car19",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "patient_id",
        "session_id",
        "montage",
        "record_id",
        "relative_edf_path",
        "event_id",
        "event_index",
        "event_t0_sec",
        "event_stop_sec",
        "seizure_type",
        "global_event_count",
        "edf_sha256",
        "signal_content_sha256",
        "public_record_sha256",
        "channel_annotation_sha256",
        "global_annotation_sha256",
        "annotation_pair_sha256",
        "target_sha256",
        "target_mask_sha256",
        "bin_states_sha256",
        "observed_label_count",
        "signal_preflight_receipt_sha256",
        "dataset",
        "official_split",
        "target_semantics",
        "event_anchor_semantics",
        "schema_version",
    }
)
_OMISSION_FIELDS = frozenset(
    {
        "patient_id",
        "relative_edf_path",
        "edf_sha256",
        "public_record_sha256",
        "reasons",
        "event_id",
    }
)
_DUPLICATE_EDF_ALIAS_FIELDS = frozenset(
    {
        "alias_relative_edf_path",
        "canonical_relative_edf_path",
        "patient_id",
        "official_split",
        "exact_edf_sha256",
        "canonical_channel_annotation_sha256",
        "alias_channel_annotation_sha256",
        "canonical_global_annotation_sha256",
        "alias_global_annotation_sha256",
        "task_semantics_sha256",
        "task_semantics_policy",
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
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    try:
        canonical = _canonical_json(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains a non-canonical JSON value") from exc
    if raw != canonical:
        raise ValueError(f"{label} must use canonical JSON bytes")
    return payload


def _require_exact_fields(
    payload: dict[str, object], expected: frozenset[str], *, label: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _require_json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _require_json_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a JSON string")
    return value


def _require_json_int(
    value: object, *, field: str, minimum: int = 0
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be a JSON integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _require_json_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _require_json_string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a JSON string array")
    return tuple(value)


def _file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_fingerprint != after_fingerprint:
        raise RuntimeError(f"File changed while hashing: {path.name}")
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return text


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(values.shape), "dtype": str(values.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def tusz_signal_preflight_receipt_sha256(loaded: LoadedEDFEvent) -> str:
    """Hash one loaded TUSZ signal event using the frozen preflight contract.

    This public helper lets target-free cache materializers replay the exact
    signal/QC receipt stored in :class:`TUSZIctalEventRecord` without reading
    or reconstructing ictal target sidecars.
    """

    if not isinstance(loaded, LoadedEDFEvent):
        raise TypeError("loaded must be a LoadedEDFEvent")
    return _canonical_sha256(
        {
            "edf_receipt": asdict(loaded.edf_receipt),
            "signal_receipt": asdict(loaded.signal_receipt),
            "window": {
                "shape": list(loaded.window.data.shape),
                "tensor_sha256": _tensor_sha256(loaded.window.data),
                "sfreq_hz": loaded.window.sfreq_hz,
                "start_sec": loaded.window.start_sec,
                "stop_sec": loaded.window.stop_sec,
                "onset_index": loaded.window.onset_index,
                "onset_sample_in_record": loaded.window.onset_sample_in_record,
                "requested_onset_sec": loaded.window.requested_onset_sec,
                "aligned_onset_sec": loaded.window.aligned_onset_sec,
                "alignment_error_sec": loaded.window.alignment_error_sec,
                "onset_rounding": loaded.window.onset_rounding,
            },
        }
    )


# Backward-compatible internal alias.  Keep all historical call sites on the
# exact same implementation rather than duplicating the canonical hash logic.
_preflight_receipt_sha256 = tusz_signal_preflight_receipt_sha256


def _validate_training_preprocess_config(config: CausalEDFConfig) -> None:
    if not isinstance(config, CausalEDFConfig):
        raise TypeError("preprocess_config must be a CausalEDFConfig")
    if config.reference_policy != "primary_ref" or config.sensitivity_reference is not None:
        raise ValueError("TUSZ concept training requires explicit physical -REF inputs")
    if not config.apply_car19:
        raise ValueError("TUSZ concept training requires the frozen CAR19 model input")
    if (
        abs(config.output_sfreq_hz - 200.0) > 1e-12
        or abs(config.pre_onset_sec - 12.0) > 1e-12
        or abs(config.post_onset_sec - 48.0) > 1e-12
    ):
        raise ValueError("TUSZ concept training requires frozen 200 Hz [-12,+48) windows")


def _validated_edf_root(edf_root: str | Path) -> Path:
    lexical = Path(edf_root).absolute()
    if lexical.is_symlink():
        raise ValueError("TUSZ EDF root may not be a symlink")
    if not lexical.is_dir():
        raise FileNotFoundError(lexical)
    root = lexical.resolve(strict=True)
    if root != lexical:
        raise ValueError("TUSZ EDF root must use its canonical non-symlink path")
    if root.name != "edf":
        raise ValueError("TUSZ root must be the version's canonical edf directory")
    train = root / TUSZ_OFFICIAL_SPLIT
    if train.is_symlink():
        raise ValueError("TUSZ official train directory may not be a symlink")
    if not train.is_dir():
        raise FileNotFoundError(train)
    return root


def _require_regular_unsymlinked(path: Path, *, root: Path, label: str) -> Path:
    lexical = path.absolute()
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} lies outside the TUSZ EDF root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} contains a non-canonical path component")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} may not traverse a symlink")
    if not lexical.is_file():
        raise FileNotFoundError(lexical)
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise ValueError(f"{label} must use its canonical path")
    return resolved


@dataclass(frozen=True)
class TUSZOfficialTrainFile:
    """One canonical official-train EDF and its strict sibling sidecars."""

    edf_root: Path
    edf_path: Path
    channel_annotation_path: Path
    global_annotation_path: Path
    relative_edf_path: str
    patient_id: str
    session_id: str
    montage: str
    record_id: str

    def __post_init__(self) -> None:
        if not self.edf_root.is_absolute() or not self.edf_path.is_absolute():
            raise ValueError("Discovered TUSZ paths must be absolute")
        if self.patient_id != normalize_public_patient_key(self.patient_id):
            raise ValueError("TUSZ patient ID is not canonical")
        if not _PATIENT_RE.fullmatch(self.patient_id):
            raise ValueError("TUSZ patient ID must be the canonical eight-character ID")
        if not _SESSION_RE.fullmatch(self.session_id):
            raise ValueError("TUSZ session directory is not canonical")
        if self.montage not in _MONTAGES:
            raise ValueError("TUSZ montage directory is not supported")
        if not _RECORD_RE.fullmatch(self.record_id):
            raise ValueError("TUSZ record basename is not canonical")
        relative = PurePosixPath(self.relative_edf_path)
        if relative.is_absolute() or len(relative.parts) != 5:
            raise ValueError("TUSZ relative EDF path must have five components")
        if relative.parts[0] != TUSZ_OFFICIAL_SPLIT:
            raise ValueError("Only TUSZ official train files are permitted")


@dataclass(frozen=True)
class TUSZDuplicateEDFAlias:
    """One non-training path whose EDF bytes equal a canonical source.

    The alias is audited but never creates another event or optimizer sample.
    Sidecar bytes may differ, so all four sidecar hashes are retained. The
    canonical sidecars are selected only after the global-event/target task
    semantics have been proven identical under ``task_semantics_policy``.
    """

    alias_relative_edf_path: str
    canonical_relative_edf_path: str
    patient_id: str
    official_split: str
    exact_edf_sha256: str
    canonical_channel_annotation_sha256: str
    alias_channel_annotation_sha256: str
    canonical_global_annotation_sha256: str
    alias_global_annotation_sha256: str
    task_semantics_sha256: str
    task_semantics_policy: str = TUSZ_DUPLICATE_TASK_SEMANTICS_POLICY

    def __post_init__(self) -> None:
        patient = normalize_public_patient_key(self.patient_id)
        object.__setattr__(self, "patient_id", patient)
        if not _PATIENT_RE.fullmatch(patient):
            raise ValueError("Duplicate EDF alias patient ID is not canonical")
        if self.official_split != TUSZ_OFFICIAL_SPLIT:
            raise ValueError("Duplicate EDF aliases must belong to official train")
        paths = {
            "alias_relative_edf_path": self.alias_relative_edf_path,
            "canonical_relative_edf_path": self.canonical_relative_edf_path,
        }
        for field, value in paths.items():
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or len(path.parts) != 5
                or path.parts[0] != self.official_split
                or path.parts[1] != patient
                or path.suffix != ".edf"
            ):
                raise ValueError(f"{field} is not a canonical official-train EDF path")
        if self.canonical_relative_edf_path >= self.alias_relative_edf_path:
            raise ValueError(
                "Duplicate EDF canonical path must be lexicographically first"
            )
        object.__setattr__(
            self,
            "exact_edf_sha256",
            _require_sha256(self.exact_edf_sha256, field="exact_edf_sha256"),
        )
        for field in (
            "canonical_channel_annotation_sha256",
            "alias_channel_annotation_sha256",
            "canonical_global_annotation_sha256",
            "alias_global_annotation_sha256",
            "task_semantics_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.task_semantics_policy != TUSZ_DUPLICATE_TASK_SEMANTICS_POLICY:
            raise ValueError("Duplicate EDF task-semantics policy cannot be changed")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _TUSZOfficialTrainDiscovery:
    canonical_files: tuple[TUSZOfficialTrainFile, ...]
    canonical_edf_sha256s: tuple[tuple[str, str], ...]
    duplicate_edf_aliases: tuple[TUSZDuplicateEDFAlias, ...]
    discovered_source_count: int

    def __post_init__(self) -> None:
        if not self.canonical_files:
            raise ValueError("TUSZ official train tree contains no EDF files")
        paths = tuple(item.relative_edf_path for item in self.canonical_files)
        if paths != tuple(sorted(set(paths))):
            raise RuntimeError("Canonical TUSZ discovery paths are not unique and sorted")
        expected_hash_paths = tuple(path for path, _ in self.canonical_edf_sha256s)
        if expected_hash_paths != paths:
            raise RuntimeError("Canonical TUSZ discovery hashes disagree with its paths")
        ordered_aliases = tuple(
            sorted(
                self.duplicate_edf_aliases,
                key=lambda item: item.alias_relative_edf_path,
            )
        )
        if ordered_aliases != self.duplicate_edf_aliases:
            raise RuntimeError("Duplicate TUSZ EDF aliases are not canonically sorted")
        if self.discovered_source_count != len(paths) + len(
            self.duplicate_edf_aliases
        ):
            raise RuntimeError("TUSZ discovery did not account for every physical EDF path")


def parse_tusz_official_train_path(
    edf_root: str | Path,
    edf_path: str | Path,
) -> TUSZOfficialTrainFile:
    """Parse a canonical TUSZ path; dev/eval/private or renamed files fail."""

    root = _validated_edf_root(edf_root)
    candidate = Path(edf_path)
    if any(part == ".." for part in candidate.parts):
        raise ValueError("TUSZ EDF path may not contain parent traversal")
    if not candidate.is_absolute():
        candidate = root / candidate
    source = _require_regular_unsymlinked(candidate, root=root, label="TUSZ EDF")
    relative = source.relative_to(root)
    if len(relative.parts) != 5 or relative.parts[0] != TUSZ_OFFICIAL_SPLIT:
        raise ValueError(
            "TUSZ EDF must follow train/<patient>/<session>/<montage>/<record>.edf"
        )
    _, patient, session, montage, filename = relative.parts
    if not _PATIENT_RE.fullmatch(patient):
        raise ValueError("TUSZ patient path component is not canonical")
    if not _SESSION_RE.fullmatch(session):
        raise ValueError("TUSZ session path component is not canonical")
    if montage not in _MONTAGES:
        raise ValueError("TUSZ montage path component is not canonical")
    if source.suffix != ".edf" or source.name != source.name.lower():
        raise ValueError("TUSZ EDF filename must use the lowercase .edf suffix")
    record_id = source.stem
    session_prefix = session.split("_", maxsplit=1)[0]
    expected_pattern = re.compile(
        rf"{re.escape(patient)}_{re.escape(session_prefix)}_t[0-9]{{3}}"
    )
    if not _RECORD_RE.fullmatch(record_id) or not expected_pattern.fullmatch(record_id):
        raise ValueError("TUSZ EDF basename disagrees with patient/session path")
    channel = _require_regular_unsymlinked(
        source.with_suffix(".csv"), root=root, label="TUSZ channel annotation"
    )
    global_annotation = _require_regular_unsymlinked(
        source.with_suffix(".csv_bi"), root=root, label="TUSZ global annotation"
    )
    return TUSZOfficialTrainFile(
        edf_root=root,
        edf_path=source,
        channel_annotation_path=channel,
        global_annotation_path=global_annotation,
        relative_edf_path=relative.as_posix(),
        patient_id=patient,
        session_id=session,
        montage=montage,
        record_id=record_id,
    )


def _duplicate_task_semantics(
    source: TUSZOfficialTrainFile,
) -> tuple[TUSZAnnotationPairSummary, dict[str, object]]:
    """Extract only the ictal-manifest semantics used after deduplication."""

    summary = inspect_tusz_annotation_pair(
        source.channel_annotation_path,
        source.global_annotation_path,
        source_path=source.edf_path,
    )
    events: list[dict[str, object]] = []
    for global_event in summary.global_seizure_events:
        target = load_tusz_ictal_involvement_target(
            source.channel_annotation_path,
            source.global_annotation_path,
            event_index=global_event.event_index,
            source_path=source.edf_path,
        )
        events.append(
            {
                "event_index": global_event.event_index,
                "event_t0_sec": target.event_t0_sec,
                "event_stop_sec": target.event_stop_sec,
                "seizure_type": global_event.seizure_type,
                "target_sha256": _tensor_sha256(target.targets),
                # Persisted field name is retained for formal-v3 compatibility;
                # the tensor is strictly the source-only TUSZ supervision mask.
                "target_mask_sha256": _tensor_sha256(
                    target.source_target_mask
                ),
                "bin_states_sha256": _canonical_sha256(target.bin_states),
            }
        )
    return summary, {
        "global_seizure_events": tuple(
            asdict(event) for event in summary.global_seizure_events
        ),
        "event_targets": tuple(events),
        "task_semantics_policy": TUSZ_DUPLICATE_TASK_SEMANTICS_POLICY,
    }


def _discover_tusz_official_train_files_with_aliases(
    edf_root: str | Path,
) -> _TUSZOfficialTrainDiscovery:
    """Hash once, collapse same-patient exact-byte aliases, and audit all paths."""

    root = _validated_edf_root(edf_root)
    paths = tuple(sorted((root / TUSZ_OFFICIAL_SPLIT).rglob("*.edf")))
    if not paths:
        raise ValueError("TUSZ official train tree contains no EDF files")
    discovered = tuple(parse_tusz_official_train_path(root, path) for path in paths)
    relative_paths = tuple(item.relative_edf_path for item in discovered)
    if len(set(relative_paths)) != len(relative_paths):
        raise RuntimeError("TUSZ discovery produced duplicate canonical paths")
    by_exact_sha: dict[str, list[TUSZOfficialTrainFile]] = {}
    for item in discovered:
        by_exact_sha.setdefault(_file_sha256(item.edf_path), []).append(item)

    canonical: list[TUSZOfficialTrainFile] = []
    canonical_hashes: list[tuple[str, str]] = []
    aliases: list[TUSZDuplicateEDFAlias] = []
    for exact_sha, group in sorted(by_exact_sha.items()):
        roles = {(item.patient_id, TUSZ_OFFICIAL_SPLIT) for item in group}
        if len(roles) != 1:
            raise ValueError(
                "Identical exact EDF bytes occur across TUSZ patients or splits"
            )
        ordered = tuple(sorted(group, key=lambda item: item.relative_edf_path))
        canonical_item = ordered[0]
        canonical.append(canonical_item)
        canonical_hashes.append((canonical_item.relative_edf_path, exact_sha))
        if len(ordered) == 1:
            continue
        canonical_summary, canonical_semantics = _duplicate_task_semantics(
            canonical_item
        )
        task_semantics_sha256 = _canonical_sha256(canonical_semantics)
        for alias in ordered[1:]:
            alias_summary, alias_semantics = _duplicate_task_semantics(alias)
            if alias_semantics != canonical_semantics:
                raise ValueError(
                    "Exact-byte TUSZ duplicates have different ictal task semantics: "
                    f"{canonical_item.relative_edf_path!r}, "
                    f"{alias.relative_edf_path!r}"
                )
            aliases.append(
                TUSZDuplicateEDFAlias(
                alias_relative_edf_path=alias.relative_edf_path,
                canonical_relative_edf_path=canonical_item.relative_edf_path,
                patient_id=alias.patient_id,
                official_split=TUSZ_OFFICIAL_SPLIT,
                exact_edf_sha256=exact_sha,
                    canonical_channel_annotation_sha256=(
                        canonical_summary.channel_annotation_sha256
                    ),
                    alias_channel_annotation_sha256=(
                        alias_summary.channel_annotation_sha256
                    ),
                    canonical_global_annotation_sha256=(
                        canonical_summary.global_annotation_sha256
                    ),
                    alias_global_annotation_sha256=(
                        alias_summary.global_annotation_sha256
                    ),
                    task_semantics_sha256=task_semantics_sha256,
                )
            )
    canonical.sort(key=lambda item: item.relative_edf_path)
    canonical_hashes.sort()
    aliases.sort(key=lambda item: item.alias_relative_edf_path)
    return _TUSZOfficialTrainDiscovery(
        canonical_files=tuple(canonical),
        canonical_edf_sha256s=tuple(canonical_hashes),
        duplicate_edf_aliases=tuple(aliases),
        discovered_source_count=len(discovered),
    )


def discover_tusz_official_train_files(
    edf_root: str | Path,
) -> tuple[TUSZOfficialTrainFile, ...]:
    """Return one canonical training source per exact EDF byte identity.

    Exact-byte aliases within one patient are deterministically collapsed to
    the lexicographically first root-relative path. Identical bytes assigned
    to different patients fail closed.
    """

    return _discover_tusz_official_train_files_with_aliases(
        edf_root
    ).canonical_files


@dataclass(frozen=True)
class TUSZManifestOmission:
    """Auditable reason an authorized source/event produced no training event."""

    patient_id: str
    relative_edf_path: str
    edf_sha256: str
    public_record_sha256: str
    reasons: tuple[str, ...]
    event_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "patient_id", normalize_public_patient_key(self.patient_id)
        )
        object.__setattr__(
            self,
            "edf_sha256",
            _require_sha256(self.edf_sha256, field="edf_sha256"),
        )
        object.__setattr__(
            self,
            "public_record_sha256",
            _require_sha256(
                self.public_record_sha256, field="public_record_sha256"
            ),
        )
        path = PurePosixPath(self.relative_edf_path)
        if path.is_absolute() or len(path.parts) != 5 or path.parts[0] != "train":
            raise ValueError("Omission path must be a canonical official-train path")
        if not self.reasons or tuple(sorted(set(self.reasons))) != self.reasons:
            raise ValueError("Omission reasons must be non-empty, unique, and sorted")
        if self.event_id is not None and not str(self.event_id).strip():
            raise ValueError("Omission event_id cannot be blank")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TUSZIctalEventRecord:
    """Immutable event identity, target lineage, and overlap authorization."""

    patient_id: str
    session_id: str
    montage: str
    record_id: str
    relative_edf_path: str
    event_id: str
    event_index: int
    event_t0_sec: float
    event_stop_sec: float
    seizure_type: str
    global_event_count: int
    edf_sha256: str
    signal_content_sha256: str
    public_record_sha256: str
    channel_annotation_sha256: str
    global_annotation_sha256: str
    annotation_pair_sha256: str
    target_sha256: str
    target_mask_sha256: str
    bin_states_sha256: str
    observed_label_count: int
    signal_preflight_receipt_sha256: str | None
    dataset: str = "tusz"
    official_split: str = TUSZ_OFFICIAL_SPLIT
    target_semantics: str = "bipolar_edge_ictal_involvement_not_soz"
    event_anchor_semantics: str = TUSZ_EVENT_ANCHOR_SEMANTICS
    schema_version: str = TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        patient = normalize_public_patient_key(self.patient_id)
        object.__setattr__(self, "patient_id", patient)
        if not _PATIENT_RE.fullmatch(patient):
            raise ValueError("Event patient ID is not a canonical TUSZ ID")
        if not _SESSION_RE.fullmatch(self.session_id) or self.montage not in _MONTAGES:
            raise ValueError("Event session/montage is not canonical")
        if not _RECORD_RE.fullmatch(self.record_id):
            raise ValueError("Event record ID is not canonical")
        relative = PurePosixPath(self.relative_edf_path)
        expected_parts = (
            "train",
            patient,
            self.session_id,
            self.montage,
            f"{self.record_id}.edf",
        )
        if relative.parts != expected_parts:
            raise ValueError("Event relative path disagrees with its identity fields")
        if (
            isinstance(self.event_index, bool)
            or not isinstance(self.event_index, int)
            or self.event_index < 0
        ):
            raise ValueError("Event index must be a non-negative integer")
        if (
            isinstance(self.global_event_count, bool)
            or not isinstance(self.global_event_count, int)
            or self.global_event_count < 1
            or self.event_index >= self.global_event_count
        ):
            raise ValueError("Event index is outside its official global ictal roster")
        expected_event_id = f"{self.record_id}__global_ictal_{self.event_index:04d}"
        if self.event_id != expected_event_id:
            raise ValueError("Event ID is not the canonical global-event identity")
        if not (
            math.isfinite(float(self.event_t0_sec))
            and math.isfinite(float(self.event_stop_sec))
            and float(self.event_t0_sec) >= 0
            and float(self.event_stop_sec) > float(self.event_t0_sec)
        ):
            raise ValueError("Event interval is invalid")
        object.__setattr__(self, "event_t0_sec", float(self.event_t0_sec))
        object.__setattr__(self, "event_stop_sec", float(self.event_stop_sec))
        if self.seizure_type not in TUSZ_SEIZURE_TYPE_LABELS:
            raise ValueError("Event seizure type is outside the frozen TUSZ vocabulary")
        for field in (
            "edf_sha256",
            "signal_content_sha256",
            "public_record_sha256",
            "channel_annotation_sha256",
            "global_annotation_sha256",
            "annotation_pair_sha256",
            "target_sha256",
            "target_mask_sha256",
            "bin_states_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.signal_preflight_receipt_sha256 is not None:
            object.__setattr__(
                self,
                "signal_preflight_receipt_sha256",
                _require_sha256(
                    self.signal_preflight_receipt_sha256,
                    field="signal_preflight_receipt_sha256",
                ),
            )
        if (
            isinstance(self.observed_label_count, bool)
            or not isinstance(self.observed_label_count, int)
            or self.observed_label_count < 1
        ):
            raise ValueError("Manifest events require at least one explicit target label")
        if self.dataset != "tusz" or self.official_split != "train":
            raise ValueError("Manifest event source must be TUSZ official train")
        if self.target_semantics != "bipolar_edge_ictal_involvement_not_soz":
            raise ValueError("TUSZ concept target cannot be relabeled as SOZ")
        if self.event_anchor_semantics != TUSZ_EVENT_ANCHOR_SEMANTICS:
            raise ValueError("TUSZ event anchor semantics cannot be changed")
        if self.schema_version != TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA:
            raise ValueError("Unexpected TUSZ ictal manifest event schema")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def event_record_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


@dataclass(frozen=True)
class TUSZIctalTrainingManifest(Sequence[TUSZIctalEventRecord]):
    """Path-independent frozen event roster for one overlap-safe cohort."""

    events: tuple[TUSZIctalEventRecord, ...]
    omissions: tuple[TUSZManifestOmission, ...]
    cohort_receipt: ConceptCohortReceipt
    preprocess_config: CausalEDFConfig
    preflight_performed: bool
    discovered_source_count: int
    authorized_source_record_sha256s: tuple[str, ...]
    excluded_source_record_sha256s: tuple[str, ...]
    duplicate_edf_aliases: tuple[TUSZDuplicateEDFAlias, ...] = ()
    derived_from_manifest_sha256: str | None = None
    schema_version: str = TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA:
            raise ValueError("Unexpected TUSZ ictal training manifest schema")
        if not isinstance(self.cohort_receipt, ConceptCohortReceipt):
            raise TypeError("Manifest requires a validated concept cohort receipt")
        if not isinstance(self.preprocess_config, CausalEDFConfig):
            raise TypeError("Manifest requires a CausalEDFConfig")
        _validate_training_preprocess_config(self.preprocess_config)
        if not isinstance(self.preflight_performed, bool):
            raise TypeError("preflight_performed must be boolean")
        if self.derived_from_manifest_sha256 is not None:
            object.__setattr__(
                self,
                "derived_from_manifest_sha256",
                _require_sha256(
                    self.derived_from_manifest_sha256,
                    field="derived_from_manifest_sha256",
                ),
            )
        if (
            isinstance(self.discovered_source_count, bool)
            or not isinstance(self.discovered_source_count, int)
            or self.discovered_source_count < 1
        ):
            raise ValueError("discovered_source_count must be a positive integer")
        if not self.events:
            raise ValueError("TUSZ ictal training manifest contains no usable events")
        event_order = tuple(
            sorted(
                self.events,
                key=lambda event: (
                    event.patient_id,
                    event.relative_edf_path,
                    event.event_index,
                ),
            )
        )
        if event_order != self.events:
            raise ValueError("Manifest events must be in canonical order")
        event_ids = tuple(event.event_id for event in self.events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Manifest event IDs must be globally unique")
        omission_order = tuple(
            sorted(
                self.omissions,
                key=lambda item: (
                    item.patient_id,
                    item.relative_edf_path,
                    "" if item.event_id is None else item.event_id,
                    item.reasons,
                ),
            )
        )
        if omission_order != self.omissions:
            raise ValueError("Manifest omissions must be in canonical order")
        authorized = tuple(
            sorted(
                _require_sha256(value, field="authorized_source_record_sha256")
                for value in self.authorized_source_record_sha256s
            )
        )
        excluded = tuple(
            sorted(
                _require_sha256(value, field="excluded_source_record_sha256")
                for value in self.excluded_source_record_sha256s
            )
        )
        if authorized != self.authorized_source_record_sha256s or len(
            set(authorized)
        ) != len(authorized):
            raise ValueError("Authorized source roster must be unique and sorted")
        if excluded != self.excluded_source_record_sha256s or len(
            set(excluded)
        ) != len(excluded):
            raise ValueError("Excluded source roster must be unique and sorted")
        if set(authorized) & set(excluded):
            raise ValueError("Authorized and excluded source rosters must be disjoint")
        if not set(authorized) <= set(self.cohort_receipt.allowed_record_sha256s):
            raise ValueError("Authorized sources disagree with the concept cohort receipt")
        if not set(excluded) <= set(
            self.cohort_receipt.must_exclude_record_sha256s
        ):
            raise ValueError("Excluded sources disagree with the concept cohort receipt")
        if any(
            not isinstance(item, TUSZDuplicateEDFAlias)
            for item in self.duplicate_edf_aliases
        ):
            raise TypeError("duplicate_edf_aliases must contain typed alias receipts")
        aliases = tuple(
            sorted(
                self.duplicate_edf_aliases,
                key=lambda item: item.alias_relative_edf_path,
            )
        )
        if aliases != self.duplicate_edf_aliases:
            raise ValueError("Duplicate EDF aliases must be canonically sorted")
        alias_paths = tuple(item.alias_relative_edf_path for item in aliases)
        if len(set(alias_paths)) != len(alias_paths):
            raise ValueError("Duplicate EDF alias paths must be unique")
        if self.discovered_source_count != (
            len(authorized) + len(excluded) + len(aliases)
        ):
            raise ValueError("Source rosters do not account for complete discovery")
        if any(event.public_record_sha256 not in authorized for event in self.events):
            raise ValueError("Manifest event was not authorized by the overlap cohort")
        if any(event.public_record_sha256 in excluded for event in self.events):
            raise ValueError("An overlap-protected source entered the event manifest")
        omitted_sources = {item.public_record_sha256 for item in self.omissions}
        event_sources = {event.public_record_sha256 for event in self.events}
        all_sources = set(authorized) | set(excluded)
        if (omitted_sources | event_sources) - all_sources:
            raise ValueError("An event or omission refers to an undiscovered source")
        if set(excluded) - omitted_sources:
            raise ValueError("Every protected discovered source needs an omission receipt")
        if set(authorized) - (omitted_sources | event_sources):
            raise ValueError("Every authorized source must yield an event or omission")
        if any(
            item.public_record_sha256 in excluded and item.event_id is not None
            for item in self.omissions
        ):
            raise ValueError("Protected source omissions must not expose event identities")
        excluded_omission_counts = {record_id: 0 for record_id in excluded}
        for item in self.omissions:
            if item.public_record_sha256 in excluded_omission_counts:
                excluded_omission_counts[item.public_record_sha256] += 1
        if any(count != 1 for count in excluded_omission_counts.values()):
            raise ValueError("Every protected source requires exactly one omission")

        source_identity: dict[str, tuple[str, str, str]] = {}
        for item in (*self.events, *self.omissions):
            identity = (
                item.patient_id,
                item.relative_edf_path,
                item.edf_sha256,
            )
            previous = source_identity.setdefault(item.public_record_sha256, identity)
            if previous != identity:
                raise ValueError("One source record has contradictory path/hash metadata")
        if set(source_identity) != all_sources:
            raise ValueError("Canonical source identities do not cover discovery exactly")
        canonical_path_identity = {
            path: (patient, exact_sha)
            for patient, path, exact_sha in source_identity.values()
        }
        if len(canonical_path_identity) != len(source_identity):
            raise ValueError("Canonical source paths must be unique")
        if set(alias_paths) & set(canonical_path_identity):
            raise ValueError("An EDF alias path cannot also be a canonical source")
        alias_group_receipts: dict[str, tuple[str, str, str, str]] = {}
        for alias in aliases:
            if canonical_path_identity.get(alias.canonical_relative_edf_path) != (
                alias.patient_id,
                alias.exact_edf_sha256,
            ):
                raise ValueError(
                    "Duplicate EDF alias disagrees with its canonical source identity"
                )
            group_receipt = (
                alias.canonical_channel_annotation_sha256,
                alias.canonical_global_annotation_sha256,
                alias.task_semantics_sha256,
                alias.task_semantics_policy,
            )
            previous_group_receipt = alias_group_receipts.setdefault(
                alias.canonical_relative_edf_path, group_receipt
            )
            if previous_group_receipt != group_receipt:
                raise ValueError(
                    "Aliases of one canonical EDF have contradictory semantic receipts"
                )
        event_annotation_identity: dict[str, tuple[str, str]] = {}
        for event in self.events:
            identity = (
                event.channel_annotation_sha256,
                event.global_annotation_sha256,
            )
            previous = event_annotation_identity.setdefault(
                event.relative_edf_path, identity
            )
            if previous != identity:
                raise ValueError("One EDF source has contradictory event sidecar hashes")
        for canonical_path, group_receipt in alias_group_receipts.items():
            event_sidecars = event_annotation_identity.get(canonical_path)
            if event_sidecars is not None and event_sidecars != group_receipt[:2]:
                raise ValueError(
                    "Duplicate EDF canonical sidecar audit disagrees with its events"
                )
        if self.preflight_performed:
            if any(event.signal_preflight_receipt_sha256 is None for event in self.events):
                raise ValueError("Preflighted manifest lacks an event preflight receipt")
        elif any(event.signal_preflight_receipt_sha256 is not None for event in self.events):
            raise ValueError("Non-preflighted manifest cannot contain preflight receipts")

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> TUSZIctalEventRecord:
        return self.events[index]

    def __iter__(self) -> Iterator[TUSZIctalEventRecord]:
        return iter(self.events)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({event.patient_id for event in self.events}))

    @property
    def canonical_source_count(self) -> int:
        """Number of unique exact-byte source identities used for accounting."""

        return len(self.authorized_source_record_sha256s) + len(
            self.excluded_source_record_sha256s
        )

    def events_for_patient(self, patient_id: object) -> tuple[TUSZIctalEventRecord, ...]:
        patient = normalize_public_patient_key(patient_id)
        events = tuple(event for event in self.events if event.patient_id == patient)
        if not events:
            raise KeyError(f"Patient {patient!r} is absent from the TUSZ manifest")
        return events

    @property
    def canonical_payload(self) -> dict[str, object]:
        """Closed, path-independent payload whose byte SHA is the manifest ID."""

        return {
            "schema_version": self.schema_version,
            "derived_from_manifest_sha256": self.derived_from_manifest_sha256,
            "cohort_receipt_sha256": self.cohort_receipt.receipt_sha256,
            "cohort_receipt": asdict(self.cohort_receipt),
            "preprocess_config": asdict(self.preprocess_config),
            "preflight_performed": self.preflight_performed,
            "discovered_source_count": self.discovered_source_count,
            "duplicate_edf_aliases": tuple(
                item.canonical_payload for item in self.duplicate_edf_aliases
            ),
            "authorized_source_record_sha256s": self.authorized_source_record_sha256s,
            "excluded_source_record_sha256s": self.excluded_source_record_sha256s,
            "events": tuple(event.canonical_payload for event in self.events),
            "omissions": tuple(item.canonical_payload for item in self.omissions),
        }

    @property
    def manifest_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


@dataclass(frozen=True)
class TUSZIctalTrainingManifestArtifact:
    """Hashes returned after atomic publication of a manifest bundle."""

    path: Path
    bundle_manifest_sha256: str
    source_manifest_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "bundle_manifest_sha256",
            "source_manifest_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.receipt_sha256 != self.source_manifest_sha256:
            raise ValueError("Receipt bytes must be the canonical source manifest")


def _cohort_receipt_from_payload(value: object) -> ConceptCohortReceipt:
    payload = _require_json_object(value, field="cohort_receipt")
    _require_exact_fields(
        payload, _COHORT_RECEIPT_FIELDS, label="cohort_receipt"
    )
    return ConceptCohortReceipt(
        ledger_sha256=_require_json_string(
            payload["ledger_sha256"], field="cohort_receipt.ledger_sha256"
        ),
        ledger_receipt_sha256=_require_json_string(
            payload["ledger_receipt_sha256"],
            field="cohort_receipt.ledger_receipt_sha256",
        ),
        target_split=_require_json_string(
            payload["target_split"], field="cohort_receipt.target_split"
        ),
        concept_datasets=_require_json_string_list(
            payload["concept_datasets"], field="cohort_receipt.concept_datasets"
        ),
        heldout_target_patient_keys=_require_json_string_list(
            payload["heldout_target_patient_keys"],
            field="cohort_receipt.heldout_target_patient_keys",
        ),
        heldout_target_roster_sha256=_require_json_string(
            payload["heldout_target_roster_sha256"],
            field="cohort_receipt.heldout_target_roster_sha256",
        ),
        allowed_record_sha256s=_require_json_string_list(
            payload["allowed_record_sha256s"],
            field="cohort_receipt.allowed_record_sha256s",
        ),
        must_exclude_record_sha256s=_require_json_string_list(
            payload["must_exclude_record_sha256s"],
            field="cohort_receipt.must_exclude_record_sha256s",
        ),
        allowed_roster_sha256=_require_json_string(
            payload["allowed_roster_sha256"],
            field="cohort_receipt.allowed_roster_sha256",
        ),
        must_exclude_roster_sha256=_require_json_string(
            payload["must_exclude_roster_sha256"],
            field="cohort_receipt.must_exclude_roster_sha256",
        ),
        exclusion_reason_roster_sha256=_require_json_string(
            payload["exclusion_reason_roster_sha256"],
            field="cohort_receipt.exclusion_reason_roster_sha256",
        ),
        schema_version=_require_json_string(
            payload["schema_version"], field="cohort_receipt.schema_version"
        ),
    )


def _preprocess_config_from_payload(value: object) -> CausalEDFConfig:
    payload = _require_json_object(value, field="preprocess_config")
    _require_exact_fields(
        payload, _PREPROCESS_CONFIG_FIELDS, label="preprocess_config"
    )
    sensitivity = payload["sensitivity_reference"]
    if sensitivity is not None and not isinstance(sensitivity, str):
        raise TypeError(
            "preprocess_config.sensitivity_reference must be null or a string"
        )
    apply_car19 = payload["apply_car19"]
    if not isinstance(apply_car19, bool):
        raise TypeError("preprocess_config.apply_car19 must be boolean")
    return CausalEDFConfig(
        output_sfreq_hz=_require_json_number(
            payload["output_sfreq_hz"], field="preprocess_config.output_sfreq_hz"
        ),
        highpass_hz=_require_json_number(
            payload["highpass_hz"], field="preprocess_config.highpass_hz"
        ),
        lowpass_hz=_require_json_number(
            payload["lowpass_hz"], field="preprocess_config.lowpass_hz"
        ),
        butterworth_order=_require_json_int(
            payload["butterworth_order"],
            field="preprocess_config.butterworth_order",
            minimum=1,
        ),
        warmup_sec=_require_json_number(
            payload["warmup_sec"], field="preprocess_config.warmup_sec"
        ),
        pre_onset_sec=_require_json_number(
            payload["pre_onset_sec"], field="preprocess_config.pre_onset_sec"
        ),
        post_onset_sec=_require_json_number(
            payload["post_onset_sec"], field="preprocess_config.post_onset_sec"
        ),
        fir_half_length_per_rate=_require_json_int(
            payload["fir_half_length_per_rate"],
            field="preprocess_config.fir_half_length_per_rate",
            minimum=1,
        ),
        flatline_run_sec=_require_json_number(
            payload["flatline_run_sec"],
            field="preprocess_config.flatline_run_sec",
        ),
        clipping_run_sec=_require_json_number(
            payload["clipping_run_sec"],
            field="preprocess_config.clipping_run_sec",
        ),
        qc_tolerance_volts=_require_json_number(
            payload["qc_tolerance_volts"],
            field="preprocess_config.qc_tolerance_volts",
        ),
        reference_policy=_require_json_string(
            payload["reference_policy"],
            field="preprocess_config.reference_policy",
        ),
        sensitivity_reference=sensitivity,
        apply_car19=apply_car19,
    )


def _event_from_payload(value: object, *, index: int) -> TUSZIctalEventRecord:
    field_prefix = f"events[{index}]"
    payload = _require_json_object(value, field=field_prefix)
    _require_exact_fields(payload, _EVENT_FIELDS, label=field_prefix)
    string_fields = (
        "patient_id",
        "session_id",
        "montage",
        "record_id",
        "relative_edf_path",
        "event_id",
        "seizure_type",
        "edf_sha256",
        "signal_content_sha256",
        "public_record_sha256",
        "channel_annotation_sha256",
        "global_annotation_sha256",
        "annotation_pair_sha256",
        "target_sha256",
        "target_mask_sha256",
        "bin_states_sha256",
        "dataset",
        "official_split",
        "target_semantics",
        "event_anchor_semantics",
        "schema_version",
    )
    normalized_strings = {
        field: _require_json_string(
            payload[field], field=f"{field_prefix}.{field}"
        )
        for field in string_fields
    }
    preflight = payload["signal_preflight_receipt_sha256"]
    if preflight is not None:
        preflight = _require_json_string(
            preflight,
            field=f"{field_prefix}.signal_preflight_receipt_sha256",
        )
    return TUSZIctalEventRecord(
        **normalized_strings,
        event_index=_require_json_int(
            payload["event_index"], field=f"{field_prefix}.event_index"
        ),
        event_t0_sec=_require_json_number(
            payload["event_t0_sec"], field=f"{field_prefix}.event_t0_sec"
        ),
        event_stop_sec=_require_json_number(
            payload["event_stop_sec"], field=f"{field_prefix}.event_stop_sec"
        ),
        global_event_count=_require_json_int(
            payload["global_event_count"],
            field=f"{field_prefix}.global_event_count",
            minimum=1,
        ),
        observed_label_count=_require_json_int(
            payload["observed_label_count"],
            field=f"{field_prefix}.observed_label_count",
            minimum=1,
        ),
        signal_preflight_receipt_sha256=preflight,
    )


def _omission_from_payload(value: object, *, index: int) -> TUSZManifestOmission:
    field_prefix = f"omissions[{index}]"
    payload = _require_json_object(value, field=field_prefix)
    _require_exact_fields(payload, _OMISSION_FIELDS, label=field_prefix)
    event_id = payload["event_id"]
    if event_id is not None:
        event_id = _require_json_string(event_id, field=f"{field_prefix}.event_id")
    return TUSZManifestOmission(
        patient_id=_require_json_string(
            payload["patient_id"], field=f"{field_prefix}.patient_id"
        ),
        relative_edf_path=_require_json_string(
            payload["relative_edf_path"],
            field=f"{field_prefix}.relative_edf_path",
        ),
        edf_sha256=_require_json_string(
            payload["edf_sha256"], field=f"{field_prefix}.edf_sha256"
        ),
        public_record_sha256=_require_json_string(
            payload["public_record_sha256"],
            field=f"{field_prefix}.public_record_sha256",
        ),
        reasons=_require_json_string_list(
            payload["reasons"], field=f"{field_prefix}.reasons"
        ),
        event_id=event_id,
    )


def _duplicate_edf_alias_from_payload(
    value: object, *, index: int
) -> TUSZDuplicateEDFAlias:
    field_prefix = f"duplicate_edf_aliases[{index}]"
    payload = _require_json_object(value, field=field_prefix)
    _require_exact_fields(
        payload, _DUPLICATE_EDF_ALIAS_FIELDS, label=field_prefix
    )
    return TUSZDuplicateEDFAlias(
        alias_relative_edf_path=_require_json_string(
            payload["alias_relative_edf_path"],
            field=f"{field_prefix}.alias_relative_edf_path",
        ),
        canonical_relative_edf_path=_require_json_string(
            payload["canonical_relative_edf_path"],
            field=f"{field_prefix}.canonical_relative_edf_path",
        ),
        patient_id=_require_json_string(
            payload["patient_id"], field=f"{field_prefix}.patient_id"
        ),
        official_split=_require_json_string(
            payload["official_split"], field=f"{field_prefix}.official_split"
        ),
        exact_edf_sha256=_require_json_string(
            payload["exact_edf_sha256"],
            field=f"{field_prefix}.exact_edf_sha256",
        ),
        canonical_channel_annotation_sha256=_require_json_string(
            payload["canonical_channel_annotation_sha256"],
            field=f"{field_prefix}.canonical_channel_annotation_sha256",
        ),
        alias_channel_annotation_sha256=_require_json_string(
            payload["alias_channel_annotation_sha256"],
            field=f"{field_prefix}.alias_channel_annotation_sha256",
        ),
        canonical_global_annotation_sha256=_require_json_string(
            payload["canonical_global_annotation_sha256"],
            field=f"{field_prefix}.canonical_global_annotation_sha256",
        ),
        alias_global_annotation_sha256=_require_json_string(
            payload["alias_global_annotation_sha256"],
            field=f"{field_prefix}.alias_global_annotation_sha256",
        ),
        task_semantics_sha256=_require_json_string(
            payload["task_semantics_sha256"],
            field=f"{field_prefix}.task_semantics_sha256",
        ),
        task_semantics_policy=_require_json_string(
            payload["task_semantics_policy"],
            field=f"{field_prefix}.task_semantics_policy",
        ),
    )


def _manifest_from_receipt_payload(
    payload: dict[str, object],
) -> TUSZIctalTrainingManifest:
    _require_exact_fields(payload, _RECEIPT_FIELDS, label="receipt.json")
    schema = _require_json_string(
        payload["schema_version"], field="receipt.schema_version"
    )
    if schema != TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported TUSZ training manifest schema: {schema!r}")
    cohort_receipt = _cohort_receipt_from_payload(payload["cohort_receipt"])
    declared_cohort_sha = _require_json_string(
        payload["cohort_receipt_sha256"], field="cohort_receipt_sha256"
    )
    if cohort_receipt.receipt_sha256 != declared_cohort_sha:
        raise ValueError("Nested concept cohort receipt SHA-256 mismatch")
    preflight = payload["preflight_performed"]
    if not isinstance(preflight, bool):
        raise TypeError("preflight_performed must be boolean")
    raw_events = payload["events"]
    raw_omissions = payload["omissions"]
    raw_aliases = payload["duplicate_edf_aliases"]
    if not isinstance(raw_events, list):
        raise TypeError("events must be a JSON object array")
    if not isinstance(raw_omissions, list):
        raise TypeError("omissions must be a JSON object array")
    if not isinstance(raw_aliases, list):
        raise TypeError("duplicate_edf_aliases must be a JSON object array")
    derived_from = payload["derived_from_manifest_sha256"]
    if derived_from is not None:
        derived_from = _require_json_string(
            derived_from, field="derived_from_manifest_sha256"
        )
    return TUSZIctalTrainingManifest(
        events=tuple(
            _event_from_payload(event, index=index)
            for index, event in enumerate(raw_events)
        ),
        omissions=tuple(
            _omission_from_payload(omission, index=index)
            for index, omission in enumerate(raw_omissions)
        ),
        duplicate_edf_aliases=tuple(
            _duplicate_edf_alias_from_payload(alias, index=index)
            for index, alias in enumerate(raw_aliases)
        ),
        derived_from_manifest_sha256=derived_from,
        cohort_receipt=cohort_receipt,
        preprocess_config=_preprocess_config_from_payload(
            payload["preprocess_config"]
        ),
        preflight_performed=preflight,
        discovered_source_count=_require_json_int(
            payload["discovered_source_count"],
            field="discovered_source_count",
            minimum=1,
        ),
        authorized_source_record_sha256s=_require_json_string_list(
            payload["authorized_source_record_sha256s"],
            field="authorized_source_record_sha256s",
        ),
        excluded_source_record_sha256s=_require_json_string_list(
            payload["excluded_source_record_sha256s"],
            field="excluded_source_record_sha256s",
        ),
        schema_version=schema,
    )


def _validate_bundle_manifest(payload: dict[str, object]) -> dict[str, object]:
    _require_exact_fields(
        payload, _BUNDLE_MANIFEST_FIELDS, label="manifest.json"
    )
    schema = _require_json_string(
        payload["schema_version"], field="manifest.schema_version"
    )
    if schema != TUSZ_ICTAL_TRAINING_BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported TUSZ manifest bundle schema: {schema!r}")
    if payload["serialization"] != "canonical_json_no_pickle":
        raise ValueError("TUSZ manifest bundle must use safe canonical JSON")
    if payload["receipt_file"] != _BUNDLE_RECEIPT_FILE:
        raise ValueError("TUSZ manifest bundle receipt filename is invalid")
    normalized = dict(payload)
    normalized["receipt_sha256"] = _require_sha256(
        _require_json_string(
            payload["receipt_sha256"], field="manifest.receipt_sha256"
        ),
        field="manifest.receipt_sha256",
    )
    normalized["source_manifest_sha256"] = _require_sha256(
        _require_json_string(
            payload["source_manifest_sha256"],
            field="manifest.source_manifest_sha256",
        ),
        field="manifest.source_manifest_sha256",
    )
    normalized["receipt_size_bytes"] = _require_json_int(
        payload["receipt_size_bytes"],
        field="manifest.receipt_size_bytes",
        minimum=1,
    )
    if normalized["receipt_size_bytes"] > _MAX_BUNDLE_RECEIPT_BYTES:
        raise ValueError("TUSZ manifest receipt exceeds the maximum accepted size")
    normalized["event_count"] = _require_json_int(
        payload["event_count"], field="manifest.event_count", minimum=1
    )
    normalized["patient_count"] = _require_json_int(
        payload["patient_count"], field="manifest.patient_count", minimum=1
    )
    normalized["omission_count"] = _require_json_int(
        payload["omission_count"], field="manifest.omission_count"
    )
    normalized["duplicate_edf_alias_count"] = _require_json_int(
        payload["duplicate_edf_alias_count"],
        field="manifest.duplicate_edf_alias_count",
    )
    if normalized["receipt_sha256"] != normalized["source_manifest_sha256"]:
        raise ValueError(
            "Canonical receipt file SHA must equal the source manifest SHA"
        )
    return normalized


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_tusz_ictal_training_manifest(
    path: str | Path,
    manifest: TUSZIctalTrainingManifest,
) -> TUSZIctalTrainingManifestArtifact:
    """Atomically publish a two-file canonical-JSON manifest bundle.

    Existing targets are always rejected.  The bundle contains no raw EEG,
    tensors, pickle payloads, or absolute source paths.
    """

    if not isinstance(manifest, TUSZIctalTrainingManifest):
        raise TypeError("manifest must be a TUSZIctalTrainingManifest")
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"TUSZ manifest bundle already exists: {target}")

    receipt_bytes = _canonical_json(manifest.canonical_payload)
    if len(receipt_bytes) < 1 or len(receipt_bytes) > _MAX_BUNDLE_RECEIPT_BYTES:
        raise ValueError("Serialized TUSZ manifest receipt has an invalid size")
    # Exercise the exact safe reconstruction path before publishing bytes.
    reconstructed = _manifest_from_receipt_payload(
        _parse_canonical_json(receipt_bytes, label="receipt.json")
    )
    if reconstructed != manifest:
        raise ValueError("TUSZ manifest is not stable under safe JSON reconstruction")
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha256 != manifest.manifest_sha256:
        raise RuntimeError("Canonical receipt bytes do not reproduce manifest_sha256")

    bundle_payload = _validate_bundle_manifest(
        {
            "schema_version": TUSZ_ICTAL_TRAINING_BUNDLE_SCHEMA,
            "serialization": "canonical_json_no_pickle",
            "receipt_file": _BUNDLE_RECEIPT_FILE,
            "receipt_sha256": receipt_sha256,
            "receipt_size_bytes": len(receipt_bytes),
            "source_manifest_sha256": manifest.manifest_sha256,
            "event_count": len(manifest),
            "patient_count": len(manifest.patient_ids),
            "omission_count": len(manifest.omissions),
            "duplicate_edf_alias_count": len(manifest.duplicate_edf_aliases),
        }
    )
    bundle_bytes = _canonical_json(bundle_payload)
    if len(bundle_bytes) > _MAX_BUNDLE_MANIFEST_BYTES:
        raise ValueError("Serialized TUSZ bundle manifest is unexpectedly large")
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        receipt_path = temporary / _BUNDLE_RECEIPT_FILE
        bundle_path = temporary / _BUNDLE_MANIFEST_FILE
        receipt_path.write_bytes(receipt_bytes)
        bundle_path.write_bytes(bundle_bytes)
        _fsync_file(receipt_path)
        _fsync_file(bundle_path)
        _fsync_directory(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"TUSZ manifest bundle already exists: {target}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return TUSZIctalTrainingManifestArtifact(
        path=target,
        bundle_manifest_sha256=bundle_sha256,
        source_manifest_sha256=manifest.manifest_sha256,
        receipt_sha256=receipt_sha256,
    )


def load_tusz_ictal_training_manifest(
    path: str | Path,
    *,
    expected_bundle_manifest_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> TUSZIctalTrainingManifest:
    """Strictly validate and reconstruct a frozen TUSZ manifest bundle."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"TUSZ manifest bundle must be a regular directory: {source}")
    if source.resolve(strict=True) != source:
        raise ValueError("TUSZ manifest bundle path may not traverse a symlink")
    actual_files = {item.name for item in source.iterdir()}
    expected_files = {_BUNDLE_MANIFEST_FILE, _BUNDLE_RECEIPT_FILE}
    if actual_files != expected_files:
        raise ValueError(
            "TUSZ manifest bundle contains missing or unknown files; "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    bundle_path = source / _BUNDLE_MANIFEST_FILE
    receipt_path = source / _BUNDLE_RECEIPT_FILE
    if any(
        member.is_symlink() or not member.is_file()
        for member in (bundle_path, receipt_path)
    ):
        raise ValueError("TUSZ manifest bundle members must be regular files")
    bundle_size = bundle_path.stat().st_size
    if bundle_size < 1 or bundle_size > _MAX_BUNDLE_MANIFEST_BYTES:
        raise ValueError("TUSZ bundle manifest file has an invalid size")
    bundle_bytes = bundle_path.read_bytes()
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    if expected_bundle_manifest_sha256 is not None:
        expected_bundle = _require_sha256(
            expected_bundle_manifest_sha256,
            field="expected_bundle_manifest_sha256",
        )
        if bundle_sha256 != expected_bundle:
            raise ValueError("TUSZ bundle manifest SHA-256 mismatch")
    bundle = _validate_bundle_manifest(
        _parse_canonical_json(bundle_bytes, label="manifest.json")
    )

    receipt_size = receipt_path.stat().st_size
    if receipt_size != bundle["receipt_size_bytes"]:
        raise ValueError("TUSZ manifest receipt file size mismatch")
    if receipt_size < 1 or receipt_size > _MAX_BUNDLE_RECEIPT_BYTES:
        raise ValueError("TUSZ manifest receipt file has an invalid size")
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha256 != bundle["receipt_sha256"]:
        raise ValueError("TUSZ manifest receipt file SHA-256 mismatch")
    receipt_payload = _parse_canonical_json(receipt_bytes, label="receipt.json")
    manifest = _manifest_from_receipt_payload(receipt_payload)
    recomputed_source_sha256 = manifest.manifest_sha256
    if recomputed_source_sha256 != bundle["source_manifest_sha256"]:
        raise ValueError("Reconstructed TUSZ source manifest SHA-256 mismatch")
    if recomputed_source_sha256 != receipt_sha256:
        raise ValueError("Receipt bytes are not the canonical source manifest payload")
    if expected_source_manifest_sha256 is not None:
        expected_source = _require_sha256(
            expected_source_manifest_sha256,
            field="expected_source_manifest_sha256",
        )
        if recomputed_source_sha256 != expected_source:
            raise ValueError("TUSZ source manifest SHA-256 mismatch")
    if len(manifest) != bundle["event_count"]:
        raise ValueError("TUSZ bundle event count disagrees with its receipt")
    if len(manifest.patient_ids) != bundle["patient_count"]:
        raise ValueError("TUSZ bundle patient count disagrees with its receipt")
    if len(manifest.omissions) != bundle["omission_count"]:
        raise ValueError("TUSZ bundle omission count disagrees with its receipt")
    if len(manifest.duplicate_edf_aliases) != bundle["duplicate_edf_alias_count"]:
        raise ValueError("TUSZ bundle duplicate-alias count disagrees with its receipt")
    return manifest


def _cohort_tusz_train_records(
    cohort: ConceptTrainingCohort,
) -> tuple[
    dict[str, PublicDataRecord],
    dict[str, tuple[PublicDataRecord, tuple[str, ...]]],
]:
    if not isinstance(cohort, ConceptTrainingCohort):
        raise TypeError("cohort must be a ConceptTrainingCohort")
    if "tusz" not in cohort.receipt.concept_datasets:
        raise ValueError("Concept cohort did not authorize TUSZ as an input dataset")
    exclusion_reason_payload = tuple(
        sorted(
            (item.record.record_sha256, item.reasons)
            for item in cohort.must_exclude_records
        )
    )
    if _canonical_sha256(exclusion_reason_payload) != (
        cohort.receipt.exclusion_reason_roster_sha256
    ):
        raise ValueError("Concept cohort exclusion reasons disagree with its receipt")
    allowed_records = tuple(
        record
        for record in cohort.allowed_records
        if record.dataset == "tusz" and record.split == "train"
    )
    excluded_records = tuple(
        item
        for item in cohort.must_exclude_records
        if item.record.dataset == "tusz" and item.record.split == "train"
    )
    allowed = {
        record.file_sha256: record
        for record in allowed_records
    }
    excluded = {
        item.record.file_sha256: (item.record, item.reasons)
        for item in excluded_records
    }
    if len(allowed) != len(allowed_records) or len(excluded) != len(
        excluded_records
    ):
        raise ValueError("Concept cohort contains duplicate TUSZ exact-file identities")
    if set(allowed) & set(excluded):
        raise RuntimeError("Cohort assigned one TUSZ file to both source rosters")
    if not allowed:
        raise ValueError("Overlap policy leaves no authorized TUSZ official-train source")
    return allowed, excluded


def _event_id(record_id: str, event_index: int) -> str:
    return f"{record_id}__global_ictal_{event_index:04d}"


def _validate_target_record(
    target: TUSZIctalInvolvementTarget,
    event: TUSZIctalEventRecord,
) -> None:
    receipt = target.receipt
    checks = {
        "event_index": receipt.selected_global_event_index == event.event_index,
        "global_event_count": receipt.global_seizure_event_count == event.global_event_count,
        "event_t0": receipt.selected_global_t0_sec == event.event_t0_sec,
        "event_stop": receipt.selected_global_stop_sec == event.event_stop_sec,
        "seizure_type": (
            receipt.selected_global_seizure_type == event.seizure_type
        ),
        "edf_sha256": receipt.source_sha256 == event.edf_sha256,
        "channel_annotation_sha256": (
            receipt.channel_annotation_sha256 == event.channel_annotation_sha256
        ),
        "global_annotation_sha256": (
            receipt.global_annotation_sha256 == event.global_annotation_sha256
        ),
        "annotation_pair_sha256": (
            receipt.annotation_pair_sha256 == event.annotation_pair_sha256
        ),
        "target_sha256": _tensor_sha256(target.targets) == event.target_sha256,
        "target_mask_sha256": (
            _tensor_sha256(target.source_target_mask)
            == event.target_mask_sha256
        ),
        "bin_states_sha256": (
            _canonical_sha256(target.bin_states) == event.bin_states_sha256
        ),
        "observed_label_count": (
            int(target.source_target_mask.sum().item())
            == event.observed_label_count
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Frozen TUSZ event {event.event_id} failed replay fields {failed}"
        )


def build_tusz_ictal_training_manifest(
    edf_root: str | Path,
    cohort: ConceptTrainingCohort,
    *,
    preflight_signal: bool = False,
    preprocess_config: CausalEDFConfig = CausalEDFConfig(),
    reader_factory: ReaderFactory | None = None,
) -> TUSZIctalTrainingManifest:
    """Build a frozen overlap-safe event roster without starting training.

    ``preflight_signal=True`` performs the same strict direct physical
    standard-19, uniform ``-REF``, causal-window load that later materializes
    a patient bag.  Expected event-level signal ineligibility is recorded as
    an explicit omission. Missing files, invalid configuration, unexpected
    reader failures, and internal errors still abort construction.
    """

    if not isinstance(preflight_signal, bool):
        raise TypeError("preflight_signal must be boolean")
    _validate_training_preprocess_config(preprocess_config)
    discovery = _discover_tusz_official_train_files_with_aliases(edf_root)
    discovered = discovery.canonical_files
    allowed_by_file, excluded_by_file = _cohort_tusz_train_records(cohort)
    source_by_relative = {source.relative_edf_path: source for source in discovered}
    discovered_hashes = {
        source_sha: source_by_relative[relative_path]
        for relative_path, source_sha in discovery.canonical_edf_sha256s
    }
    cohort_hashes = set(allowed_by_file) | set(excluded_by_file)
    if set(discovered_hashes) != cohort_hashes:
        missing_from_cohort = tuple(sorted(set(discovered_hashes) - cohort_hashes))
        missing_from_tree = tuple(sorted(cohort_hashes - set(discovered_hashes)))
        raise ValueError(
            "TUSZ discovery and frozen overlap cohort are not the same source roster; "
            f"unregistered_discovery={missing_from_cohort}, absent_from_tree={missing_from_tree}"
        )

    events: list[TUSZIctalEventRecord] = []
    omissions: list[TUSZManifestOmission] = []
    first_preflight_failure: tuple[str, str] | None = None
    authorized_source_ids: list[str] = []
    excluded_source_ids: list[str] = []
    for source_sha, source in sorted(
        discovered_hashes.items(), key=lambda item: item[1].relative_edf_path
    ):
        if source_sha in excluded_by_file:
            public_record, reasons = excluded_by_file[source_sha]
            if public_record.patient_key != source.patient_id:
                raise ValueError(
                    "Excluded cohort patient key disagrees with canonical TUSZ path"
                )
            excluded_source_ids.append(public_record.record_sha256)
            omissions.append(
                TUSZManifestOmission(
                    patient_id=source.patient_id,
                    relative_edf_path=source.relative_edf_path,
                    edf_sha256=source_sha,
                    public_record_sha256=public_record.record_sha256,
                    reasons=tuple(sorted(reasons)),
                )
            )
            continue

        public_record = allowed_by_file[source_sha]
        if public_record.patient_key != source.patient_id:
            raise ValueError(
                "Authorized cohort patient key disagrees with canonical TUSZ path"
            )
        authorized_source_ids.append(public_record.record_sha256)
        summary = inspect_tusz_annotation_pair(
            source.channel_annotation_path,
            source.global_annotation_path,
            source_path=source.edf_path,
        )
        if summary.source_sha256 != source_sha or summary.bname != source.record_id:
            raise RuntimeError("TUSZ annotation summary drifted from discovered source")
        if not summary.global_seizure_events:
            omissions.append(
                TUSZManifestOmission(
                    patient_id=source.patient_id,
                    relative_edf_path=source.relative_edf_path,
                    edf_sha256=source_sha,
                    public_record_sha256=public_record.record_sha256,
                    reasons=("no_official_global_ictal_label",),
                )
            )
            continue

        for global_event in summary.global_seizure_events:
            target = load_tusz_ictal_involvement_target(
                source.channel_annotation_path,
                source.global_annotation_path,
                event_index=global_event.event_index,
                source_path=source.edf_path,
            )
            event_id = _event_id(source.record_id, global_event.event_index)
            observed = int(target.source_target_mask.sum().item())
            if observed < 1:
                omissions.append(
                    TUSZManifestOmission(
                        patient_id=source.patient_id,
                        relative_edf_path=source.relative_edf_path,
                        edf_sha256=source_sha,
                        public_record_sha256=public_record.record_sha256,
                        event_id=event_id,
                        reasons=("no_explicit_edge_time_labels",),
                    )
                )
                continue
            preflight_sha: str | None = None
            if preflight_signal:
                try:
                    loaded = load_standard19_edf_event(
                        source.edf_path,
                        target.event_t0_sec,
                        config=preprocess_config,
                        reader_factory=reader_factory,
                    )
                except EDFEventEligibilityError as exc:
                    omissions.append(
                        TUSZManifestOmission(
                            patient_id=source.patient_id,
                            relative_edf_path=source.relative_edf_path,
                            edf_sha256=source_sha,
                            public_record_sha256=public_record.record_sha256,
                            event_id=event_id,
                            reasons=(f"signal_preflight_{exc.code}",),
                        )
                    )
                    if first_preflight_failure is None:
                        first_preflight_failure = (event_id, str(exc))
                    continue
                if loaded.edf_receipt.edf_sha256 != source_sha:
                    raise RuntimeError("EDF preflight receipt source SHA drifted")
                preflight_sha = _preflight_receipt_sha256(loaded)
            record = TUSZIctalEventRecord(
                patient_id=source.patient_id,
                session_id=source.session_id,
                montage=source.montage,
                record_id=source.record_id,
                relative_edf_path=source.relative_edf_path,
                event_id=event_id,
                event_index=global_event.event_index,
                event_t0_sec=target.event_t0_sec,
                event_stop_sec=target.event_stop_sec,
                seizure_type=global_event.seizure_type,
                global_event_count=target.receipt.global_seizure_event_count,
                edf_sha256=source_sha,
                signal_content_sha256=public_record.signal_content_sha256,
                public_record_sha256=public_record.record_sha256,
                channel_annotation_sha256=target.receipt.channel_annotation_sha256,
                global_annotation_sha256=target.receipt.global_annotation_sha256,
                annotation_pair_sha256=target.receipt.annotation_pair_sha256,
                target_sha256=_tensor_sha256(target.targets),
                target_mask_sha256=_tensor_sha256(target.source_target_mask),
                bin_states_sha256=_canonical_sha256(target.bin_states),
                observed_label_count=observed,
                signal_preflight_receipt_sha256=preflight_sha,
            )
            _validate_target_record(target, record)
            events.append(record)

    ordered_events = tuple(
        sorted(
            events,
            key=lambda event: (
                event.patient_id,
                event.relative_edf_path,
                event.event_index,
            ),
        )
    )
    ordered_omissions = tuple(
        sorted(
            omissions,
            key=lambda item: (
                item.patient_id,
                item.relative_edf_path,
                "" if item.event_id is None else item.event_id,
                item.reasons,
            ),
        )
    )
    if not ordered_events and first_preflight_failure is not None:
        first_event_id, first_failure_message = first_preflight_failure
        raise ValueError(
            "TUSZ signal preflight leaves no usable events; "
            f"first_failure={first_event_id}: {first_failure_message}"
        )
    return TUSZIctalTrainingManifest(
        events=ordered_events,
        omissions=ordered_omissions,
        cohort_receipt=cohort.receipt,
        preprocess_config=preprocess_config,
        preflight_performed=preflight_signal,
        discovered_source_count=discovery.discovered_source_count,
        authorized_source_record_sha256s=tuple(sorted(authorized_source_ids)),
        excluded_source_record_sha256s=tuple(sorted(excluded_source_ids)),
        duplicate_edf_aliases=discovery.duplicate_edf_aliases,
    )


def derive_tusz_ictal_training_manifest(
    master_manifest: TUSZIctalTrainingManifest,
    target_cohort: ConceptTrainingCohort,
) -> TUSZIctalTrainingManifest:
    """Derive a stricter cohort manifest without touching EDF or annotation files.

    The target cohort must use the exact overlap ledger and exact canonical
    TUSZ source identities bound by ``master_manifest``. It may only move
    master-authorized sources into the protected roster; authorization can
    never expand. Every protected source receives the target cohort's own
    exclusion reasons, while retained events and their preflight receipts are
    copied byte-for-byte at the dataclass level.
    """

    if not isinstance(master_manifest, TUSZIctalTrainingManifest):
        raise TypeError("master_manifest must be a TUSZIctalTrainingManifest")
    allowed_by_file, excluded_by_file = _cohort_tusz_train_records(target_cohort)
    master_receipt = master_manifest.cohort_receipt
    target_receipt = target_cohort.receipt
    if target_receipt.ledger_sha256 != master_receipt.ledger_sha256:
        raise ValueError("Target cohort was built from a different overlap ledger")
    if target_receipt.ledger_receipt_sha256 != master_receipt.ledger_receipt_sha256:
        raise ValueError("Target cohort overlap-ledger receipt drifted from master")

    target_allowed = {
        record.record_sha256: record for record in allowed_by_file.values()
    }
    target_excluded = {
        record.record_sha256: (record, reasons)
        for record, reasons in excluded_by_file.values()
    }
    if len(target_allowed) != len(allowed_by_file) or len(target_excluded) != len(
        excluded_by_file
    ):
        raise RuntimeError("Target cohort contains duplicate TUSZ source identities")
    target_all = set(target_allowed) | set(target_excluded)
    master_authorized = set(master_manifest.authorized_source_record_sha256s)
    master_excluded = set(master_manifest.excluded_source_record_sha256s)
    master_all = master_authorized | master_excluded
    if target_all != master_all:
        raise ValueError(
            "Target cohort does not account for the exact master TUSZ discovery roster"
        )
    unauthorized_expansion = set(target_allowed) - master_authorized
    if unauthorized_expansion:
        raise ValueError(
            "Target cohort attempts to expand master authorization: "
            f"{tuple(sorted(unauthorized_expansion))}"
        )
    if not master_excluded <= set(target_excluded):
        raise ValueError("Target cohort attempts to unprotect a master-excluded source")

    source_identity: dict[str, tuple[str, str, str]] = {}
    for item in (*master_manifest.events, *master_manifest.omissions):
        identity = (item.patient_id, item.relative_edf_path, item.edf_sha256)
        previous = source_identity.setdefault(item.public_record_sha256, identity)
        if previous != identity:
            raise RuntimeError("Master manifest source identity is internally inconsistent")
    if set(source_identity) != master_all:
        raise RuntimeError("Master manifest source identity roster is incomplete")
    for record_id, record in {
        **target_allowed,
        **{key: value[0] for key, value in target_excluded.items()},
    }.items():
        patient_id, _, edf_sha256 = source_identity[record_id]
        if record.patient_key != patient_id or record.file_sha256 != edf_sha256:
            raise ValueError(
                "Target cohort TUSZ path-independent identity drifted from master"
            )

    retained_events = tuple(
        event
        for event in master_manifest.events
        if event.public_record_sha256 in target_allowed
    )
    retained_omissions = [
        omission
        for omission in master_manifest.omissions
        if omission.public_record_sha256 in target_allowed
    ]
    for record_id, (_, reasons) in target_excluded.items():
        patient_id, relative_path, edf_sha256 = source_identity[record_id]
        retained_omissions.append(
            TUSZManifestOmission(
                patient_id=patient_id,
                relative_edf_path=relative_path,
                edf_sha256=edf_sha256,
                public_record_sha256=record_id,
                reasons=tuple(sorted(reasons)),
            )
        )
    ordered_omissions = tuple(
        sorted(
            retained_omissions,
            key=lambda item: (
                item.patient_id,
                item.relative_edf_path,
                "" if item.event_id is None else item.event_id,
                item.reasons,
            ),
        )
    )
    return TUSZIctalTrainingManifest(
        events=retained_events,
        omissions=ordered_omissions,
        cohort_receipt=target_receipt,
        preprocess_config=master_manifest.preprocess_config,
        preflight_performed=master_manifest.preflight_performed,
        discovered_source_count=master_manifest.discovered_source_count,
        authorized_source_record_sha256s=tuple(sorted(target_allowed)),
        excluded_source_record_sha256s=tuple(sorted(target_excluded)),
        duplicate_edf_aliases=master_manifest.duplicate_edf_aliases,
        derived_from_manifest_sha256=master_manifest.manifest_sha256,
    )


def materialize_tusz_ictal_patient_bag(
    manifest: TUSZIctalTrainingManifest,
    patient_id: object,
    edf_root: str | Path,
    *,
    reader_factory: ReaderFactory | None = None,
) -> IctalPatientBag:
    """Replay and load every frozen event for exactly one manifest patient."""

    if not isinstance(manifest, TUSZIctalTrainingManifest):
        raise TypeError("manifest must be a TUSZIctalTrainingManifest")
    root = _validated_edf_root(edf_root)
    patient_events = manifest.events_for_patient(patient_id)
    eeg: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    event_ids: list[str] = []
    for event in patient_events:
        source = parse_tusz_official_train_path(root, event.relative_edf_path)
        if (
            source.patient_id != event.patient_id
            or source.session_id != event.session_id
            or source.montage != event.montage
            or source.record_id != event.record_id
        ):
            raise ValueError("Frozen TUSZ event path replay changed source identity")
        if _file_sha256(source.edf_path) != event.edf_sha256:
            raise ValueError(f"Frozen TUSZ EDF changed after manifest: {event.event_id}")
        target = load_tusz_ictal_involvement_target(
            source.channel_annotation_path,
            source.global_annotation_path,
            event_index=event.event_index,
            source_path=source.edf_path,
        )
        _validate_target_record(target, event)
        loaded = load_standard19_edf_event(
            source.edf_path,
            event.event_t0_sec,
            config=manifest.preprocess_config,
            reader_factory=reader_factory,
        )
        if loaded.edf_receipt.edf_sha256 != event.edf_sha256:
            raise ValueError("Patient-bag EDF receipt disagrees with the manifest")
        replay_preflight_sha = _preflight_receipt_sha256(loaded)
        if (
            event.signal_preflight_receipt_sha256 is not None
            and replay_preflight_sha != event.signal_preflight_receipt_sha256
        ):
            raise ValueError("Patient-bag signal receipt disagrees with preflight")
        eeg.append(loaded.window.data.detach().to(dtype=torch.float32, device="cpu"))
        targets.append(target.targets.detach().to(dtype=torch.float32, device="cpu"))
        # Only source annotation coverage enters the Stage-3I loss bag.
        masks.append(
            target.source_target_mask.detach().to(
                dtype=torch.bool, device="cpu"
            )
        )
        event_ids.append(event.event_id)
    expected_event_ids = tuple(event.event_id for event in patient_events)
    if tuple(event_ids) != expected_event_ids:
        raise RuntimeError("Patient-bag materialization omitted a frozen event")
    return IctalPatientBag(
        patient_id=patient_events[0].patient_id,
        event_ids=tuple(event_ids),
        expected_event_ids=expected_event_ids,
        source_manifest_sha256=manifest.manifest_sha256,
        eeg_volts=torch.stack(eeg, dim=0),
        targets=torch.stack(targets, dim=0),
        target_mask=torch.stack(masks, dim=0),
    )


__all__ = [
    "TUSZ_ICTAL_TRAINING_BUNDLE_SCHEMA",
    "TUSZ_ICTAL_TRAINING_MANIFEST_SCHEMA",
    "TUSZDuplicateEDFAlias",
    "TUSZIctalEventRecord",
    "TUSZIctalTrainingManifest",
    "TUSZIctalTrainingManifestArtifact",
    "TUSZManifestOmission",
    "TUSZOfficialTrainFile",
    "build_tusz_ictal_training_manifest",
    "derive_tusz_ictal_training_manifest",
    "discover_tusz_official_train_files",
    "load_tusz_ictal_training_manifest",
    "materialize_tusz_ictal_patient_bag",
    "parse_tusz_official_train_path",
    "save_tusz_ictal_training_manifest",
    "tusz_signal_preflight_receipt_sha256",
]
