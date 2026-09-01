"""Closed source-dev native-ictal evaluation artifacts.

This module is the evaluation-only counterpart of the official-train TUSZ
concept corpus.  It derives a native edge-time target manifest exclusively
from the externally pinned DeepSOZ signal-preflight ``source_dev`` roster and
the corresponding TUSZ ``.csv``/``.csv_bi`` annotations.  DeepSOZ electrode
targets are neither read nor serialized.

The manifest and token-corpus schemas deliberately differ from every formal
training schema.  Their verified artifacts carry ``training_authorized=False``
and the dataset builder propagates that bit into every patient bag, so both
training entry points fail closed if an evaluation artifact is supplied.
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
from typing import Callable, Iterator, Mapping, Sequence

import torch

from .cached_concept_training import IctalTokenBagDataset, IctalTokenPatientBag
from .concept_token_io import (
    CONCEPT_TOKEN_SHAPE,
    LoadedLaBraMConceptTokens,
    labram_feature_receipt_sha256,
    load_labram_concept_tokens,
    save_labram_concept_tokens,
)
from .data import deepsoz_signal_preflight as _signal_preflight
from .data.deepsoz_signal_preflight import (
    DEEPSOZ_SIGNAL_PREFLIGHT_FILENAME,
    VerifiedDeepSOZSignalPreflightBundle,
)
from .data.edf import (
    CausalEDFConfig,
    EDF_PREPROCESS_SCHEMA,
    load_standard19_edf_event,
)
from .data.tusz import load_tusz_ictal_involvement_target
from .models.foundation import TiledFoundationEncoder
from .models.labram import (
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LaBraMFeatureReceipt,
    OfficialLaBraMEncoder,
    bind_labram_record_positions,
    require_feature_receipt_position_binding,
)


ICTAL_NATIVE_EVAL_PURPOSE = "ictal_native_eval_only"
ICTAL_NATIVE_EVAL_MANIFEST_SCHEMA = "soz_ictal_native_eval_manifest_v1"
ICTAL_NATIVE_EVAL_MANIFEST_ARTIFACT_SCHEMA = (
    "soz_ictal_native_eval_manifest_artifact_v1"
)
ICTAL_NATIVE_EVAL_TOKEN_CORPUS_SCHEMA = (
    "soz_ictal_native_eval_token_corpus_index_v1"
)
ICTAL_NATIVE_EVAL_TARGET_SEMANTICS = (
    "bipolar_edge_ictal_involvement_not_soz"
)
ICTAL_NATIVE_EVAL_MANIFEST_FILENAME = "manifest.json"
ICTAL_NATIVE_EVAL_TOKEN_INDEX_FILENAME = "index.json"
ICTAL_NATIVE_EVAL_EVENTS_DIRECTORY = "events"
ICTAL_NATIVE_EVAL_SERIALIZATION = "canonical_json_utf8_newline_no_pickle"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PUBLIC_PATIENT_RE = re.compile(r"[a-z0-9]{8}")
_SAFE_EVENT_RE = re.compile(r"[A-Za-z0-9_.-]+")
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_TIME_TOLERANCE_SEC = 1e-6
_VERIFIED_MANIFEST_MARKER = object()
_VERIFIED_CORPUS_MARKER = object()

_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "target_patient_id",
        "public_patient_id",
        "signal_event_record_sha256",
        "relative_edf_path",
        "relative_channel_annotation_path",
        "relative_global_annotation_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "global_seizure_type",
        "edf_sha256",
        "channel_annotation_sha256",
        "global_annotation_sha256",
        "annotation_pair_sha256",
        "preprocess_config_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
        "processed_window_shape",
        "processed_window_dtype",
        "native_target_sha256",
        "native_target_mask_sha256",
        "native_bin_states_sha256",
        "observed_label_count",
        "positive_label_count",
        "negative_label_count",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "evaluation_only",
        "training_authorized",
        "contains_soz_labels",
        "model_split",
        "official_split",
        "target_semantics",
        "source_signal_preflight_artifact_sha256",
        "source_signal_preflight_receipt_sha256",
        "preprocess_config",
        "preprocess_config_sha256",
        "event_count",
        "patient_count",
        "target_patient_count",
        "event_roster_sha256",
        "public_patient_roster_sha256",
        "target_patient_roster_sha256",
        "events",
    }
)
_MANIFEST_ARTIFACT_FIELDS = frozenset(
    {"schema_version", "serialization", "receipt_sha256", "receipt"}
)
_CORPUS_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "evaluation_only",
        "training_authorized",
        "contains_soz_labels",
        "serialization",
        "manifest",
        "signal_preflight",
        "foundation",
        "event_count",
        "patient_count",
        "event_roster_sha256",
        "patient_roster_sha256",
        "patient_event_roster_sha256",
        "tensor_roster_sha256",
        "events",
    }
)
_CORPUS_MANIFEST_FIELDS = frozenset(
    {"artifact_sha256", "receipt_sha256", "event_count", "patient_count"}
)
_CORPUS_SIGNAL_FIELDS = frozenset(
    {"artifact_sha256", "receipt_sha256", "model_split", "official_split"}
)
_CORPUS_FOUNDATION_FIELDS = frozenset(
    {
        "feature_receipt_sha256",
        "checkpoint_sha256",
        "audited_expected_checkpoint_sha256",
        "modeling_sha256",
        "audited_expected_modeling_sha256",
        "token_shape",
        "tile_seconds",
        "frozen",
        "materialization_device",
    }
)
_CORPUS_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "public_patient_id",
        "evaluation_event_record_sha256",
        "signal_event_record_sha256",
        "bundle_path",
        "bundle_manifest_sha256",
        "tensor_sha256",
    }
)


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Native-evaluation artifact is not canonical JSON data") from exc
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {"shape": list(values.shape), "dtype": str(values.dtype)}
        )
    )
    digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _signal_tensor_sha256(tensor: torch.Tensor) -> str:
    """Match the DeepSOZ signal-preflight processed-window receipt exactly."""

    values = tensor.detach().cpu().contiguous()
    metadata = _canonical_json_bytes(
        {"dtype": str(values.dtype), "shape": list(values.shape)}
    )
    raw = values.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _preprocess_config_sha256(value: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {"preprocess_schema": EDF_PREPROCESS_SCHEMA, "config": dict(value)}
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, field: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{field} violates its closed schema; "
            f"missing={sorted(set(expected)-actual)}, unknown={sorted(actual-set(expected))}"
        )


def _json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _strict_json(raw: bytes, *, field: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} contains duplicate field {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field} contains forbidden constant {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    result = _json_object(payload, field=field)
    if _canonical_json_bytes(result, newline=True) != raw:
        raise ValueError(f"{field} bytes are not canonical JSON")
    return result


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _stable_file(
    path: str | Path, *, field: str, max_bytes: int
) -> tuple[bytes, str]:
    source = _absolute_no_symlink(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular non-symlinked file")
    before = source.stat()
    if not 1 <= before.st_size <= max_bytes:
        raise ValueError(f"{field} has an invalid size")
    raw = source.read_bytes()
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{field} changed while it was read")
    return raw, hashlib.sha256(raw).hexdigest()


def _safe_relative(root: Path, value: object, *, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a canonical relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{field} must be a canonical relative path")
    candidate = _absolute_no_symlink(root.joinpath(*relative.parts), field=field)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the TUSZ root") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{field} does not resolve to a regular file")
    return relative.as_posix(), candidate


def _roster_sha256(values: Sequence[object]) -> str:
    normalized = tuple(sorted(str(value) for value in values))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("Roster must be non-empty and unique")
    return _canonical_sha256(normalized)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_new_directory(path: str | Path, *, field: str) -> Path:
    target = _absolute_no_symlink(path, field=field)
    if target.name in {"", ".", ".."}:
        raise ValueError(f"{field} requires a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(f"{field} already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"{field} parent does not exist")
    return target


def _preflight_split_roster(
    bundle: VerifiedDeepSOZSignalPreflightBundle, model_split: str
) -> tuple[str, ...]:
    rows = bundle.receipt["eligible_split_patient_ids"]
    mapping = {str(row[0]): tuple(str(value) for value in row[1]) for row in rows}
    try:
        result = mapping[model_split]
    except KeyError as exc:
        raise ValueError(f"Signal-preflight lacks {model_split!r} roster") from exc
    if result != tuple(sorted(set(result))) or not result:
        raise ValueError("Signal-preflight source-dev roster is not canonical")
    return result


def load_bound_deepsoz_signal_preflight_artifact(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedDeepSOZSignalPreflightBundle:
    """Load one externally pinned preflight receipt without DeepSOZ targets.

    The source preflight's public full-replay loader remains authoritative for
    creating that artifact.  This boundary verifies its closed canonical file,
    complete receipt schema and mandatory external hashes; downstream manifest
    and corpus loaders independently replay native annotations and EEG signals.
    """

    directory = _absolute_no_symlink(bundle_directory, field="signal-preflight bundle")
    if not directory.is_dir():
        raise FileNotFoundError("Signal-preflight bundle directory does not exist")
    entries = tuple(directory.iterdir())
    if len(entries) != 1 or entries[0].name != DEEPSOZ_SIGNAL_PREFLIGHT_FILENAME:
        raise ValueError("Signal-preflight bundle violates its closed file schema")
    raw, artifact_sha = _stable_file(
        entries[0], field="signal-preflight artifact", max_bytes=256 * 1024 * 1024
    )
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Signal-preflight artifact SHA mismatch")
    _, receipt = _signal_preflight._parse_artifact(raw)
    receipt_sha = _canonical_sha256(receipt)
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Signal-preflight receipt SHA mismatch")
    return VerifiedDeepSOZSignalPreflightBundle(
        receipt=receipt,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
    )


@dataclass(frozen=True)
class IctalNativeEvalEvent:
    event_id: str
    target_patient_id: str
    public_patient_id: str
    signal_event_record_sha256: str
    relative_edf_path: str
    relative_channel_annotation_path: str
    relative_global_annotation_path: str
    global_event_index: int
    global_t0_sec: float
    global_stop_sec: float
    global_seizure_type: str
    edf_sha256: str
    channel_annotation_sha256: str
    global_annotation_sha256: str
    annotation_pair_sha256: str
    preprocess_config_sha256: str
    edf_receipt_sha256: str
    signal_receipt_sha256: str
    processed_window_sha256: str
    processed_window_shape: tuple[int, int]
    processed_window_dtype: str
    native_target_sha256: str
    native_target_mask_sha256: str
    native_bin_states_sha256: str
    observed_label_count: int
    positive_label_count: int
    negative_label_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not _SAFE_EVENT_RE.fullmatch(self.event_id):
            raise ValueError("Evaluation event_id is not filesystem-safe")
        if not isinstance(self.target_patient_id, str) or not self.target_patient_id:
            raise ValueError("target_patient_id must be non-empty")
        if not _PUBLIC_PATIENT_RE.fullmatch(self.public_patient_id):
            raise ValueError("public_patient_id is not canonical")
        for field in (
            "signal_event_record_sha256",
            "edf_sha256",
            "channel_annotation_sha256",
            "global_annotation_sha256",
            "annotation_pair_sha256",
            "preprocess_config_sha256",
            "edf_receipt_sha256",
            "signal_receipt_sha256",
            "processed_window_sha256",
            "native_target_sha256",
            "native_target_mask_sha256",
            "native_bin_states_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        edf = PurePosixPath(self.relative_edf_path)
        channel = PurePosixPath(self.relative_channel_annotation_path)
        global_path = PurePosixPath(self.relative_global_annotation_path)
        if (
            len(edf.parts) != 5
            or edf.parts[0] != "dev"
            or edf.parts[1] != self.public_patient_id
            or channel != edf.with_suffix(".csv")
            or global_path != edf.with_suffix(".csv_bi")
        ):
            raise ValueError("Evaluation EDF/annotation paths are not canonical source-dev paths")
        if isinstance(self.global_event_index, bool) or not isinstance(
            self.global_event_index, int
        ) or self.global_event_index < 0:
            raise ValueError("global_event_index must be a non-negative integer")
        if not (
            math.isfinite(float(self.global_t0_sec))
            and math.isfinite(float(self.global_stop_sec))
            and self.global_t0_sec >= 0
            and self.global_stop_sec > self.global_t0_sec
        ):
            raise ValueError("Evaluation event interval is invalid")
        if self.processed_window_shape != (19, 12_000):
            raise ValueError("Processed evaluation window must have shape [19,12000]")
        if self.processed_window_dtype != "torch.float32":
            raise ValueError("Processed evaluation window must be torch.float32")
        counts = (
            self.observed_label_count,
            self.positive_label_count,
            self.negative_label_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("Native target counts must be non-negative integers")
        if self.observed_label_count < 1 or self.observed_label_count != (
            self.positive_label_count + self.negative_label_count
        ):
            raise ValueError("Native target support counts are inconsistent")

    @property
    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["processed_window_shape"] = list(self.processed_window_shape)
        return payload

    @property
    def event_record_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


@dataclass(frozen=True)
class IctalNativeEvalManifest(Sequence[IctalNativeEvalEvent]):
    events: tuple[IctalNativeEvalEvent, ...]
    source_signal_preflight_artifact_sha256: str
    source_signal_preflight_receipt_sha256: str
    preprocess_config: Mapping[str, object]
    preprocess_config_sha256: str
    schema_version: str = ICTAL_NATIVE_EVAL_MANIFEST_SCHEMA
    purpose: str = ICTAL_NATIVE_EVAL_PURPOSE
    evaluation_only: bool = True
    training_authorized: bool = False
    contains_soz_labels: bool = False
    model_split: str = "source_dev"
    official_split: str = "dev"
    target_semantics: str = ICTAL_NATIVE_EVAL_TARGET_SEMANTICS

    def __post_init__(self) -> None:
        if (
            self.schema_version != ICTAL_NATIVE_EVAL_MANIFEST_SCHEMA
            or self.purpose != ICTAL_NATIVE_EVAL_PURPOSE
            or self.evaluation_only is not True
            or self.training_authorized is not False
            or self.contains_soz_labels is not False
            or self.model_split != "source_dev"
            or self.official_split != "dev"
            or self.target_semantics != ICTAL_NATIVE_EVAL_TARGET_SEMANTICS
        ):
            raise ValueError("Native evaluation manifest policy is immutable")
        _require_sha256(
            self.source_signal_preflight_artifact_sha256,
            field="source_signal_preflight_artifact_sha256",
        )
        _require_sha256(
            self.source_signal_preflight_receipt_sha256,
            field="source_signal_preflight_receipt_sha256",
        )
        if not isinstance(self.preprocess_config, Mapping):
            raise TypeError("preprocess_config must be a mapping")
        config_payload = dict(self.preprocess_config)
        CausalEDFConfig(**config_payload)
        if self.preprocess_config_sha256 != _preprocess_config_sha256(config_payload):
            raise ValueError("preprocess_config_sha256 mismatch")
        if not self.events or any(not isinstance(event, IctalNativeEvalEvent) for event in self.events):
            raise ValueError("Native evaluation manifest requires typed events")
        ordered = tuple(sorted(self.events, key=lambda row: (row.public_patient_id, row.event_id)))
        if ordered != self.events:
            raise ValueError("Native evaluation events are not canonically ordered")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("Native evaluation event IDs must be unique")
        if any(event.preprocess_config_sha256 != self.preprocess_config_sha256 for event in self.events):
            raise ValueError("An event uses a different preprocessing policy")

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> IctalNativeEvalEvent:
        return self.events[index]

    def __iter__(self) -> Iterator[IctalNativeEvalEvent]:
        return iter(self.events)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({event.public_patient_id for event in self.events}))

    @property
    def target_patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({event.target_patient_id for event in self.events}))

    def events_for_patient(self, patient_id: object) -> tuple[IctalNativeEvalEvent, ...]:
        normalized = str(patient_id).strip()
        events = tuple(event for event in self.events if event.public_patient_id == normalized)
        if not events:
            raise KeyError(f"Patient {normalized!r} is absent from native evaluation")
        return events

    @property
    def canonical_payload(self) -> dict[str, object]:
        event_roster = tuple(
            (
                event.event_id,
                event.target_patient_id,
                event.public_patient_id,
                event.event_record_sha256,
                event.signal_event_record_sha256,
            )
            for event in self.events
        )
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "evaluation_only": self.evaluation_only,
            "training_authorized": self.training_authorized,
            "contains_soz_labels": self.contains_soz_labels,
            "model_split": self.model_split,
            "official_split": self.official_split,
            "target_semantics": self.target_semantics,
            "source_signal_preflight_artifact_sha256": self.source_signal_preflight_artifact_sha256,
            "source_signal_preflight_receipt_sha256": self.source_signal_preflight_receipt_sha256,
            "preprocess_config": dict(self.preprocess_config),
            "preprocess_config_sha256": self.preprocess_config_sha256,
            "event_count": len(self.events),
            "patient_count": len(self.patient_ids),
            "target_patient_count": len(self.target_patient_ids),
            "event_roster_sha256": _canonical_sha256(event_roster),
            "public_patient_roster_sha256": _roster_sha256(self.patient_ids),
            "target_patient_roster_sha256": _roster_sha256(self.target_patient_ids),
            "events": [event.canonical_payload for event in self.events],
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


@dataclass(frozen=True, init=False)
class VerifiedIctalNativeEvalManifestArtifact:
    path: Path
    artifact_sha256: str
    receipt_sha256: str
    manifest: IctalNativeEvalManifest

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        artifact_sha256: str,
        receipt_sha256: str,
        manifest: IctalNativeEvalManifest,
    ) -> None:
        if _verification_marker is not _VERIFIED_MANIFEST_MARKER:
            raise TypeError("Verified native-evaluation manifests come only from strict loaders")
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("path must be absolute")
        _require_sha256(artifact_sha256, field="artifact_sha256")
        _require_sha256(receipt_sha256, field="receipt_sha256")
        if not isinstance(manifest, IctalNativeEvalManifest):
            raise TypeError("manifest must be IctalNativeEvalManifest")
        if receipt_sha256 != manifest.receipt_sha256:
            raise ValueError("Manifest receipt SHA mismatch")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "artifact_sha256", artifact_sha256)
        object.__setattr__(self, "receipt_sha256", receipt_sha256)
        object.__setattr__(self, "manifest", manifest)


def _issue_manifest_artifact(
    *, path: Path, artifact_sha256: str, manifest: IctalNativeEvalManifest
) -> VerifiedIctalNativeEvalManifestArtifact:
    return VerifiedIctalNativeEvalManifestArtifact(
        _verification_marker=_VERIFIED_MANIFEST_MARKER,
        path=path,
        artifact_sha256=artifact_sha256,
        receipt_sha256=manifest.receipt_sha256,
        manifest=manifest,
    )


def _manifest_event_from_source(
    row: Mapping[str, object], *, tusz_root: Path
) -> IctalNativeEvalEvent:
    if row.get("model_split") != "source_dev" or row.get("official_split") != "dev":
        raise ValueError("Native evaluation accepts source_dev/dev events only")
    target_patient = str(row["patient_id"])
    public_patient = str(row["local_patient_id"])
    relative_edf, edf = _safe_relative(
        tusz_root, row["relative_edf_path"], field="relative_edf_path"
    )
    relative_channel, channel = _safe_relative(
        tusz_root,
        row["relative_channel_annotation_path"],
        field="relative_channel_annotation_path",
    )
    relative_global, global_path = _safe_relative(
        tusz_root,
        row["relative_global_annotation_path"],
        field="relative_global_annotation_path",
    )
    event_index = int(row["global_event_index"])
    target = load_tusz_ictal_involvement_target(
        channel, global_path, event_index=event_index, source_path=edf
    )
    receipt = target.receipt
    comparisons = {
        "source EDF": receipt.source_sha256 == row["edf_sha256"],
        "channel annotation": receipt.channel_annotation_sha256
        == row["channel_annotation_sha256"],
        "global annotation": receipt.global_annotation_sha256
        == row["global_annotation_sha256"],
        "annotation pair": receipt.annotation_pair_sha256
        == row["annotation_pair_sha256"],
        "global event index": receipt.selected_global_event_index == event_index,
        "global t0": abs(receipt.selected_global_t0_sec - float(row["global_t0_sec"]))
        <= _TIME_TOLERANCE_SEC,
        "global stop": abs(receipt.selected_global_stop_sec - float(row["global_stop_sec"]))
        <= _TIME_TOLERANCE_SEC,
        "seizure type": receipt.selected_global_seizure_type
        == row["global_seizure_type"],
        "target semantics": receipt.target_semantics
        == ICTAL_NATIVE_EVAL_TARGET_SEMANTICS,
        "no SOZ production": receipt.produces_soz_labels is False,
    }
    failed = tuple(name for name, passed in comparisons.items() if not passed)
    if failed:
        raise ValueError(f"Native annotation replay disagrees with preflight event: {failed}")
    observed = int(target.source_target_mask.sum().item())
    positive = int(
        target.targets[target.source_target_mask].sum().item()
    )
    return IctalNativeEvalEvent(
        event_id=str(row["event_id"]),
        target_patient_id=target_patient,
        public_patient_id=public_patient,
        signal_event_record_sha256=str(row["event_record_sha256"]),
        relative_edf_path=relative_edf,
        relative_channel_annotation_path=relative_channel,
        relative_global_annotation_path=relative_global,
        global_event_index=event_index,
        global_t0_sec=float(row["global_t0_sec"]),
        global_stop_sec=float(row["global_stop_sec"]),
        global_seizure_type=str(row["global_seizure_type"]),
        edf_sha256=str(row["edf_sha256"]),
        channel_annotation_sha256=str(row["channel_annotation_sha256"]),
        global_annotation_sha256=str(row["global_annotation_sha256"]),
        annotation_pair_sha256=str(row["annotation_pair_sha256"]),
        preprocess_config_sha256=str(row["preprocess_config_sha256"]),
        edf_receipt_sha256=str(row["edf_receipt_sha256"]),
        signal_receipt_sha256=str(row["signal_receipt_sha256"]),
        processed_window_sha256=str(row["processed_window_sha256"]),
        processed_window_shape=tuple(int(value) for value in row["processed_window_shape"]),
        processed_window_dtype=str(row["processed_window_dtype"]),
        native_target_sha256=_tensor_sha256(target.targets),
        native_target_mask_sha256=_tensor_sha256(
            target.source_target_mask
        ),
        native_bin_states_sha256=_canonical_sha256(target.bin_states),
        observed_label_count=observed,
        positive_label_count=positive,
        negative_label_count=observed - positive,
    )


def _derive_manifest(
    signal_bundle: VerifiedDeepSOZSignalPreflightBundle,
    tusz_root: str | Path,
    *,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
) -> IctalNativeEvalManifest:
    if not isinstance(signal_bundle, VerifiedDeepSOZSignalPreflightBundle):
        raise TypeError("signal_bundle must be a verified DeepSOZ signal preflight")
    if signal_bundle.artifact_sha256 != _require_sha256(
        expected_signal_artifact_sha256, field="expected_signal_artifact_sha256"
    ):
        raise ValueError("Signal-preflight artifact SHA mismatch")
    if signal_bundle.receipt_sha256 != _require_sha256(
        expected_signal_receipt_sha256, field="expected_signal_receipt_sha256"
    ):
        raise ValueError("Signal-preflight receipt SHA mismatch")
    root = _absolute_no_symlink(tusz_root, field="TUSZ root")
    if not root.is_dir():
        raise FileNotFoundError("TUSZ root does not exist")
    source_rows = tuple(
        row
        for row in signal_bundle.receipt["events"]
        if row["model_split"] == "source_dev"
    )
    if not source_rows or any(row["official_split"] != "dev" for row in source_rows):
        raise ValueError("Signal-preflight source-dev events do not map exactly to official dev")
    source_target_roster = _preflight_split_roster(signal_bundle, "source_dev")
    if tuple(sorted({str(row["patient_id"]) for row in source_rows})) != source_target_roster:
        raise ValueError("Source-dev events do not cover the exact preflight patient roster")
    config_payload = dict(signal_bundle.receipt["preprocess_config"])
    config_sha = _preprocess_config_sha256(config_payload)
    if config_sha != signal_bundle.receipt["preprocess_config_sha256"]:
        raise ValueError("Signal-preflight preprocessing config SHA mismatch")
    events = tuple(
        sorted(
            (_manifest_event_from_source(row, tusz_root=root) for row in source_rows),
            key=lambda event: (event.public_patient_id, event.event_id),
        )
    )
    manifest = IctalNativeEvalManifest(
        events=events,
        source_signal_preflight_artifact_sha256=signal_bundle.artifact_sha256,
        source_signal_preflight_receipt_sha256=signal_bundle.receipt_sha256,
        preprocess_config=config_payload,
        preprocess_config_sha256=config_sha,
    )
    if manifest.target_patient_ids != source_target_roster:
        raise ValueError("Derived manifest changed the exact source-dev target roster")
    return manifest


def _manifest_from_payload(payload: Mapping[str, object]) -> IctalNativeEvalManifest:
    _require_exact_fields(payload, _MANIFEST_FIELDS, field="native evaluation receipt")
    raw_events = payload["events"]
    if not isinstance(raw_events, list):
        raise TypeError("native evaluation events must be a list")
    events: list[IctalNativeEvalEvent] = []
    for index, raw in enumerate(raw_events):
        event = _json_object(raw, field=f"events[{index}]")
        _require_exact_fields(event, _EVENT_FIELDS, field=f"events[{index}]")
        normalized = dict(event)
        shape = normalized.pop("processed_window_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("processed_window_shape must be a two-element list")
        events.append(
            IctalNativeEvalEvent(
                **normalized,
                processed_window_shape=tuple(shape),
            )
        )
    manifest = IctalNativeEvalManifest(
        events=tuple(events),
        source_signal_preflight_artifact_sha256=str(
            payload["source_signal_preflight_artifact_sha256"]
        ),
        source_signal_preflight_receipt_sha256=str(
            payload["source_signal_preflight_receipt_sha256"]
        ),
        preprocess_config=_json_object(
            payload["preprocess_config"], field="preprocess_config"
        ),
        preprocess_config_sha256=str(payload["preprocess_config_sha256"]),
        schema_version=str(payload["schema_version"]),
        purpose=str(payload["purpose"]),
        evaluation_only=payload["evaluation_only"],
        training_authorized=payload["training_authorized"],
        contains_soz_labels=payload["contains_soz_labels"],
        model_split=str(payload["model_split"]),
        official_split=str(payload["official_split"]),
        target_semantics=str(payload["target_semantics"]),
    )
    if manifest.canonical_payload != dict(payload):
        raise ValueError("Native evaluation receipt has inconsistent counts or rosters")
    return manifest


def build_ictal_native_eval_manifest(
    signal_bundle: VerifiedDeepSOZSignalPreflightBundle,
    tusz_root: str | Path,
    output_directory: str | Path,
    *,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
) -> VerifiedIctalNativeEvalManifestArtifact:
    """Atomically publish the exact source-dev native-target manifest."""

    target = _safe_new_directory(output_directory, field="native evaluation manifest output")
    manifest = _derive_manifest(
        signal_bundle,
        tusz_root,
        expected_signal_artifact_sha256=expected_signal_artifact_sha256,
        expected_signal_receipt_sha256=expected_signal_receipt_sha256,
    )
    payload = {
        "schema_version": ICTAL_NATIVE_EVAL_MANIFEST_ARTIFACT_SCHEMA,
        "serialization": ICTAL_NATIVE_EVAL_SERIALIZATION,
        "receipt_sha256": manifest.receipt_sha256,
        "receipt": manifest.canonical_payload,
    }
    encoded = _canonical_json_bytes(payload, newline=True)
    if not 1 <= len(encoded) <= _MAX_MANIFEST_BYTES:
        raise ValueError("Native evaluation manifest artifact has an invalid size")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        artifact_path = staging / ICTAL_NATIVE_EVAL_MANIFEST_FILENAME
        with artifact_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        if os.path.lexists(target):
            raise FileExistsError("Native evaluation manifest destination already exists")
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return load_ictal_native_eval_manifest(
        target,
        signal_bundle,
        tusz_root,
        expected_artifact_sha256=hashlib.sha256(encoded).hexdigest(),
        expected_receipt_sha256=manifest.receipt_sha256,
        expected_signal_artifact_sha256=expected_signal_artifact_sha256,
        expected_signal_receipt_sha256=expected_signal_receipt_sha256,
    )


def load_ictal_native_eval_manifest(
    bundle_directory: str | Path,
    signal_bundle: VerifiedDeepSOZSignalPreflightBundle,
    tusz_root: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
) -> VerifiedIctalNativeEvalManifestArtifact:
    """Strictly load and independently replay every native annotation target."""

    directory = _absolute_no_symlink(bundle_directory, field="native evaluation manifest")
    if not directory.is_dir():
        raise FileNotFoundError("Native evaluation manifest directory does not exist")
    entries = tuple(directory.iterdir())
    if len(entries) != 1 or entries[0].name != ICTAL_NATIVE_EVAL_MANIFEST_FILENAME:
        raise ValueError("Native evaluation manifest violates its closed file schema")
    raw, artifact_sha = _stable_file(
        entries[0], field="native evaluation manifest", max_bytes=_MAX_MANIFEST_BYTES
    )
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Native evaluation manifest artifact SHA mismatch")
    artifact_payload = _strict_json(raw, field="native evaluation manifest")
    _require_exact_fields(
        artifact_payload,
        _MANIFEST_ARTIFACT_FIELDS,
        field="native evaluation manifest artifact",
    )
    if (
        artifact_payload["schema_version"]
        != ICTAL_NATIVE_EVAL_MANIFEST_ARTIFACT_SCHEMA
        or artifact_payload["serialization"] != ICTAL_NATIVE_EVAL_SERIALIZATION
    ):
        raise ValueError("Unsupported native evaluation manifest artifact policy")
    manifest = _manifest_from_payload(
        _json_object(artifact_payload["receipt"], field="receipt")
    )
    declared = _require_sha256(
        artifact_payload["receipt_sha256"], field="receipt_sha256"
    )
    if declared != manifest.receipt_sha256 or declared != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Native evaluation manifest receipt SHA mismatch")
    rebuilt = _derive_manifest(
        signal_bundle,
        tusz_root,
        expected_signal_artifact_sha256=expected_signal_artifact_sha256,
        expected_signal_receipt_sha256=expected_signal_receipt_sha256,
    )
    if _canonical_json_bytes(manifest.canonical_payload) != _canonical_json_bytes(
        rebuilt.canonical_payload
    ):
        raise ValueError("Native evaluation manifest disagrees with complete annotation replay")
    return _issue_manifest_artifact(
        path=directory,
        artifact_sha256=artifact_sha,
        manifest=manifest,
    )


@dataclass(frozen=True)
class IctalNativeEvalTokenEventBinding:
    event_id: str
    public_patient_id: str
    evaluation_event_record_sha256: str
    signal_event_record_sha256: str
    bundle_path: Path
    bundle_manifest_sha256: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not _SAFE_EVENT_RE.fullmatch(self.event_id):
            raise ValueError("Evaluation token event_id is invalid")
        if not _PUBLIC_PATIENT_RE.fullmatch(self.public_patient_id):
            raise ValueError("Evaluation token public_patient_id is invalid")
        if not isinstance(self.bundle_path, Path) or not self.bundle_path.is_absolute():
            raise ValueError("Evaluation token bundle_path must be absolute")
        for field in (
            "evaluation_event_record_sha256",
            "signal_event_record_sha256",
            "bundle_manifest_sha256",
            "tensor_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)


@dataclass(frozen=True, init=False)
class VerifiedIctalNativeEvalTokenCorpusArtifact:
    """Opaque evaluation-only token-corpus attestation."""

    path: Path
    index_sha256: str
    manifest_artifact_sha256: str
    manifest_receipt_sha256: str
    signal_preflight_artifact_sha256: str
    signal_preflight_receipt_sha256: str
    foundation_feature_receipt_sha256: str
    foundation_checkpoint_sha256: str
    foundation_modeling_sha256: str
    event_roster_sha256: str
    patient_roster_sha256: str
    patient_event_roster_sha256: str
    tensor_roster_sha256: str
    event_count: int
    patient_count: int
    events: tuple[IctalNativeEvalTokenEventBinding, ...]
    purpose: str
    evaluation_only: bool
    training_authorized: bool

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        index_sha256: str,
        manifest_artifact_sha256: str,
        manifest_receipt_sha256: str,
        signal_preflight_artifact_sha256: str,
        signal_preflight_receipt_sha256: str,
        foundation_feature_receipt_sha256: str,
        foundation_checkpoint_sha256: str,
        foundation_modeling_sha256: str,
        event_roster_sha256: str,
        patient_roster_sha256: str,
        patient_event_roster_sha256: str,
        tensor_roster_sha256: str,
        event_count: int,
        patient_count: int,
        events: Sequence[IctalNativeEvalTokenEventBinding],
    ) -> None:
        if _verification_marker is not _VERIFIED_CORPUS_MARKER:
            raise TypeError("Verified native-evaluation corpora come only from strict loaders")
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("path must be absolute")
        for field, value in (
            ("index_sha256", index_sha256),
            ("manifest_artifact_sha256", manifest_artifact_sha256),
            ("manifest_receipt_sha256", manifest_receipt_sha256),
            ("signal_preflight_artifact_sha256", signal_preflight_artifact_sha256),
            ("signal_preflight_receipt_sha256", signal_preflight_receipt_sha256),
            ("foundation_feature_receipt_sha256", foundation_feature_receipt_sha256),
            ("foundation_checkpoint_sha256", foundation_checkpoint_sha256),
            ("foundation_modeling_sha256", foundation_modeling_sha256),
            ("event_roster_sha256", event_roster_sha256),
            ("patient_roster_sha256", patient_roster_sha256),
            ("patient_event_roster_sha256", patient_event_roster_sha256),
            ("tensor_roster_sha256", tensor_roster_sha256),
        ):
            _require_sha256(value, field=field)
        normalized_events = tuple(events)
        if (
            isinstance(event_count, bool)
            or not isinstance(event_count, int)
            or event_count < 1
            or len(normalized_events) != event_count
            or any(not isinstance(event, IctalNativeEvalTokenEventBinding) for event in normalized_events)
        ):
            raise ValueError("events do not match event_count")
        if (
            isinstance(patient_count, bool)
            or not isinstance(patient_count, int)
            or patient_count < 1
            or len({event.public_patient_id for event in normalized_events}) != patient_count
        ):
            raise ValueError("events do not match patient_count")
        values = locals().copy()
        values.pop("self")
        values.pop("_verification_marker")
        values["events"] = normalized_events
        for field, value in values.items():
            object.__setattr__(self, field, value)
        object.__setattr__(self, "purpose", ICTAL_NATIVE_EVAL_PURPOSE)
        object.__setattr__(self, "evaluation_only", True)
        object.__setattr__(self, "training_authorized", False)


def _issue_corpus_artifact(**kwargs: object) -> VerifiedIctalNativeEvalTokenCorpusArtifact:
    return VerifiedIctalNativeEvalTokenCorpusArtifact(
        _verification_marker=_VERIFIED_CORPUS_MARKER, **kwargs
    )


def replay_ictal_native_eval_signal(
    event: IctalNativeEvalEvent,
    manifest: IctalNativeEvalManifest,
    tusz_root: str | Path,
    *,
    reader_factory: Callable[[str], object] | None = None,
    expected_foundation_receipt: LaBraMFeatureReceipt | None = None,
) -> torch.Tensor:
    """Replay one causal source-dev signal and verify every preflight binding."""

    if not isinstance(event, IctalNativeEvalEvent) or not isinstance(
        manifest, IctalNativeEvalManifest
    ):
        raise TypeError("event and manifest must be native-evaluation types")
    if event not in manifest.events:
        raise ValueError("Event is not part of the supplied native-evaluation manifest")
    root = _absolute_no_symlink(tusz_root, field="TUSZ root")
    _, edf = _safe_relative(root, event.relative_edf_path, field="relative_edf_path")
    config = CausalEDFConfig(**dict(manifest.preprocess_config))
    loaded = load_standard19_edf_event(
        edf,
        event.global_t0_sec,
        config=config,
        reader_factory=reader_factory,
    )
    checks = {
        "EDF SHA": loaded.edf_receipt.edf_sha256 == event.edf_sha256,
        "EDF receipt": _canonical_sha256(asdict(loaded.edf_receipt))
        == event.edf_receipt_sha256,
        "signal receipt": _canonical_sha256(asdict(loaded.signal_receipt))
        == event.signal_receipt_sha256,
        "processed window": _signal_tensor_sha256(loaded.window.data)
        == event.processed_window_sha256,
        "processed shape": tuple(loaded.window.data.shape)
        == event.processed_window_shape,
        "processed dtype": str(loaded.window.data.dtype)
        == event.processed_window_dtype,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Source-dev signal replay failed bindings: {failed}")
    if expected_foundation_receipt is not None:
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        require_feature_receipt_position_binding(
            expected_foundation_receipt,
            binding,
        )
    signal = loaded.window.data.detach().to(dtype=torch.float32, device="cpu")
    if tuple(signal.shape) != (19, 12_000) or not torch.isfinite(signal).all().item():
        raise ValueError("Replayed evaluation signal must be finite [19,12000]")
    return signal


def _validate_foundation(
    encoder: torch.nn.Module,
    *,
    expected_feature_receipt_sha256: str,
    expected_modeling_sha256: str,
) -> tuple[LaBraMFeatureReceipt, str]:
    receipt = getattr(encoder, "receipt", None)
    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("Foundation encoder lacks LaBraMFeatureReceipt")
    if receipt.checkpoint_sha256 != AUDITED_LABRAM_BASE_SHA256:
        raise ValueError("Foundation is not the audited LaBraM-Base checkpoint")
    expected_modeling = _require_sha256(
        expected_modeling_sha256, field="expected_modeling_sha256"
    )
    if (
        expected_modeling != AUDITED_LABRAM_MODELING_SHA256
        or receipt.modeling_sha256 != expected_modeling
    ):
        raise ValueError("Foundation modeling source SHA mismatch")
    receipt_sha = labram_feature_receipt_sha256(receipt)
    if receipt_sha != _require_sha256(
        expected_feature_receipt_sha256,
        field="expected_foundation_feature_receipt_sha256",
    ):
        raise ValueError("Foundation feature receipt SHA mismatch")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Evaluation token extraction requires a frozen foundation")
    encoder.eval()
    return receipt, receipt_sha


def _corpus_rosters(
    events: Sequence[Mapping[str, object]],
) -> tuple[tuple[object, ...], tuple[str, ...], tuple[object, ...]]:
    event_roster = tuple(
        (
            event["event_id"],
            event["public_patient_id"],
            event["evaluation_event_record_sha256"],
            event["signal_event_record_sha256"],
        )
        for event in events
    )
    patients = tuple(sorted({str(event["public_patient_id"]) for event in events}))
    patient_events = tuple(
        (
            patient,
            tuple(
                sorted(
                    str(event["event_id"])
                    for event in events
                    if event["public_patient_id"] == patient
                )
            ),
        )
        for patient in patients
    )
    return event_roster, patients, patient_events


def _build_corpus_index(
    *,
    manifest_artifact: VerifiedIctalNativeEvalManifestArtifact,
    foundation_receipt: LaBraMFeatureReceipt,
    foundation_receipt_sha256: str,
    materialization_device: torch.device,
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    event_roster, patients, patient_events = _corpus_rosters(events)
    manifest = manifest_artifact.manifest
    return {
        "schema_version": ICTAL_NATIVE_EVAL_TOKEN_CORPUS_SCHEMA,
        "purpose": ICTAL_NATIVE_EVAL_PURPOSE,
        "evaluation_only": True,
        "training_authorized": False,
        "contains_soz_labels": False,
        "serialization": "canonical_json_and_safe_event_bundles",
        "manifest": {
            "artifact_sha256": manifest_artifact.artifact_sha256,
            "receipt_sha256": manifest_artifact.receipt_sha256,
            "event_count": len(manifest),
            "patient_count": len(manifest.patient_ids),
        },
        "signal_preflight": {
            "artifact_sha256": manifest.source_signal_preflight_artifact_sha256,
            "receipt_sha256": manifest.source_signal_preflight_receipt_sha256,
            "model_split": "source_dev",
            "official_split": "dev",
        },
        "foundation": {
            "feature_receipt_sha256": foundation_receipt_sha256,
            "checkpoint_sha256": foundation_receipt.checkpoint_sha256,
            "audited_expected_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "modeling_sha256": foundation_receipt.modeling_sha256,
            "audited_expected_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "token_shape": list(CONCEPT_TOKEN_SHAPE),
            "tile_seconds": 4,
            "frozen": True,
            "materialization_device": str(materialization_device),
        },
        "event_count": len(events),
        "patient_count": len(patients),
        "event_roster_sha256": _canonical_sha256(event_roster),
        "patient_roster_sha256": _canonical_sha256(patients),
        "patient_event_roster_sha256": _canonical_sha256(patient_events),
        "tensor_roster_sha256": _canonical_sha256(
            tuple((event["event_id"], event["tensor_sha256"]) for event in events)
        ),
        "events": list(events),
    }


def _validate_corpus_index(payload: Mapping[str, object]) -> dict[str, object]:
    _require_exact_fields(payload, _CORPUS_INDEX_FIELDS, field="evaluation token index")
    if (
        payload["schema_version"] != ICTAL_NATIVE_EVAL_TOKEN_CORPUS_SCHEMA
        or payload["purpose"] != ICTAL_NATIVE_EVAL_PURPOSE
        or payload["evaluation_only"] is not True
        or payload["training_authorized"] is not False
        or payload["contains_soz_labels"] is not False
        or payload["serialization"] != "canonical_json_and_safe_event_bundles"
    ):
        raise ValueError("Evaluation token index policy is immutable")
    manifest = _json_object(payload["manifest"], field="manifest")
    signal = _json_object(payload["signal_preflight"], field="signal_preflight")
    foundation = _json_object(payload["foundation"], field="foundation")
    _require_exact_fields(manifest, _CORPUS_MANIFEST_FIELDS, field="manifest")
    _require_exact_fields(signal, _CORPUS_SIGNAL_FIELDS, field="signal_preflight")
    _require_exact_fields(foundation, _CORPUS_FOUNDATION_FIELDS, field="foundation")
    for block_name, block, fields in (
        ("manifest", manifest, ("artifact_sha256", "receipt_sha256")),
        ("signal_preflight", signal, ("artifact_sha256", "receipt_sha256")),
        (
            "foundation",
            foundation,
            (
                "feature_receipt_sha256",
                "checkpoint_sha256",
                "audited_expected_checkpoint_sha256",
                "modeling_sha256",
                "audited_expected_modeling_sha256",
            ),
        ),
    ):
        for field in fields:
            _require_sha256(block[field], field=f"{block_name}.{field}")
    if signal["model_split"] != "source_dev" or signal["official_split"] != "dev":
        raise ValueError("Evaluation token corpus must be source_dev/dev only")
    if (
        foundation["checkpoint_sha256"] != AUDITED_LABRAM_BASE_SHA256
        or foundation["audited_expected_checkpoint_sha256"]
        != AUDITED_LABRAM_BASE_SHA256
        or foundation["modeling_sha256"] != AUDITED_LABRAM_MODELING_SHA256
        or foundation["audited_expected_modeling_sha256"]
        != AUDITED_LABRAM_MODELING_SHA256
        or foundation["token_shape"] != list(CONCEPT_TOKEN_SHAPE)
        or foundation["tile_seconds"] != 4
        or foundation["frozen"] is not True
        or foundation["materialization_device"] not in {"cpu", "cuda"}
    ):
        raise ValueError("Evaluation token foundation binding is invalid")
    event_count = payload["event_count"]
    patient_count = payload["patient_count"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 1
        or isinstance(patient_count, bool)
        or not isinstance(patient_count, int)
        or patient_count < 1
        or manifest["event_count"] != event_count
        or manifest["patient_count"] != patient_count
    ):
        raise ValueError("Evaluation token corpus counts are invalid")
    raw_events = payload["events"]
    if not isinstance(raw_events, list) or len(raw_events) != event_count:
        raise ValueError("Evaluation token events do not match event_count")
    events: list[dict[str, object]] = []
    for index, raw in enumerate(raw_events):
        event = _json_object(raw, field=f"events[{index}]")
        _require_exact_fields(event, _CORPUS_EVENT_FIELDS, field=f"events[{index}]")
        if not isinstance(event["event_id"], str) or not _SAFE_EVENT_RE.fullmatch(event["event_id"]):
            raise ValueError("Evaluation token event_id is invalid")
        if not isinstance(event["public_patient_id"], str) or not _PUBLIC_PATIENT_RE.fullmatch(
            event["public_patient_id"]
        ):
            raise ValueError("Evaluation token patient ID is invalid")
        if event["bundle_path"] != (
            f"{ICTAL_NATIVE_EVAL_EVENTS_DIRECTORY}/{event['event_id']}"
        ):
            raise ValueError("Evaluation token bundle path is not canonical")
        for field in (
            "evaluation_event_record_sha256",
            "signal_event_record_sha256",
            "bundle_manifest_sha256",
            "tensor_sha256",
        ):
            _require_sha256(event[field], field=f"events[{index}].{field}")
        events.append(dict(event))
    if tuple(events) != tuple(
        sorted(events, key=lambda row: (row["public_patient_id"], row["event_id"]))
    ):
        raise ValueError("Evaluation token events are not canonically ordered")
    if len({event["event_id"] for event in events}) != len(events):
        raise ValueError("Evaluation token event IDs are not unique")
    event_roster, patients, patient_events = _corpus_rosters(events)
    if len(patients) != patient_count:
        raise ValueError("Evaluation token patient count does not match roster")
    expected_hashes = {
        "event_roster_sha256": _canonical_sha256(event_roster),
        "patient_roster_sha256": _canonical_sha256(patients),
        "patient_event_roster_sha256": _canonical_sha256(patient_events),
        "tensor_roster_sha256": _canonical_sha256(
            tuple((event["event_id"], event["tensor_sha256"]) for event in events)
        ),
    }
    for field, expected in expected_hashes.items():
        if payload[field] != expected:
            raise ValueError(f"Evaluation token {field} mismatch")
    normalized = dict(payload)
    normalized["manifest"] = dict(manifest)
    normalized["signal_preflight"] = dict(signal)
    normalized["foundation"] = dict(foundation)
    normalized["events"] = events
    return normalized


def load_ictal_native_eval_token_corpus(
    corpus_directory: str | Path,
    manifest_artifact: VerifiedIctalNativeEvalManifestArtifact,
    *,
    expected_index_sha256: str,
    expected_manifest_artifact_sha256: str,
    expected_manifest_receipt_sha256: str,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
) -> VerifiedIctalNativeEvalTokenCorpusArtifact:
    """Strictly reload the evaluation-only index and every token bundle."""

    if not isinstance(manifest_artifact, VerifiedIctalNativeEvalManifestArtifact):
        raise TypeError("manifest_artifact must come from the strict loader")
    expected_manifest_artifact = _require_sha256(
        expected_manifest_artifact_sha256,
        field="expected_manifest_artifact_sha256",
    )
    expected_manifest_receipt = _require_sha256(
        expected_manifest_receipt_sha256,
        field="expected_manifest_receipt_sha256",
    )
    if (
        manifest_artifact.artifact_sha256 != expected_manifest_artifact
        or manifest_artifact.receipt_sha256 != expected_manifest_receipt
    ):
        raise ValueError("Native evaluation manifest external SHA mismatch")
    source = _absolute_no_symlink(corpus_directory, field="evaluation token corpus")
    if not source.is_dir():
        raise FileNotFoundError("Evaluation token corpus directory does not exist")
    if {entry.name for entry in source.iterdir()} != {
        ICTAL_NATIVE_EVAL_TOKEN_INDEX_FILENAME,
        ICTAL_NATIVE_EVAL_EVENTS_DIRECTORY,
    }:
        raise ValueError("Evaluation token corpus violates its closed file schema")
    event_root = source / ICTAL_NATIVE_EVAL_EVENTS_DIRECTORY
    if event_root.is_symlink() or not event_root.is_dir():
        raise ValueError("Evaluation token events must be a regular directory")
    raw, index_sha = _stable_file(
        source / ICTAL_NATIVE_EVAL_TOKEN_INDEX_FILENAME,
        field="evaluation token index",
        max_bytes=_MAX_INDEX_BYTES,
    )
    if index_sha != _require_sha256(
        expected_index_sha256, field="expected_index_sha256"
    ):
        raise ValueError("Evaluation token index SHA mismatch")
    index = _validate_corpus_index(_strict_json(raw, field="evaluation token index"))
    manifest_binding = index["manifest"]
    signal_binding = index["signal_preflight"]
    if (
        manifest_binding["artifact_sha256"] != expected_manifest_artifact
        or manifest_binding["receipt_sha256"] != expected_manifest_receipt
        or signal_binding["artifact_sha256"]
        != _require_sha256(
            expected_signal_artifact_sha256,
            field="expected_signal_artifact_sha256",
        )
        or signal_binding["receipt_sha256"]
        != _require_sha256(
            expected_signal_receipt_sha256,
            field="expected_signal_receipt_sha256",
        )
        or signal_binding["artifact_sha256"]
        != manifest_artifact.manifest.source_signal_preflight_artifact_sha256
        or signal_binding["receipt_sha256"]
        != manifest_artifact.manifest.source_signal_preflight_receipt_sha256
    ):
        raise ValueError("Evaluation token corpus manifest/signal binding mismatch")
    manifest_events = {event.event_id: event for event in manifest_artifact.manifest}
    if set(manifest_events) != {event["event_id"] for event in index["events"]}:
        raise ValueError("Evaluation token corpus changed the manifest event roster")
    expected_directories = {str(event["event_id"]) for event in index["events"]}
    if {entry.name for entry in event_root.iterdir()} != expected_directories:
        raise ValueError("Evaluation token event directories do not match the index")
    foundation = index["foundation"]
    bindings: list[IctalNativeEvalTokenEventBinding] = []
    for event in index["events"]:
        source_event = manifest_events[str(event["event_id"])]
        checks = {
            "public patient": event["public_patient_id"]
            == source_event.public_patient_id,
            "evaluation event": event["evaluation_event_record_sha256"]
            == source_event.event_record_sha256,
            "signal event": event["signal_event_record_sha256"]
            == source_event.signal_event_record_sha256,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Evaluation token event changed manifest fields: {failed}")
        bundle = source / str(event["bundle_path"])
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError("Evaluation token bundle must be a regular directory")
        token = load_labram_concept_tokens(
            bundle,
            expected_manifest_sha256=str(event["bundle_manifest_sha256"]),
        )
        token_checks = {
            "event_id": token.event_id == event["event_id"],
            "source_manifest": token.source_concept_manifest_sha256
            == expected_manifest_receipt,
            "event_record": token.event_record_sha256
            == event["evaluation_event_record_sha256"],
            "preprocess": token.preprocess_receipt_sha256
            == source_event.signal_receipt_sha256,
            "foundation_receipt": token.foundation_feature_receipt_sha256
            == foundation["feature_receipt_sha256"],
            "foundation_checkpoint": token.foundation_checkpoint_sha256
            == foundation["checkpoint_sha256"],
            "foundation_modeling": token.foundation_feature_receipt.modeling_sha256
            == foundation["modeling_sha256"],
            "tensor": token.tensor_sha256 == event["tensor_sha256"],
        }
        failed = tuple(name for name, passed in token_checks.items() if not passed)
        if failed:
            raise ValueError(f"Evaluation token bundle failed fields: {failed}")
        bindings.append(
            IctalNativeEvalTokenEventBinding(
                event_id=str(event["event_id"]),
                public_patient_id=str(event["public_patient_id"]),
                evaluation_event_record_sha256=str(
                    event["evaluation_event_record_sha256"]
                ),
                signal_event_record_sha256=str(event["signal_event_record_sha256"]),
                bundle_path=bundle,
                bundle_manifest_sha256=str(event["bundle_manifest_sha256"]),
                tensor_sha256=str(event["tensor_sha256"]),
            )
        )
    return _issue_corpus_artifact(
        path=source,
        index_sha256=index_sha,
        manifest_artifact_sha256=expected_manifest_artifact,
        manifest_receipt_sha256=expected_manifest_receipt,
        signal_preflight_artifact_sha256=str(signal_binding["artifact_sha256"]),
        signal_preflight_receipt_sha256=str(signal_binding["receipt_sha256"]),
        foundation_feature_receipt_sha256=str(
            foundation["feature_receipt_sha256"]
        ),
        foundation_checkpoint_sha256=str(foundation["checkpoint_sha256"]),
        foundation_modeling_sha256=str(foundation["modeling_sha256"]),
        event_roster_sha256=str(index["event_roster_sha256"]),
        patient_roster_sha256=str(index["patient_roster_sha256"]),
        patient_event_roster_sha256=str(index["patient_event_roster_sha256"]),
        tensor_roster_sha256=str(index["tensor_roster_sha256"]),
        event_count=int(index["event_count"]),
        patient_count=int(index["patient_count"]),
        events=tuple(bindings),
    )


def materialize_ictal_native_eval_token_corpus(
    *,
    manifest_artifact: VerifiedIctalNativeEvalManifestArtifact,
    expected_manifest_artifact_sha256: str,
    expected_manifest_receipt_sha256: str,
    expected_signal_artifact_sha256: str,
    expected_signal_receipt_sha256: str,
    tusz_root: str | Path,
    labram_modeling_path: str | Path,
    labram_checkpoint_path: str | Path,
    expected_labram_modeling_sha256: str,
    expected_foundation_feature_receipt_sha256: str,
    output_directory: str | Path,
    device: str | torch.device = "cuda",
    reader_factory: Callable[[str], object] | None = None,
) -> VerifiedIctalNativeEvalTokenCorpusArtifact:
    """Atomically materialize frozen LaBraM tokens for source-dev evaluation."""

    if not isinstance(manifest_artifact, VerifiedIctalNativeEvalManifestArtifact):
        raise TypeError("manifest_artifact must come from the strict loader")
    if manifest_artifact.artifact_sha256 != _require_sha256(
        expected_manifest_artifact_sha256,
        field="expected_manifest_artifact_sha256",
    ) or manifest_artifact.receipt_sha256 != _require_sha256(
        expected_manifest_receipt_sha256,
        field="expected_manifest_receipt_sha256",
    ):
        raise ValueError("Native evaluation manifest external SHA mismatch")
    manifest = manifest_artifact.manifest
    if manifest.source_signal_preflight_artifact_sha256 != _require_sha256(
        expected_signal_artifact_sha256, field="expected_signal_artifact_sha256"
    ) or manifest.source_signal_preflight_receipt_sha256 != _require_sha256(
        expected_signal_receipt_sha256, field="expected_signal_receipt_sha256"
    ):
        raise ValueError("Native evaluation signal-preflight SHA mismatch")
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"} or execution_device.index is not None:
        raise ValueError("Evaluation token materialization supports cpu or cuda")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    target = _safe_new_directory(output_directory, field="evaluation token corpus output")
    encoder = OfficialLaBraMEncoder(
        modeling_path=labram_modeling_path,
        checkpoint_path=labram_checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=expected_labram_modeling_sha256,
        tile_seconds=4,
    )
    foundation_receipt, foundation_receipt_sha = _validate_foundation(
        encoder,
        expected_feature_receipt_sha256=expected_foundation_feature_receipt_sha256,
        expected_modeling_sha256=expected_labram_modeling_sha256,
    )
    encoder.to(execution_device).eval()
    tiled = TiledFoundationEncoder(encoder, n_calls=15).to(execution_device).eval()
    if any(parameter.requires_grad for parameter in tiled.parameters()):
        raise ValueError("Tiled evaluation foundation unexpectedly became trainable")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        event_root = staging / ICTAL_NATIVE_EVAL_EVENTS_DIRECTORY
        event_root.mkdir()
        rows: list[dict[str, object]] = []
        for event in manifest:
            eeg = replay_ictal_native_eval_signal(
                event,
                manifest,
                tusz_root,
                reader_factory=reader_factory,
                expected_foundation_receipt=foundation_receipt,
            )
            with torch.inference_mode():
                tokens = tiled(
                    eeg.unsqueeze(0).to(device=execution_device, dtype=torch.float32)
                )[0].detach().to(dtype=torch.float32, device="cpu")
            if tuple(tokens.shape) != CONCEPT_TOKEN_SHAPE:
                raise ValueError("Foundation returned the wrong evaluation token shape")
            bundle = event_root / event.event_id
            artifact = save_labram_concept_tokens(
                bundle,
                tokens,
                event_id=event.event_id,
                source_concept_manifest_sha256=manifest_artifact.receipt_sha256,
                event_record_sha256=event.event_record_sha256,
                preprocess_receipt_sha256=event.signal_receipt_sha256,
                foundation_feature_receipt=foundation_receipt,
            )
            loaded = load_labram_concept_tokens(
                artifact.path,
                expected_manifest_sha256=artifact.manifest_sha256,
            )
            if (
                loaded.event_id != event.event_id
                or loaded.source_concept_manifest_sha256
                != manifest_artifact.receipt_sha256
                or loaded.event_record_sha256 != event.event_record_sha256
                or loaded.preprocess_receipt_sha256 != event.signal_receipt_sha256
                or loaded.foundation_feature_receipt_sha256 != foundation_receipt_sha
                or loaded.tensor_sha256 != artifact.tensor_sha256
            ):
                raise ValueError("Generated evaluation token failed lineage replay")
            rows.append(
                {
                    "event_id": event.event_id,
                    "public_patient_id": event.public_patient_id,
                    "evaluation_event_record_sha256": event.event_record_sha256,
                    "signal_event_record_sha256": event.signal_event_record_sha256,
                    "bundle_path": (
                        f"{ICTAL_NATIVE_EVAL_EVENTS_DIRECTORY}/{event.event_id}"
                    ),
                    "bundle_manifest_sha256": artifact.manifest_sha256,
                    "tensor_sha256": artifact.tensor_sha256,
                }
            )
            del eeg, tokens
        index = _validate_corpus_index(
            _build_corpus_index(
                manifest_artifact=manifest_artifact,
                foundation_receipt=foundation_receipt,
                foundation_receipt_sha256=foundation_receipt_sha,
                materialization_device=execution_device,
                events=rows,
            )
        )
        encoded = _canonical_json_bytes(index, newline=True)
        if not 1 <= len(encoded) <= _MAX_INDEX_BYTES:
            raise ValueError("Evaluation token index has an invalid size")
        index_path = staging / ICTAL_NATIVE_EVAL_TOKEN_INDEX_FILENAME
        index_path.write_bytes(encoded)
        _fsync_file(index_path)
        _fsync_directory(event_root)
        _fsync_directory(staging)
        index_sha = hashlib.sha256(encoded).hexdigest()
        load_ictal_native_eval_token_corpus(
            staging,
            manifest_artifact,
            expected_index_sha256=index_sha,
            expected_manifest_artifact_sha256=expected_manifest_artifact_sha256,
            expected_manifest_receipt_sha256=expected_manifest_receipt_sha256,
            expected_signal_artifact_sha256=expected_signal_artifact_sha256,
            expected_signal_receipt_sha256=expected_signal_receipt_sha256,
        )
        if os.path.lexists(target):
            raise FileExistsError("Evaluation token corpus destination already exists")
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)
        return load_ictal_native_eval_token_corpus(
            target,
            manifest_artifact,
            expected_index_sha256=index_sha,
            expected_manifest_artifact_sha256=expected_manifest_artifact_sha256,
            expected_manifest_receipt_sha256=expected_manifest_receipt_sha256,
            expected_signal_artifact_sha256=expected_signal_artifact_sha256,
            expected_signal_receipt_sha256=expected_signal_receipt_sha256,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _load_replayed_native_target(
    event: IctalNativeEvalEvent, tusz_root: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    _, edf = _safe_relative(tusz_root, event.relative_edf_path, field="relative_edf_path")
    _, channel = _safe_relative(
        tusz_root,
        event.relative_channel_annotation_path,
        field="relative_channel_annotation_path",
    )
    _, global_path = _safe_relative(
        tusz_root,
        event.relative_global_annotation_path,
        field="relative_global_annotation_path",
    )
    target = load_tusz_ictal_involvement_target(
        channel,
        global_path,
        event_index=event.global_event_index,
        source_path=edf,
    )
    observed = int(target.source_target_mask.sum().item())
    positive = int(
        target.targets[target.source_target_mask].sum().item()
    )
    checks = {
        "target": _tensor_sha256(target.targets) == event.native_target_sha256,
        "mask": (
            _tensor_sha256(target.source_target_mask)
            == event.native_target_mask_sha256
        ),
        "bin states": _canonical_sha256(target.bin_states)
        == event.native_bin_states_sha256,
        "observed count": observed == event.observed_label_count,
        "positive count": positive == event.positive_label_count,
        "negative count": observed - positive == event.negative_label_count,
        "EDF": target.receipt.source_sha256 == event.edf_sha256,
        "annotation pair": target.receipt.annotation_pair_sha256
        == event.annotation_pair_sha256,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Native evaluation target replay failed fields: {failed}")
    return (
        target.targets.detach().clone(),
        target.source_target_mask.detach().clone(),
    )


def build_ictal_native_eval_token_bag_dataset(
    manifest_artifact: VerifiedIctalNativeEvalManifestArtifact,
    tusz_root: str | Path,
    corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
) -> IctalTokenBagDataset:
    """Join verified evaluation tokens to replayed native targets.

    The returned dataset and all its bags are explicitly unauthorized for
    optimization.  Evaluation APIs can consume it unchanged.
    """

    if not isinstance(manifest_artifact, VerifiedIctalNativeEvalManifestArtifact):
        raise TypeError("manifest_artifact must come from the strict loader")
    if not isinstance(corpus, VerifiedIctalNativeEvalTokenCorpusArtifact):
        raise TypeError("corpus must be a verified native-evaluation corpus")
    if (
        corpus.manifest_artifact_sha256 != manifest_artifact.artifact_sha256
        or corpus.manifest_receipt_sha256 != manifest_artifact.receipt_sha256
        or corpus.signal_preflight_artifact_sha256
        != manifest_artifact.manifest.source_signal_preflight_artifact_sha256
        or corpus.signal_preflight_receipt_sha256
        != manifest_artifact.manifest.source_signal_preflight_receipt_sha256
    ):
        raise ValueError("Evaluation corpus is bound to a different manifest/signal bundle")
    manifest = manifest_artifact.manifest
    if len(manifest) != corpus.event_count or len(manifest.patient_ids) != corpus.patient_count:
        raise ValueError("Evaluation corpus counts do not match the manifest")
    expected_event_roster = tuple(
        (
            event.event_id,
            event.public_patient_id,
            event.event_record_sha256,
            event.signal_event_record_sha256,
        )
        for event in manifest
    )
    expected_patient_events = tuple(
        (
            patient,
            tuple(event.event_id for event in manifest.events_for_patient(patient)),
        )
        for patient in manifest.patient_ids
    )
    checks = {
        "event roster": _canonical_sha256(expected_event_roster)
        == corpus.event_roster_sha256,
        "patient roster": _canonical_sha256(manifest.patient_ids)
        == corpus.patient_roster_sha256,
        "patient-event roster": _canonical_sha256(expected_patient_events)
        == corpus.patient_event_roster_sha256,
        "event IDs": tuple(event.event_id for event in manifest)
        == tuple(binding.event_id for binding in corpus.events),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Evaluation corpus/manifest roster mismatch: {failed}")
    root = _absolute_no_symlink(tusz_root, field="TUSZ root")
    if not root.is_dir():
        raise FileNotFoundError("TUSZ root does not exist")
    bindings = {binding.event_id: binding for binding in corpus.events}

    def load_patient(patient_id: str) -> IctalTokenPatientBag:
        patient_events = manifest.events_for_patient(patient_id)
        tokens: list[LoadedLaBraMConceptTokens] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for event in patient_events:
            binding = bindings[event.event_id]
            token = load_labram_concept_tokens(
                binding.bundle_path,
                expected_manifest_sha256=binding.bundle_manifest_sha256,
            )
            token_checks = {
                "event": token.event_id == event.event_id,
                "manifest": token.source_concept_manifest_sha256
                == manifest_artifact.receipt_sha256,
                "event record": token.event_record_sha256
                == event.event_record_sha256,
                "preprocess": token.preprocess_receipt_sha256
                == event.signal_receipt_sha256,
                "foundation": token.foundation_feature_receipt_sha256
                == corpus.foundation_feature_receipt_sha256,
                "checkpoint": token.foundation_checkpoint_sha256
                == corpus.foundation_checkpoint_sha256,
                "tensor": token.tensor_sha256 == binding.tensor_sha256,
            }
            failed = tuple(name for name, passed in token_checks.items() if not passed)
            if failed:
                raise ValueError(f"Evaluation patient token failed fields: {failed}")
            target, mask = _load_replayed_native_target(event, root)
            tokens.append(token)
            targets.append(target)
            masks.append(mask)
        event_ids = tuple(event.event_id for event in patient_events)
        return IctalTokenPatientBag(
            patient_id=patient_id,
            event_ids=event_ids,
            expected_event_ids=event_ids,
            training_manifest_sha256=manifest_artifact.receipt_sha256,
            expected_event_record_sha256s=tuple(
                event.event_record_sha256 for event in patient_events
            ),
            token_events=tuple(tokens),
            targets=torch.stack(targets),
            target_mask=torch.stack(masks),
            training_authorized=False,
        )

    return IctalTokenBagDataset(
        manifest.patient_ids,
        load_patient,
        training_manifest_sha256=manifest_artifact.receipt_sha256,
        token_source_manifest_sha256=manifest_artifact.receipt_sha256,
        foundation_feature_receipt_sha256=corpus.foundation_feature_receipt_sha256,
        formal_token_corpus_verified=False,
        training_authorized=False,
    )


__all__ = [
    "ICTAL_NATIVE_EVAL_MANIFEST_ARTIFACT_SCHEMA",
    "ICTAL_NATIVE_EVAL_MANIFEST_FILENAME",
    "ICTAL_NATIVE_EVAL_MANIFEST_SCHEMA",
    "ICTAL_NATIVE_EVAL_PURPOSE",
    "ICTAL_NATIVE_EVAL_TARGET_SEMANTICS",
    "ICTAL_NATIVE_EVAL_TOKEN_CORPUS_SCHEMA",
    "IctalNativeEvalEvent",
    "IctalNativeEvalManifest",
    "IctalNativeEvalTokenEventBinding",
    "VerifiedIctalNativeEvalManifestArtifact",
    "VerifiedIctalNativeEvalTokenCorpusArtifact",
    "build_ictal_native_eval_manifest",
    "build_ictal_native_eval_token_bag_dataset",
    "load_bound_deepsoz_signal_preflight_artifact",
    "load_ictal_native_eval_manifest",
    "load_ictal_native_eval_token_corpus",
    "materialize_ictal_native_eval_token_corpus",
    "replay_ictal_native_eval_signal",
]
