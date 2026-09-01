"""Strict source-train bridge to frozen, node-indexed LaBraM tokens.

This module does not create a new EEG representation and does not open SOZ
targets.  It binds the already-authorized 65-patient/582-event development
evidence roster to the independently materialized formal-v4 TUSZ LaBraM token
corpus.  The bridge is derived from source identity and timing fields, never
from a string replacement between the two event-ID namespaces.

The published artifact is deliberately lazy: it contains only a closed,
ordered binding receipt.  Token values remain in their original, strictly
validated bundles and are loaded as ``[19, 15, 4, 200]`` only on request.
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
from typing import Callable, Mapping, Sequence

import torch

from .concept_token_io import (
    CONCEPT_TOKEN_SHAPE,
    load_labram_concept_tokens,
)
from .data.deepsoz_signal_preflight import VerifiedDeepSOZSignalPreflightBundle
from .data.edf import CausalEDFConfig, load_standard19_edf_event
from .data.tusz_training import (
    TUSZIctalEventRecord,
    TUSZIctalTrainingManifest,
    tusz_signal_preflight_receipt_sha256,
)
from .development_reasoner_v1_1 import (
    PublishedDevelopmentIVEvidenceCapabilityV11,
)
from .formal_token_corpus import (
    VerifiedFormalTokenCorpusArtifact,
)
from .geometry import N_STANDARD_CHANNELS
from .ictal_recovery_evidence import TargetFreeOOFProtocolView
from .models.labram import (
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    bind_labram_record_positions,
    require_feature_receipt_position_binding,
)
from .source_train_iv_capability import PublishedSourceTrainIVCapability


FROZEN_H_CROSSWALK_SCHEMA = "soz_labram_frozen_h_source_train_crosswalk_v1"
FROZEN_H_CROSSWALK_ARTIFACT_SCHEMA = (
    "soz_labram_frozen_h_source_train_crosswalk_artifact_v1"
)
FROZEN_H_CROSSWALK_PURPOSE = (
    "development_only_source_train_frozen_labram_node_indexed_tokens"
)
FROZEN_H_CROSSWALK_SERIALIZATION = "canonical_json_utf8_no_pickle"
FROZEN_H_RECEIPT_FILENAME = "receipt.json"
FROZEN_H_MANIFEST_FILENAME = "manifest.json"
FROZEN_H_TOKEN_SHAPE = (N_STANDARD_CHANNELS, 15, 4, 200)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TIME_TOLERANCE_SEC = 1e-6
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_VERIFIED_MARKER = object()

_EVENT_FIELDS = frozenset(
    {
        "ordinal",
        "evidence_event_id",
        "target_patient_id",
        "public_patient_id",
        "oof_fold",
        "token_event_id",
        "token_bundle_relative_path",
        "token_bundle_manifest_sha256",
        "token_tensor_sha256",
        "token_event_record_sha256",
        "token_preprocess_receipt_sha256",
        "relative_edf_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "seizure_type",
        "edf_sha256",
        "channel_annotation_sha256",
        "global_annotation_sha256",
        "annotation_pair_sha256",
        "deepsoz_event_record_sha256",
        "preprocess_config_sha256",
        "processed_window_sha256",
        "processed_window_shape",
        "processed_window_dtype",
        "deepsoz_edf_receipt_sha256",
        "deepsoz_signal_receipt_sha256",
        "tusz_signal_preflight_receipt_sha256",
        "labram_position_binding_policy",
        "labram_position_names",
        "labram_position_ids",
        "raw_replay_sha256",
    }
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "development_only",
        "model_split",
        "lazy_token_binding",
        "raw_eeg_serialized",
        "foundation_token_values_serialized",
        "deepsoz_target_values_loaded",
        "source_train_evidence_values_used",
        "tusz_involvement_target_values_loaded",
        "source_dev_signal_loaded",
        "source_dev_token_loaded",
        "source_dev_target_loaded",
        "source_eval_used",
        "private_used",
        "formal_promotion",
        "candidate_input_authorized",
        "source_train_capability_manifest_sha256",
        "source_train_authorization_receipt_sha256",
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "master_manifest_bundle_sha256",
        "master_manifest_source_sha256",
        "formal_token_corpus_index_sha256",
        "formal_token_corpus_tensor_roster_sha256",
        "preprocessing_selection_artifact_sha256",
        "preprocessing_protocol_receipt_sha256",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "foundation_modeling_sha256",
        "foundation_position_binding_policy",
        "cached_token_event_shape",
        "frozen_h_event_shape",
        "reshape_policy",
        "event_count",
        "patient_count",
        "event_order_sha256",
        "patient_roster_sha256",
        "token_binding_roster_sha256",
        "raw_replay_roster_sha256",
        "raw_replay_verified",
        "events",
    }
)

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "serialization",
        "development_only",
        "model_split",
        "lazy_token_binding",
        "deepsoz_target_values_loaded",
        "source_dev_used",
        "source_eval_used",
        "private_used",
        "formal_promotion",
        "receipt_file",
        "receipt_sha256",
        "receipt_size_bytes",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Frozen-H artifact contains non-canonical JSON data") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


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
            f"missing={sorted(set(expected)-actual)}, "
            f"unknown={sorted(actual-set(expected))}"
        )


def _json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object")
    return dict(value)


def _strict_json(raw: bytes, *, field: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field} contains forbidden constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    result = _json_object(value, field=field)
    if _canonical_json_bytes(result) != raw:
        raise ValueError(f"{field} is not canonical JSON")
    return result


def _signal_tensor_sha256(tensor: torch.Tensor) -> str:
    """Match the DeepSOZ processed-window receipt exactly."""

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


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _safe_source_file(root: Path, relative_value: object) -> Path:
    if not isinstance(relative_value, str) or not relative_value or "\\" in relative_value:
        raise ValueError("relative_edf_path must be a canonical relative path")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("relative_edf_path is not canonical")
    source = _absolute_no_symlink(root.joinpath(*relative.parts), field="TUSZ EDF")
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError("relative_edf_path escapes the TUSZ root") from exc
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    return source


def _stable_file(path: Path, *, field: str, maximum: int) -> tuple[bytes, str]:
    source = _absolute_no_symlink(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular file")
    before = source.stat()
    if not 1 <= before.st_size <= maximum:
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


def _safe_new_directory(path: str | Path) -> Path:
    target = _absolute_no_symlink(path, field="Frozen-H output")
    if target.name in {"", ".", ".."}:
        raise ValueError("Frozen-H output requires a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    return target


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _guard_output_topology(
    target: Path,
    *,
    tusz_root: Path,
    token_corpus_root: Path,
    capability_root: Path,
) -> None:
    """Keep publication outside every immutable source tree."""

    inputs = {
        "TUSZ EDF root": tusz_root,
        "formal token corpus": token_corpus_root,
        "source-train capability": capability_root,
    }
    overlaps = tuple(
        name for name, source in inputs.items() if _paths_overlap(target, source)
    )
    if overlaps:
        raise ValueError(
            "Frozen-H output topology overlaps immutable inputs: "
            + ", ".join(overlaps)
        )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _close_time(left: object, right: object) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= _TIME_TOLERANCE_SEC


def _match_master_event(
    signal_event: Mapping[str, object],
    master_by_source_identity: Mapping[tuple[str, int], TUSZIctalEventRecord],
    *,
    expected_public_patient_id: str,
) -> TUSZIctalEventRecord:
    """Resolve by source path/index and verify all independent identity fields."""

    path = str(signal_event["relative_edf_path"])
    index = signal_event["global_event_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("DeepSOZ signal event has an invalid global index")
    try:
        event = master_by_source_identity[(path, index)]
    except KeyError as exc:
        raise ValueError(
            "DeepSOZ event has no unique formal-v4 TUSZ source identity"
        ) from exc
    checks = {
        "public patient": event.patient_id == expected_public_patient_id,
        "relative EDF": event.relative_edf_path == path,
        "global index": event.event_index == index,
        "global t0": _close_time(event.event_t0_sec, signal_event["global_t0_sec"]),
        "global stop": _close_time(
            event.event_stop_sec, signal_event["global_stop_sec"]
        ),
        "seizure type": event.seizure_type
        == str(signal_event["global_seizure_type"]),
        "EDF SHA": event.edf_sha256 == signal_event["edf_sha256"],
        "channel annotation": event.channel_annotation_sha256
        == signal_event["channel_annotation_sha256"],
        "global annotation": event.global_annotation_sha256
        == signal_event["global_annotation_sha256"],
        "annotation pair": event.annotation_pair_sha256
        == signal_event["annotation_pair_sha256"],
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "DeepSOZ/formal-v4 source identity mismatch: " + ", ".join(failed)
        )
    return event


def _verify_raw_replay(
    *,
    signal_event: Mapping[str, object],
    master_event: TUSZIctalEventRecord,
    loaded: object,
) -> None:
    """Bridge the DeepSOZ and TUSZ preprocessing receipt namespaces."""

    edf_receipt = getattr(loaded, "edf_receipt", None)
    signal_receipt = getattr(loaded, "signal_receipt", None)
    window = getattr(loaded, "window", None)
    if edf_receipt is None or signal_receipt is None or window is None:
        raise TypeError("Raw replay must return a complete LoadedEDFEvent")
    checks = {
        "EDF SHA": edf_receipt.edf_sha256 == signal_event["edf_sha256"],
        "processed window": _signal_tensor_sha256(window.data)
        == signal_event["processed_window_sha256"],
        "processed shape": list(window.data.shape)
        == list(signal_event["processed_window_shape"]),
        "processed dtype": str(window.data.dtype)
        == signal_event["processed_window_dtype"],
        "DeepSOZ EDF receipt": _canonical_sha256(asdict(edf_receipt))
        == signal_event["edf_receipt_sha256"],
        "DeepSOZ signal receipt": _canonical_sha256(asdict(signal_receipt))
        == signal_event["signal_receipt_sha256"],
        "TUSZ signal receipt": tusz_signal_preflight_receipt_sha256(loaded)
        == master_event.signal_preflight_receipt_sha256,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Raw signal replay failed for {signal_event['event_id']}: {failed}"
        )


def _raw_replay_sha256(row: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            "evidence_event_id": row["evidence_event_id"],
            "token_event_id": row["token_event_id"],
            "relative_edf_path": row["relative_edf_path"],
            "global_event_index": row["global_event_index"],
            "global_t0_sec": row["global_t0_sec"],
            "global_stop_sec": row["global_stop_sec"],
            "edf_sha256": row["edf_sha256"],
            "annotation_pair_sha256": row["annotation_pair_sha256"],
            "processed_window_sha256": row["processed_window_sha256"],
            "deepsoz_edf_receipt_sha256": row[
                "deepsoz_edf_receipt_sha256"
            ],
            "deepsoz_signal_receipt_sha256": row[
                "deepsoz_signal_receipt_sha256"
            ],
            "tusz_signal_preflight_receipt_sha256": row[
                "tusz_signal_preflight_receipt_sha256"
            ],
            "token_tensor_sha256": row["token_tensor_sha256"],
        }
    )


def _validate_event_row(value: object, *, ordinal: int) -> dict[str, object]:
    row = _json_object(value, field=f"events[{ordinal}]")
    _require_exact_fields(row, _EVENT_FIELDS, field=f"events[{ordinal}]")
    if row["ordinal"] != ordinal:
        raise ValueError("Frozen-H event ordinal/order changed")
    for field in (
        "evidence_event_id",
        "target_patient_id",
        "public_patient_id",
        "token_event_id",
        "token_bundle_relative_path",
        "relative_edf_path",
        "seizure_type",
        "processed_window_dtype",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"events[{ordinal}].{field} must be non-empty")
    for field in (
        "token_bundle_manifest_sha256",
        "token_tensor_sha256",
        "token_event_record_sha256",
        "token_preprocess_receipt_sha256",
        "edf_sha256",
        "channel_annotation_sha256",
        "global_annotation_sha256",
        "annotation_pair_sha256",
        "deepsoz_event_record_sha256",
        "preprocess_config_sha256",
        "processed_window_sha256",
        "deepsoz_edf_receipt_sha256",
        "deepsoz_signal_receipt_sha256",
        "tusz_signal_preflight_receipt_sha256",
        "raw_replay_sha256",
    ):
        _require_sha256(row[field], field=f"events[{ordinal}].{field}")
    if row["token_preprocess_receipt_sha256"] != row[
        "tusz_signal_preflight_receipt_sha256"
    ]:
        raise ValueError("Token and TUSZ preprocessing receipts differ")
    if row["labram_position_binding_policy"] != (
        LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
    ):
        raise ValueError("Frozen-H event uses another position-binding policy")
    names = row["labram_position_names"]
    ids = row["labram_position_ids"]
    if (
        not isinstance(names, list)
        or len(names) != N_STANDARD_CHANNELS
        or any(not isinstance(item, str) or not item for item in names)
        or len(set(names)) != N_STANDARD_CHANNELS
        or not isinstance(ids, list)
        or len(ids) != N_STANDARD_CHANNELS
        or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
        or any(item < 0 for item in ids)
        or len(set(ids)) != N_STANDARD_CHANNELS
    ):
        raise ValueError(
            "Frozen-H position binding must contain 19 unique names and IDs"
        )
    if row["processed_window_shape"] != [19, 12_000]:
        raise ValueError("Frozen-H source window must have shape [19,12000]")
    if row["processed_window_dtype"] not in {"torch.float32", "torch.float64"}:
        raise ValueError("Frozen-H source window has an unsupported dtype")
    for field in ("oof_fold", "global_event_index"):
        item = row[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"events[{ordinal}].{field} is invalid")
    if row["oof_fold"] not in range(5):
        raise ValueError("Frozen-H source-train event requires OOF fold 0..4")
    if not _close_time(row["global_t0_sec"], row["global_t0_sec"]) or not (
        _close_time(row["global_stop_sec"], row["global_stop_sec"])
        and float(row["global_stop_sec"]) > float(row["global_t0_sec"])
    ):
        raise ValueError("Frozen-H event interval is invalid")
    expected_bundle = f"events/{row['token_event_id']}"
    if row["token_bundle_relative_path"] != expected_bundle:
        raise ValueError("Frozen-H token bundle path is not canonical")
    if row["raw_replay_sha256"] != _raw_replay_sha256(row):
        raise ValueError("Frozen-H raw replay receipt changed")
    return row


def _validate_receipt(value: object) -> dict[str, object]:
    receipt = _json_object(value, field="Frozen-H receipt")
    _require_exact_fields(receipt, _RECEIPT_FIELDS, field="Frozen-H receipt")
    fixed = {
        "schema_version": FROZEN_H_CROSSWALK_SCHEMA,
        "purpose": FROZEN_H_CROSSWALK_PURPOSE,
        "development_only": True,
        "model_split": "source_train",
        "lazy_token_binding": True,
        "raw_eeg_serialized": False,
        "foundation_token_values_serialized": False,
        "deepsoz_target_values_loaded": False,
        "source_train_evidence_values_used": False,
        "tusz_involvement_target_values_loaded": False,
        "source_dev_signal_loaded": False,
        "source_dev_token_loaded": False,
        "source_dev_target_loaded": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
        "candidate_input_authorized": True,
        "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
        "foundation_position_binding_policy": (
            LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
        ),
        "cached_token_event_shape": list(CONCEPT_TOKEN_SHAPE),
        "frozen_h_event_shape": list(FROZEN_H_TOKEN_SHAPE),
        "reshape_policy": "channel_major_60_tokens_to_15_calls_x_4_slots_v1",
        "raw_replay_verified": True,
    }
    changed = tuple(name for name, expected in fixed.items() if receipt[name] != expected)
    if changed:
        raise ValueError(f"Frozen-H scientific boundary changed: {changed}")
    for field in (
        "source_train_capability_manifest_sha256",
        "source_train_authorization_receipt_sha256",
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "master_manifest_bundle_sha256",
        "master_manifest_source_sha256",
        "formal_token_corpus_index_sha256",
        "formal_token_corpus_tensor_roster_sha256",
        "preprocessing_selection_artifact_sha256",
        "preprocessing_protocol_receipt_sha256",
        "foundation_feature_receipt_sha256",
        "event_order_sha256",
        "patient_roster_sha256",
        "token_binding_roster_sha256",
        "raw_replay_roster_sha256",
    ):
        _require_sha256(receipt[field], field=field)
    event_count = receipt["event_count"]
    patient_count = receipt["patient_count"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 1
        or isinstance(patient_count, bool)
        or not isinstance(patient_count, int)
        or patient_count < 1
    ):
        raise ValueError("Frozen-H counts must be positive integers")
    values = receipt["events"]
    if not isinstance(values, list) or len(values) != event_count:
        raise ValueError("Frozen-H event list does not match event_count")
    events = [_validate_event_row(row, ordinal=i) for i, row in enumerate(values)]
    evidence_ids = tuple(str(row["evidence_event_id"]) for row in events)
    token_ids = tuple(str(row["token_event_id"]) for row in events)
    if len(set(evidence_ids)) != event_count or len(set(token_ids)) != event_count:
        raise ValueError("Frozen-H event bindings must be one-to-one")
    patients = tuple(sorted({str(row["target_patient_id"]) for row in events}))
    if len(patients) != patient_count:
        raise ValueError("Frozen-H patient_count disagrees with event rows")
    expected_hashes = {
        "event_order_sha256": _canonical_sha256(evidence_ids),
        "patient_roster_sha256": _canonical_sha256(patients),
        "token_binding_roster_sha256": _canonical_sha256(
            tuple(
                (
                    row["evidence_event_id"],
                    row["token_event_id"],
                    row["token_bundle_manifest_sha256"],
                    row["token_tensor_sha256"],
                )
                for row in events
            )
        ),
        "raw_replay_roster_sha256": _canonical_sha256(
            tuple(
                (row["evidence_event_id"], row["raw_replay_sha256"])
                for row in events
            )
        ),
    }
    for field, expected in expected_hashes.items():
        if receipt[field] != expected:
            raise ValueError(f"Frozen-H {field} disagrees with event rows")
    receipt["events"] = events
    return receipt


@dataclass(frozen=True)
class FrozenHEventBinding:
    ordinal: int
    evidence_event_id: str
    target_patient_id: str
    public_patient_id: str
    oof_fold: int
    token_event_id: str
    bundle_path: Path
    bundle_manifest_sha256: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        if self.ordinal < 0 or self.oof_fold not in range(5):
            raise ValueError("Frozen-H event ordinal/fold is invalid")
        if not isinstance(self.bundle_path, Path) or not self.bundle_path.is_absolute():
            raise ValueError("Frozen-H bundle path must be absolute")
        _require_sha256(self.bundle_manifest_sha256, field="bundle_manifest_sha256")
        _require_sha256(self.tensor_sha256, field="tensor_sha256")


@dataclass(frozen=True, init=False)
class VerifiedFrozenHSourceTrainArtifact:
    """Opaque development-only lazy access to ordered frozen-H tokens."""

    path: Path
    manifest_sha256: str
    receipt_sha256: str
    receipt: Mapping[str, object]
    events: tuple[FrozenHEventBinding, ...]

    def __init__(
        self,
        *,
        _marker: object,
        path: Path,
        manifest_sha256: str,
        receipt_sha256: str,
        receipt: Mapping[str, object],
        events: Sequence[FrozenHEventBinding],
    ) -> None:
        if _marker is not _VERIFIED_MARKER:
            raise TypeError("Verified Frozen-H artifacts require the strict issuer")
        validated = _validate_receipt(dict(receipt))
        _require_sha256(manifest_sha256, field="manifest_sha256")
        expected_receipt_sha = _canonical_sha256(validated)
        if receipt_sha256 != expected_receipt_sha:
            raise ValueError("Frozen-H receipt SHA mismatch")
        bindings = tuple(events)
        if len(bindings) != validated["event_count"] or tuple(
            event.ordinal for event in bindings
        ) != tuple(range(len(bindings))):
            raise ValueError("Frozen-H runtime bindings changed order")
        for field, value in {
            "path": path,
            "manifest_sha256": manifest_sha256,
            "receipt_sha256": receipt_sha256,
            "receipt": validated,
            "events": bindings,
        }.items():
            object.__setattr__(self, field, value)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.evidence_event_id for event in self.events)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({event.target_patient_id for event in self.events}))

    @property
    def patient_ids_by_event(self) -> tuple[str, ...]:
        return tuple(event.target_patient_id for event in self.events)

    @property
    def oof_folds(self) -> tuple[int, ...]:
        return tuple(event.oof_fold for event in self.events)

    def assert_unchanged(self) -> None:
        """Reject shallow-frozen receipt or runtime-binding mutation."""

        validated = _validate_receipt(dict(self.receipt))
        if _canonical_sha256(validated) != self.receipt_sha256:
            raise ValueError("Frozen-H verified receipt changed in memory")
        rows = validated["events"]
        if len(self.events) != len(rows):
            raise ValueError("Frozen-H runtime event roster changed in memory")
        for binding, row in zip(self.events, rows):
            checks = {
                "ordinal": binding.ordinal == row["ordinal"],
                "evidence event": binding.evidence_event_id
                == row["evidence_event_id"],
                "target patient": binding.target_patient_id
                == row["target_patient_id"],
                "public patient": binding.public_patient_id
                == row["public_patient_id"],
                "OOF fold": binding.oof_fold == row["oof_fold"],
                "token event": binding.token_event_id == row["token_event_id"],
                "bundle manifest": binding.bundle_manifest_sha256
                == row["token_bundle_manifest_sha256"],
                "tensor": binding.tensor_sha256 == row["token_tensor_sha256"],
            }
            failed = tuple(name for name, passed in checks.items() if not passed)
            if failed:
                raise ValueError(
                    f"Frozen-H runtime binding changed in memory: {failed}"
                )

    def _load_bound_tokens(self, binding: FrozenHEventBinding) -> torch.Tensor:
        token = load_labram_concept_tokens(
            binding.bundle_path,
            expected_manifest_sha256=binding.bundle_manifest_sha256,
        )
        checks = {
            "event": token.event_id == binding.token_event_id,
            "tensor": token.tensor_sha256 == binding.tensor_sha256,
            "foundation receipt": token.foundation_feature_receipt_sha256
            == self.receipt["foundation_feature_receipt_sha256"],
            "checkpoint": token.foundation_checkpoint_sha256
            == self.receipt["foundation_checkpoint_sha256"],
            "modeling": token.foundation_feature_receipt.modeling_sha256
            == self.receipt["foundation_modeling_sha256"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Frozen-H token binding changed: {failed}")
        result = (
            token.tokens.detach().cpu().reshape(FROZEN_H_TOKEN_SHAPE).contiguous()
        )
        if result.requires_grad or result.grad_fn is not None:
            raise RuntimeError("Frozen-H lazy loader returned differentiable tokens")
        return result

    def load_event_tokens(self, index: int) -> torch.Tensor:
        """Load one event as detached ``[19,15,4,200]`` frozen tokens."""

        self.assert_unchanged()
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TypeError("Frozen-H event index must be a non-negative integer")
        try:
            binding = self.events[index]
        except IndexError as exc:
            raise IndexError("Frozen-H event index is outside the artifact") from exc
        return self._load_bound_tokens(binding)

    def materialize_tokens(self) -> torch.Tensor:
        """Load the complete ordered CPU tensor ``[E,19,15,4,200]``.

        This is intentionally explicit rather than cached inside the artifact:
        for the frozen 582-event roster the returned float32 tensor occupies
        about 506 MiB.  Every source bundle is revalidated during the call.
        """

        self.assert_unchanged()
        tokens = torch.stack(
            tuple(self._load_bound_tokens(binding) for binding in self.events), dim=0
        ).contiguous()
        expected = (len(self.events), *FROZEN_H_TOKEN_SHAPE)
        if tuple(tokens.shape) != expected or tokens.device.type != "cpu":
            raise RuntimeError("Frozen-H complete tensor has an invalid shape/device")
        if tokens.requires_grad or tokens.grad_fn is not None:
            raise RuntimeError("Frozen-H complete tensor must remain detached")
        return tokens


def _runtime_bindings(
    receipt: Mapping[str, object],
    token_corpus: VerifiedFormalTokenCorpusArtifact,
) -> tuple[FrozenHEventBinding, ...]:
    token_by_id = {event.event_id: event for event in token_corpus.events}
    result = []
    for raw in receipt["events"]:
        event = dict(raw)
        token = token_by_id.get(str(event["token_event_id"]))
        if token is None:
            raise ValueError("Frozen-H receipt points outside the verified token corpus")
        checks = {
            "bundle path": token.bundle_path.relative_to(token_corpus.path).as_posix()
            == event["token_bundle_relative_path"],
            "bundle manifest": token.bundle_manifest_sha256
            == event["token_bundle_manifest_sha256"],
            "tensor": token.tensor_sha256 == event["token_tensor_sha256"],
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Frozen-H runtime token binding changed: {failed}")
        result.append(
            FrozenHEventBinding(
                ordinal=int(event["ordinal"]),
                evidence_event_id=str(event["evidence_event_id"]),
                target_patient_id=str(event["target_patient_id"]),
                public_patient_id=str(event["public_patient_id"]),
                oof_fold=int(event["oof_fold"]),
                token_event_id=str(event["token_event_id"]),
                bundle_path=token.bundle_path,
                bundle_manifest_sha256=token.bundle_manifest_sha256,
                tensor_sha256=token.tensor_sha256,
            )
        )
    return tuple(result)


def _validate_upstream_types(
    capability: PublishedDevelopmentIVEvidenceCapabilityV11,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    master_manifest: TUSZIctalTrainingManifest,
    token_corpus: VerifiedFormalTokenCorpusArtifact,
) -> None:
    if type(capability) is not PublishedDevelopmentIVEvidenceCapabilityV11:
        raise TypeError("Frozen-H requires the strict v1.1 evidence capability")
    if type(signal) is not VerifiedDeepSOZSignalPreflightBundle:
        raise TypeError("Frozen-H requires the strict signal-preflight capability")
    if type(protocol) is not TargetFreeOOFProtocolView:
        raise TypeError("Frozen-H requires the strict target-free OOF protocol")
    if type(master_manifest) is not TUSZIctalTrainingManifest:
        raise TypeError("Frozen-H requires the strict TUSZ master manifest")
    if type(token_corpus) is not VerifiedFormalTokenCorpusArtifact:
        raise TypeError("Frozen-H requires the strict formal token corpus")
    capability.capability.assert_unchanged()
    protocol.assert_unchanged()


def _validate_source_only_upstream_types(
    capability: PublishedSourceTrainIVCapability,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    master_manifest: TUSZIctalTrainingManifest,
    token_corpus: VerifiedFormalTokenCorpusArtifact,
) -> None:
    if type(capability) is not PublishedSourceTrainIVCapability:
        raise TypeError(
            "Frozen-H source-only replay requires the strict source-train I/V "
            "capability"
        )
    if type(signal) is not VerifiedDeepSOZSignalPreflightBundle:
        raise TypeError("Frozen-H requires the strict signal-preflight capability")
    if type(protocol) is not TargetFreeOOFProtocolView:
        raise TypeError("Frozen-H requires the strict target-free OOF protocol")
    if type(master_manifest) is not TUSZIctalTrainingManifest:
        raise TypeError("Frozen-H requires the strict TUSZ master manifest")
    if type(token_corpus) is not VerifiedFormalTokenCorpusArtifact:
        raise TypeError("Frozen-H requires the strict formal token corpus")
    capability.assert_unchanged()
    protocol.assert_unchanged()


def _derive_receipt_for_source(
    *,
    source: object,
    source_train_capability_manifest_sha256: str,
    source_train_authorization_receipt_sha256: str,
    expected_signal_preflight_artifact_sha256: str,
    expected_signal_preflight_receipt_sha256: str,
    expected_oof_protocol_artifact_sha256: str,
    expected_oof_protocol_receipt_sha256: str,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    master_manifest: TUSZIctalTrainingManifest,
    token_corpus: VerifiedFormalTokenCorpusArtifact,
    tusz_root: Path | None,
    reader_factory: Callable[[str], object] | None,
) -> dict[str, object]:
    lineage_checks = {
        "signal artifact": expected_signal_preflight_artifact_sha256
        == signal.artifact_sha256,
        "signal receipt": expected_signal_preflight_receipt_sha256
        == signal.receipt_sha256,
        "OOF artifact": expected_oof_protocol_artifact_sha256
        == protocol.artifact_sha256,
        "OOF receipt": expected_oof_protocol_receipt_sha256
        == protocol.receipt_sha256,
        "master manifest": token_corpus.training_source_manifest_sha256
        == master_manifest.manifest_sha256,
        "token master source": token_corpus.master_source_manifest_sha256
        == master_manifest.manifest_sha256,
        "token corpus is master": token_corpus.training_bundle_manifest_sha256
        == token_corpus.master_bundle_manifest_sha256,
        "master role": master_manifest.derived_from_manifest_sha256 is None,
        "master preflight": master_manifest.preflight_performed is True,
    }
    failed = tuple(name for name, passed in lineage_checks.items() if not passed)
    if failed:
        raise ValueError(f"Frozen-H upstream lineage mismatch: {failed}")
    config = CausalEDFConfig(**dict(signal.receipt["preprocess_config"]))
    if config != master_manifest.preprocess_config:
        raise ValueError("DeepSOZ and formal-v4 preprocessing configs differ")

    if source.model_split != "source_train":
        raise ValueError("Frozen-H rejects non-source-train evidence")
    if len(source.event_ids) != len(set(source.event_ids)):
        raise ValueError("Frozen-H evidence event IDs repeat")
    signal_rows = {
        str(row["event_id"]): row
        for row in signal.receipt["events"]
        if row["model_split"] == "source_train"
    }
    source_train_signal_count = sum(
        row["model_split"] == "source_train" for row in signal.receipt["events"]
    )
    if (
        source_train_signal_count != len(signal_rows)
        or set(signal_rows) != set(source.event_ids)
    ):
        raise ValueError("Frozen-H evidence and signal source-train rosters differ")
    master_by_identity: dict[tuple[str, int], TUSZIctalEventRecord] = {}
    for event in master_manifest:
        key = (event.relative_edf_path, event.event_index)
        if key in master_by_identity:
            raise ValueError("Formal-v4 master source identity is not unique")
        master_by_identity[key] = event
    token_by_id = {event.event_id: event for event in token_corpus.events}
    if len(token_by_id) != token_corpus.event_count:
        raise ValueError("Formal token event identity is not unique")

    root = None
    if tusz_root is not None:
        root = _absolute_no_symlink(tusz_root, field="TUSZ root")
        if not root.is_dir():
            raise FileNotFoundError(root)
    public_crosswalk = protocol.crosswalk
    rows: list[dict[str, object]] = []
    foundation_receipt_sha: str | None = None
    for ordinal, (event_id, target_patient_id, fold) in enumerate(
        zip(source.event_ids, source.patient_ids_by_event, source.oof_folds)
    ):
        if fold is None or int(fold) != protocol.fold_for_target(target_patient_id):
            raise ValueError("Frozen-H evidence fold differs from OOF protocol")
        try:
            expected_public = public_crosswalk[target_patient_id]
        except KeyError as exc:
            raise ValueError("Evidence patient lacks a public TUSZ crosswalk") from exc
        signal_event = signal_rows[event_id]
        if signal_event["patient_id"] != target_patient_id:
            raise ValueError("Evidence and signal target patient identities differ")
        master_event = _match_master_event(
            signal_event,
            master_by_identity,
            expected_public_patient_id=expected_public,
        )
        token_binding = token_by_id.get(master_event.event_id)
        if token_binding is None:
            raise ValueError("Matched formal-v4 event lacks a token bundle")
        token = load_labram_concept_tokens(
            token_binding.bundle_path,
            expected_manifest_sha256=token_binding.bundle_manifest_sha256,
        )
        token_checks = {
            "event ID": token.event_id == master_event.event_id,
            "event record": token.event_record_sha256
            == master_event.event_record_sha256,
            "TUSZ preprocess": token.preprocess_receipt_sha256
            == master_event.signal_preflight_receipt_sha256,
            "tensor": token.tensor_sha256 == token_binding.tensor_sha256,
            "checkpoint": token.foundation_checkpoint_sha256
            == AUDITED_LABRAM_BASE_SHA256,
            "modeling": token.foundation_feature_receipt.modeling_sha256
            == AUDITED_LABRAM_MODELING_SHA256,
        }
        failed = tuple(name for name, passed in token_checks.items() if not passed)
        if failed:
            raise ValueError(f"Frozen-H token lineage mismatch: {failed}")
        current_foundation_sha = token.foundation_feature_receipt_sha256
        if foundation_receipt_sha is None:
            foundation_receipt_sha = current_foundation_sha
        elif foundation_receipt_sha != current_foundation_sha:
            raise ValueError("Frozen-H events use different foundation receipts")
        edf_receipt = _json_object(
            signal_event["edf_receipt"], field="signal event EDF receipt"
        )
        binding = bind_labram_record_positions(
            edf_receipt["raw_channel_names"],
            semantic_channels=edf_receipt["semantic_channels"],
        )
        require_feature_receipt_position_binding(
            token.foundation_feature_receipt, binding
        )
        if root is not None:
            loaded = load_standard19_edf_event(
                _safe_source_file(root, master_event.relative_edf_path),
                master_event.event_t0_sec,
                config=config,
                reader_factory=reader_factory,
            )
            _verify_raw_replay(
                signal_event=signal_event,
                master_event=master_event,
                loaded=loaded,
            )
        row: dict[str, object] = {
            "ordinal": ordinal,
            "evidence_event_id": event_id,
            "target_patient_id": target_patient_id,
            "public_patient_id": expected_public,
            "oof_fold": int(fold),
            "token_event_id": master_event.event_id,
            "token_bundle_relative_path": token_binding.bundle_path.relative_to(
                token_corpus.path
            ).as_posix(),
            "token_bundle_manifest_sha256": token_binding.bundle_manifest_sha256,
            "token_tensor_sha256": token_binding.tensor_sha256,
            "token_event_record_sha256": master_event.event_record_sha256,
            "token_preprocess_receipt_sha256": token.preprocess_receipt_sha256,
            "relative_edf_path": master_event.relative_edf_path,
            "global_event_index": master_event.event_index,
            "global_t0_sec": master_event.event_t0_sec,
            "global_stop_sec": master_event.event_stop_sec,
            "seizure_type": master_event.seizure_type,
            "edf_sha256": master_event.edf_sha256,
            "channel_annotation_sha256": master_event.channel_annotation_sha256,
            "global_annotation_sha256": master_event.global_annotation_sha256,
            "annotation_pair_sha256": master_event.annotation_pair_sha256,
            "deepsoz_event_record_sha256": signal_event["event_record_sha256"],
            "preprocess_config_sha256": signal_event[
                "preprocess_config_sha256"
            ],
            "processed_window_sha256": signal_event["processed_window_sha256"],
            "processed_window_shape": signal_event["processed_window_shape"],
            "processed_window_dtype": signal_event["processed_window_dtype"],
            "deepsoz_edf_receipt_sha256": signal_event["edf_receipt_sha256"],
            "deepsoz_signal_receipt_sha256": signal_event[
                "signal_receipt_sha256"
            ],
            "tusz_signal_preflight_receipt_sha256": (
                master_event.signal_preflight_receipt_sha256
            ),
            "labram_position_binding_policy": binding.policy,
            "labram_position_names": list(binding.position_names),
            "labram_position_ids": list(binding.position_ids),
        }
        row["raw_replay_sha256"] = _raw_replay_sha256(row)
        rows.append(_validate_event_row(row, ordinal=ordinal))
    if foundation_receipt_sha is None:
        raise ValueError("Frozen-H source-train roster is empty")

    evidence_ids = tuple(str(row["evidence_event_id"]) for row in rows)
    patient_ids = tuple(sorted({str(row["target_patient_id"]) for row in rows}))
    receipt = {
        "schema_version": FROZEN_H_CROSSWALK_SCHEMA,
        "purpose": FROZEN_H_CROSSWALK_PURPOSE,
        "development_only": True,
        "model_split": "source_train",
        "lazy_token_binding": True,
        "raw_eeg_serialized": False,
        "foundation_token_values_serialized": False,
        "deepsoz_target_values_loaded": False,
        "source_train_evidence_values_used": False,
        "tusz_involvement_target_values_loaded": False,
        "source_dev_signal_loaded": False,
        "source_dev_token_loaded": False,
        "source_dev_target_loaded": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
        "candidate_input_authorized": True,
        "source_train_capability_manifest_sha256": (
            source_train_capability_manifest_sha256
        ),
        "source_train_authorization_receipt_sha256": (
            source_train_authorization_receipt_sha256
        ),
        "signal_preflight_artifact_sha256": signal.artifact_sha256,
        "signal_preflight_receipt_sha256": signal.receipt_sha256,
        "oof_protocol_artifact_sha256": protocol.artifact_sha256,
        "oof_protocol_receipt_sha256": protocol.receipt_sha256,
        "master_manifest_bundle_sha256": token_corpus.master_bundle_manifest_sha256,
        "master_manifest_source_sha256": master_manifest.manifest_sha256,
        "formal_token_corpus_index_sha256": token_corpus.index_sha256,
        "formal_token_corpus_tensor_roster_sha256": token_corpus.tensor_roster_sha256,
        "preprocessing_selection_artifact_sha256": (
            token_corpus.preprocessing_selection_artifact_sha256
        ),
        "preprocessing_protocol_receipt_sha256": (
            token_corpus.preprocessing_protocol_receipt_sha256
        ),
        "foundation_feature_receipt_sha256": foundation_receipt_sha,
        "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
        "foundation_position_binding_policy": (
            LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
        ),
        "cached_token_event_shape": list(CONCEPT_TOKEN_SHAPE),
        "frozen_h_event_shape": list(FROZEN_H_TOKEN_SHAPE),
        "reshape_policy": "channel_major_60_tokens_to_15_calls_x_4_slots_v1",
        "event_count": len(rows),
        "patient_count": len(patient_ids),
        "event_order_sha256": _canonical_sha256(evidence_ids),
        "patient_roster_sha256": _canonical_sha256(patient_ids),
        "token_binding_roster_sha256": _canonical_sha256(
            tuple(
                (
                    row["evidence_event_id"],
                    row["token_event_id"],
                    row["token_bundle_manifest_sha256"],
                    row["token_tensor_sha256"],
                )
                for row in rows
            )
        ),
        "raw_replay_roster_sha256": _canonical_sha256(
            tuple(
                (row["evidence_event_id"], row["raw_replay_sha256"])
                for row in rows
            )
        ),
        "raw_replay_verified": True,
        "events": rows,
    }
    return _validate_receipt(receipt)


def _derive_receipt(
    *,
    capability: PublishedDevelopmentIVEvidenceCapabilityV11,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    master_manifest: TUSZIctalTrainingManifest,
    token_corpus: VerifiedFormalTokenCorpusArtifact,
    tusz_root: Path | None,
    reader_factory: Callable[[str], object] | None,
) -> dict[str, object]:
    """Replay the original receipt from the historical shared v1.1 bundle."""

    _validate_upstream_types(
        capability, signal, protocol, master_manifest, token_corpus
    )
    authorization = capability.capability.receipt
    return _derive_receipt_for_source(
        source=capability.capability.base.capability.source_train,
        source_train_capability_manifest_sha256=capability.manifest_sha256,
        source_train_authorization_receipt_sha256=(
            capability.authorization_receipt_sha256
        ),
        expected_signal_preflight_artifact_sha256=(
            authorization.signal_preflight_artifact_sha256
        ),
        expected_signal_preflight_receipt_sha256=(
            authorization.signal_preflight_receipt_sha256
        ),
        expected_oof_protocol_artifact_sha256=(
            authorization.oof_protocol_artifact_sha256
        ),
        expected_oof_protocol_receipt_sha256=(
            authorization.oof_protocol_receipt_sha256
        ),
        signal=signal,
        protocol=protocol,
        master_manifest=master_manifest,
        token_corpus=token_corpus,
        tusz_root=tusz_root,
        reader_factory=reader_factory,
    )


def _derive_receipt_from_source_only(
    *,
    capability: PublishedSourceTrainIVCapability,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    master_manifest: TUSZIctalTrainingManifest,
    token_corpus: VerifiedFormalTokenCorpusArtifact,
) -> dict[str, object]:
    """Replay the frozen receipt without opening the shared v1.1 tensors.

    The historical crosswalk names the parent v1.1 manifest and authorization
    receipt.  The physically isolated child carries those two hashes as closed
    lineage, so the byte-identical receipt can be replayed from source-train
    evidence alone.
    """

    _validate_source_only_upstream_types(
        capability, signal, protocol, master_manifest, token_corpus
    )
    lineage = capability.receipt.lineage
    return _derive_receipt_for_source(
        source=capability.split,
        source_train_capability_manifest_sha256=lineage[
            "parent_v1_1_manifest_sha256"
        ],
        source_train_authorization_receipt_sha256=lineage[
            "parent_v1_1_authorization_receipt_sha256"
        ],
        expected_signal_preflight_artifact_sha256=lineage[
            "signal_preflight_artifact_sha256"
        ],
        expected_signal_preflight_receipt_sha256=lineage[
            "signal_preflight_receipt_sha256"
        ],
        expected_oof_protocol_artifact_sha256=lineage[
            "oof_protocol_artifact_sha256"
        ],
        expected_oof_protocol_receipt_sha256=lineage[
            "oof_protocol_receipt_sha256"
        ],
        signal=signal,
        protocol=protocol,
        master_manifest=master_manifest,
        token_corpus=token_corpus,
        tusz_root=None,
        reader_factory=None,
    )


def _issue_verified(
    *,
    path: Path,
    manifest_sha256: str,
    receipt: Mapping[str, object],
    token_corpus: VerifiedFormalTokenCorpusArtifact,
) -> VerifiedFrozenHSourceTrainArtifact:
    validated = _validate_receipt(dict(receipt))
    return VerifiedFrozenHSourceTrainArtifact(
        _marker=_VERIFIED_MARKER,
        path=path,
        manifest_sha256=manifest_sha256,
        receipt_sha256=_canonical_sha256(validated),
        receipt=validated,
        events=_runtime_bindings(validated, token_corpus),
    )


def materialize_frozen_h_source_train_crosswalk(
    *,
    capability: PublishedDevelopmentIVEvidenceCapabilityV11,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    master_manifest: TUSZIctalTrainingManifest,
    token_corpus: VerifiedFormalTokenCorpusArtifact,
    tusz_root: str | Path,
    output_directory: str | Path,
    reader_factory: Callable[[str], object] | None = None,
) -> VerifiedFrozenHSourceTrainArtifact:
    """Replay all 582 source-train signals and publish a lazy H crosswalk."""

    target = _safe_new_directory(output_directory)
    root = _absolute_no_symlink(tusz_root, field="TUSZ root")
    if not root.is_dir():
        raise FileNotFoundError(root)
    _guard_output_topology(
        target,
        tusz_root=root,
        token_corpus_root=token_corpus.path,
        capability_root=capability.path,
    )
    receipt = _derive_receipt(
        capability=capability,
        signal=signal,
        protocol=protocol,
        master_manifest=master_manifest,
        token_corpus=token_corpus,
        tusz_root=root,
        reader_factory=reader_factory,
    )
    receipt_raw = _canonical_json_bytes(receipt)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    manifest = {
        "schema_version": FROZEN_H_CROSSWALK_ARTIFACT_SCHEMA,
        "purpose": FROZEN_H_CROSSWALK_PURPOSE,
        "serialization": FROZEN_H_CROSSWALK_SERIALIZATION,
        "development_only": True,
        "model_split": "source_train",
        "lazy_token_binding": True,
        "deepsoz_target_values_loaded": False,
        "source_dev_used": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
        "receipt_file": FROZEN_H_RECEIPT_FILENAME,
        "receipt_sha256": receipt_sha,
        "receipt_size_bytes": len(receipt_raw),
    }
    manifest_raw = _canonical_json_bytes(manifest)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / FROZEN_H_RECEIPT_FILENAME).write_bytes(receipt_raw)
        (staging / FROZEN_H_MANIFEST_FILENAME).write_bytes(manifest_raw)
        _fsync_file(staging / FROZEN_H_RECEIPT_FILENAME)
        _fsync_file(staging / FROZEN_H_MANIFEST_FILENAME)
        _fsync_directory(staging)
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return load_frozen_h_source_train_crosswalk(
        target,
        capability=capability,
        signal=signal,
        protocol=protocol,
        master_manifest=master_manifest,
        token_corpus=token_corpus,
        expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        expected_receipt_sha256=receipt_sha,
    )


def load_frozen_h_source_train_crosswalk(
    directory: str | Path,
    *,
    capability: PublishedDevelopmentIVEvidenceCapabilityV11,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    master_manifest: TUSZIctalTrainingManifest,
    token_corpus: VerifiedFormalTokenCorpusArtifact,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedFrozenHSourceTrainArtifact:
    """Strictly reload the closed receipt and replay all upstream bindings."""

    source = _absolute_no_symlink(directory, field="Frozen-H artifact")
    if source.is_symlink() or not source.is_dir() or {
        entry.name for entry in source.iterdir()
    } != {FROZEN_H_MANIFEST_FILENAME, FROZEN_H_RECEIPT_FILENAME}:
        raise ValueError("Frozen-H artifact violates its closed directory schema")
    manifest_raw, manifest_sha = _stable_file(
        source / FROZEN_H_MANIFEST_FILENAME,
        field="Frozen-H manifest",
        maximum=_MAX_MANIFEST_BYTES,
    )
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Frozen-H manifest SHA mismatch")
    manifest = _strict_json(manifest_raw, field="Frozen-H manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, field="Frozen-H manifest")
    fixed = {
        "schema_version": FROZEN_H_CROSSWALK_ARTIFACT_SCHEMA,
        "purpose": FROZEN_H_CROSSWALK_PURPOSE,
        "serialization": FROZEN_H_CROSSWALK_SERIALIZATION,
        "development_only": True,
        "model_split": "source_train",
        "lazy_token_binding": True,
        "deepsoz_target_values_loaded": False,
        "source_dev_used": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
        "receipt_file": FROZEN_H_RECEIPT_FILENAME,
    }
    changed = tuple(name for name, expected in fixed.items() if manifest[name] != expected)
    if changed:
        raise ValueError(f"Frozen-H manifest boundary changed: {changed}")
    receipt_raw, receipt_file_sha = _stable_file(
        source / FROZEN_H_RECEIPT_FILENAME,
        field="Frozen-H receipt",
        maximum=_MAX_RECEIPT_BYTES,
    )
    expected_receipt = _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    )
    if (
        receipt_file_sha != expected_receipt
        or manifest["receipt_sha256"] != expected_receipt
        or manifest["receipt_size_bytes"] != len(receipt_raw)
    ):
        raise ValueError("Frozen-H receipt file binding changed")
    receipt = _validate_receipt(_strict_json(receipt_raw, field="Frozen-H receipt"))
    replay = _derive_receipt(
        capability=capability,
        signal=signal,
        protocol=protocol,
        master_manifest=master_manifest,
        token_corpus=token_corpus,
        tusz_root=None,
        reader_factory=None,
    )
    if _canonical_json_bytes(receipt) != _canonical_json_bytes(replay):
        raise ValueError("Frozen-H receipt does not replay from current upstream artifacts")
    return _issue_verified(
        path=source,
        manifest_sha256=manifest_sha,
        receipt=receipt,
        token_corpus=token_corpus,
    )


def load_frozen_h_source_train_crosswalk_from_source_only(
    directory: str | Path,
    *,
    capability: PublishedSourceTrainIVCapability,
    signal: VerifiedDeepSOZSignalPreflightBundle,
    protocol: TargetFreeOOFProtocolView,
    master_manifest: TUSZIctalTrainingManifest,
    token_corpus: VerifiedFormalTokenCorpusArtifact,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedFrozenHSourceTrainArtifact:
    """Reload Frozen-H using only the physically isolated train capability.

    Unlike :func:`load_frozen_h_source_train_crosswalk`, this entry point never
    accepts or dereferences the shared v1.1 I/V capability.  It strictly
    replays the same immutable crosswalk from the child capability's frozen
    parent lineage and its source-train event/fold roster.
    """

    source = _absolute_no_symlink(directory, field="Frozen-H artifact")
    if source.is_symlink() or not source.is_dir() or {
        entry.name for entry in source.iterdir()
    } != {FROZEN_H_MANIFEST_FILENAME, FROZEN_H_RECEIPT_FILENAME}:
        raise ValueError("Frozen-H artifact violates its closed directory schema")
    manifest_raw, manifest_sha = _stable_file(
        source / FROZEN_H_MANIFEST_FILENAME,
        field="Frozen-H manifest",
        maximum=_MAX_MANIFEST_BYTES,
    )
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Frozen-H manifest SHA mismatch")
    manifest = _strict_json(manifest_raw, field="Frozen-H manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, field="Frozen-H manifest")
    fixed = {
        "schema_version": FROZEN_H_CROSSWALK_ARTIFACT_SCHEMA,
        "purpose": FROZEN_H_CROSSWALK_PURPOSE,
        "serialization": FROZEN_H_CROSSWALK_SERIALIZATION,
        "development_only": True,
        "model_split": "source_train",
        "lazy_token_binding": True,
        "deepsoz_target_values_loaded": False,
        "source_dev_used": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
        "receipt_file": FROZEN_H_RECEIPT_FILENAME,
    }
    changed = tuple(name for name, expected in fixed.items() if manifest[name] != expected)
    if changed:
        raise ValueError(f"Frozen-H manifest boundary changed: {changed}")
    receipt_raw, receipt_file_sha = _stable_file(
        source / FROZEN_H_RECEIPT_FILENAME,
        field="Frozen-H receipt",
        maximum=_MAX_RECEIPT_BYTES,
    )
    expected_receipt = _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    )
    if (
        receipt_file_sha != expected_receipt
        or manifest["receipt_sha256"] != expected_receipt
        or manifest["receipt_size_bytes"] != len(receipt_raw)
    ):
        raise ValueError("Frozen-H receipt file binding changed")
    receipt = _validate_receipt(_strict_json(receipt_raw, field="Frozen-H receipt"))
    replay = _derive_receipt_from_source_only(
        capability=capability,
        signal=signal,
        protocol=protocol,
        master_manifest=master_manifest,
        token_corpus=token_corpus,
    )
    if _canonical_json_bytes(receipt) != _canonical_json_bytes(replay):
        raise ValueError(
            "Frozen-H receipt does not replay from source-train-only upstream "
            "artifacts"
        )
    return _issue_verified(
        path=source,
        manifest_sha256=manifest_sha,
        receipt=receipt,
        token_corpus=token_corpus,
    )


# Stable runner-facing spelling.  The longer original name remains available
# for compatibility with the materialization protocol and its tests.
load_source_train_frozen_h_crosswalk = load_frozen_h_source_train_crosswalk
load_source_train_frozen_h_crosswalk_from_source_only = (
    load_frozen_h_source_train_crosswalk_from_source_only
)


__all__ = [
    "FROZEN_H_CROSSWALK_ARTIFACT_SCHEMA",
    "FROZEN_H_CROSSWALK_PURPOSE",
    "FROZEN_H_CROSSWALK_SCHEMA",
    "FROZEN_H_TOKEN_SHAPE",
    "FrozenHEventBinding",
    "VerifiedFrozenHSourceTrainArtifact",
    "load_frozen_h_source_train_crosswalk",
    "load_frozen_h_source_train_crosswalk_from_source_only",
    "load_source_train_frozen_h_crosswalk",
    "load_source_train_frozen_h_crosswalk_from_source_only",
    "materialize_frozen_h_source_train_crosswalk",
]
