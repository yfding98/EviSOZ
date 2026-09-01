"""Locked target-free LaBraM block-9 prefix cache for ``source_eval``.

The artifact contains only frozen foundation activations derived from raw EEG.
It is intentionally separate from the source-train PEFT cache: no DeepSOZ SOZ
value, TUSZ channel target, annotation path, development prediction, threshold,
or fitted parameter is accepted by this closed schema.

One 60 s event is split into fifteen exact non-overlapping 4 s LaBraM calls.
The detached activation immediately before official transformer block 10 is
stored as ``[15,77,200]`` per event (CLS plus 19 x 4 patch tokens).
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
import stat
import tempfile
from typing import Mapping, Sequence

import torch

from .concept_token_io import labram_feature_receipt_sha256
from .geometry import N_STANDARD_CHANNELS, STANDARD_19
from .models.labram import (
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    LaBraMFeatureReceipt,
    bind_labram_record_positions,
)
from .models.labram_peft import (
    LABRAM_PEFT_BLOCKS,
    LABRAM_PEFT_PREFIX_TOKENS,
    LABRAM_PEFT_SECONDS_PER_CALL,
    LABRAM_PEFT_TOKEN_DIM,
)


LABRAM_SOURCE_EVAL_PREFIX_FULL_SCHEMA = (
    "soz_labram_source_eval_prefix_v1_full"
)
LABRAM_SOURCE_EVAL_PREFIX_SMOKE_SCHEMA = (
    "soz_labram_source_eval_prefix_v1_smoke"
)
LABRAM_SOURCE_EVAL_PREFIX_PURPOSE = (
    "locked_source_eval_target_free_labram_block9_prefix_feature_production"
)
LABRAM_SOURCE_EVAL_PREFIX_SERIALIZATION = (
    "canonical_json_plus_safetensors_no_pickle_v1"
)
LABRAM_SOURCE_EVAL_PREFIX_MANIFEST_FILENAME = "manifest.json"
LABRAM_SOURCE_EVAL_PREFIX_TENSOR_FILENAME = "prefix.safetensors"
LABRAM_SOURCE_EVAL_PREFIX_TENSOR_NAME = "prefix_tokens"
LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE = (
    15,
    LABRAM_PEFT_PREFIX_TOKENS,
    LABRAM_PEFT_TOKEN_DIM,
)

EXPECTED_SOURCE_EVAL_EVENT_COUNT = 185
EXPECTED_SOURCE_EVAL_PATIENT_COUNT = 21
EXPECTED_SOURCE_EVAL_EVENT_ORDER_SHA256 = (
    "e7271d18232732a101b727558cfc5a794c8c43d504151d7a08f7993da834b8d7"
)
EXPECTED_SOURCE_EVAL_PATIENT_ROSTER_SHA256 = (
    "4df31ceaaaaa832b2ca6805a0925afba06b43df996d4819f7dcf3b127da34f70"
)
EXPECTED_SOURCE_EVAL_FIRST_EVENT_ID = "aaaaaaaq_s006_t000__ev0000"
EXPECTED_SOURCE_EVAL_LAST_EVENT_ID = "aaaaatba_s003_t015__ev0001"
EXPECTED_SIGNAL_PREFLIGHT_ARTIFACT_SHA256 = (
    "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
)
EXPECTED_SIGNAL_PREFLIGHT_RECEIPT_SHA256 = (
    "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
)
EXPECTED_PREPROCESS_CONFIG_SHA256 = (
    "f95ee10a3f67b6864ed2a87c7347f60668b07b854f82a22335edef5008f0111b"
)

_FILES = frozenset(
    {
        LABRAM_SOURCE_EVAL_PREFIX_MANIFEST_FILENAME,
        LABRAM_SOURCE_EVAL_PREFIX_TENSOR_FILENAME,
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_TENSOR_BYTES = 1024 * 1024 * 1024

_EVENT_BASE_FIELDS = frozenset(
    {
        "ordinal",
        "event_id",
        "patient_id",
        "local_patient_id",
        "official_split",
        "model_split",
        "relative_edf_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "event_record_sha256",
        "edf_sha256",
        "preprocess_config_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
        "processed_window_shape",
        "processed_window_dtype",
        "labram_position_binding_policy",
        "labram_position_names",
        "labram_position_ids",
        "chunk_reassembly_exact",
    }
)
_EVENT_FIELDS = _EVENT_BASE_FIELDS | frozenset(
    {"input_binding_sha256", "prefix_tensor_sha256"}
)
_LINEAGE_FIELDS = frozenset(
    {
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "locked_source_eval_roster_receipt_sha256",
        "preprocess_config_sha256",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "serialization",
        "model_split",
        "official_split",
        "locked_evaluation",
        "inference_feature_only",
        "scope_kind",
        "full_scope",
        "smoke_only",
        "formal_evaluation_input_authorized",
        "training_authorized",
        "model_selection_authorized",
        "threshold_tuning_authorized",
        "selection_policy",
        "source_roster_event_count",
        "source_roster_patient_count",
        "event_count",
        "patient_count",
        "events",
        "event_order_sha256",
        "patient_roster_sha256",
        "input_binding_roster_sha256",
        "prefix_tensor_roster_sha256",
        "lineage",
        "foundation_backbone",
        "foundation_checkpoint_sha256",
        "foundation_modeling_sha256",
        "foundation_feature_receipt",
        "foundation_feature_receipt_sha256",
        "foundation_prefix_implementation_sha256",
        "foundation_prefix_blocks",
        "foundation_prefix_block_stop_exclusive",
        "foundation_trainable_parameter_count",
        "raw_event_shape",
        "sampling_frequency_hz",
        "event_seconds",
        "call_count_per_event",
        "call_seconds",
        "call_input_shape",
        "call_output_shape",
        "chunk_policy",
        "prefix_cut",
        "tensor_name",
        "tensor_shape",
        "tensor_dtype",
        "tensor_file",
        "tensor_file_sha256",
        "tensor_file_size_bytes",
        "materialization_device",
        "elapsed_sec",
        "seconds_per_event",
        "peak_cuda_memory_bytes",
        "raw_replay_verified",
        "chunk_reassembly_verified",
        "source_train_eeg_loaded",
        "source_train_target_values_loaded",
        "source_dev_eeg_loaded",
        "source_dev_target_values_loaded",
        "source_eval_eeg_loaded",
        "source_eval_event_count_loaded",
        "source_eval_target_values_loaded",
        "deepsoz_target_values_loaded",
        "tusz_channel_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "training_performed",
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
        raise ValueError("LaBraM source-eval prefix contains non-canonical data") from exc


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
        raise ValueError(f"{field} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    if _canonical_json_bytes(value) != raw:
        raise ValueError(f"{field} must be canonical JSON")
    return value


def _absolute_no_symlink(path: str | Path, *, field: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot traverse symlinks")
    return result


def _safe_new_directory(path: str | Path) -> Path:
    target = _absolute_no_symlink(path, field="LaBraM source-eval prefix output")
    if target.name in {"", ".", ".."}:
        raise ValueError("LaBraM source-eval prefix output needs a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise FileNotFoundError(target.parent)
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_manifest(path: Path) -> tuple[bytes, str]:
    source = _absolute_no_symlink(path, field="LaBraM source-eval manifest")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= _MAX_MANIFEST_BYTES:
            raise ValueError("LaBraM source-eval manifest is not a bounded regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError("LaBraM source-eval manifest changed while read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError("LaBraM source-eval manifest read was incomplete")
    return raw, hashlib.sha256(raw).hexdigest()


def _stable_tensor_sha256(path: Path) -> tuple[str, os.stat_result]:
    source = _absolute_no_symlink(path, field="LaBraM source-eval tensor")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= _MAX_TENSOR_BYTES:
            raise ValueError("LaBraM source-eval tensor is not a bounded regular file")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError("LaBraM source-eval tensor changed while hashed")
    return digest.hexdigest(), after


def source_eval_prefix_event_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one detached float32 ``[15,77,200]`` activation."""

    values = tensor.detach().cpu().contiguous()
    if tuple(values.shape) != LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE:
        raise ValueError("LaBraM source-eval prefix event tensor shape changed")
    if values.dtype != torch.float32 or values.requires_grad:
        raise TypeError("LaBraM source-eval prefix tensor must be detached float32")
    metadata = _canonical_json_bytes(
        {
            "name": LABRAM_SOURCE_EVAL_PREFIX_TENSOR_NAME,
            "dtype": "torch.float32",
            "shape": list(values.shape),
        }
    )
    raw = values.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def source_eval_prefix_implementation_sha256() -> str:
    path = Path(__file__).resolve().parent / "models" / "labram_peft.py"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_binding_sha256(row: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {name: row[name] for name in sorted(_EVENT_BASE_FIELDS)}
    )


def _validate_event(value: object, *, ordinal: int) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"events[{ordinal}] must be an object")
    row = dict(value)
    _require_exact_fields(row, _EVENT_FIELDS, field=f"events[{ordinal}]")
    if row["ordinal"] != ordinal:
        raise ValueError("LaBraM source-eval event ordinal/order changed")
    for field in ("event_id", "patient_id", "local_patient_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"events[{ordinal}].{field} must be non-empty")
    if row["model_split"] != "source_eval" or row["official_split"] != "eval":
        raise ValueError("LaBraM source-eval event escaped the locked eval split")
    relative = PurePosixPath(str(row["relative_edf_path"]))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 5
        or relative.parts[0] != "eval"
        or relative.suffix.lower() != ".edf"
    ):
        raise ValueError("LaBraM source-eval EDF path is not canonical")
    for field in (
        "event_record_sha256",
        "edf_sha256",
        "preprocess_config_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
        "input_binding_sha256",
        "prefix_tensor_sha256",
    ):
        _require_sha256(row[field], field=f"events[{ordinal}].{field}")
    if row["preprocess_config_sha256"] != EXPECTED_PREPROCESS_CONFIG_SHA256:
        raise ValueError("LaBraM source-eval preprocess configuration changed")
    index = row["global_event_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("LaBraM source-eval global event index is invalid")
    try:
        start = float(row["global_t0_sec"])
        stop = float(row["global_stop_sec"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LaBraM source-eval interval is invalid") from exc
    if not math.isfinite(start) or not math.isfinite(stop) or start < 0 or stop <= start:
        raise ValueError("LaBraM source-eval interval is invalid")
    if row["processed_window_shape"] != [N_STANDARD_CHANNELS, 12_000] or row[
        "processed_window_dtype"
    ] != "torch.float32":
        raise ValueError("LaBraM source-eval signal must be [19,12000] float32")
    if row["labram_position_binding_policy"] != LABRAM_RAW_HEADER_POSITION_BINDING_POLICY:
        raise ValueError("LaBraM source-eval position policy changed")
    names = row["labram_position_names"]
    ids = row["labram_position_ids"]
    if not isinstance(names, list) or not isinstance(ids, list):
        raise ValueError("LaBraM source-eval position binding must use JSON lists")
    binding = bind_labram_record_positions(names)
    if list(binding.position_names) != names or list(binding.position_ids) != ids:
        raise ValueError("LaBraM source-eval position binding is not reproducible")
    if row["chunk_reassembly_exact"] is not True:
        raise ValueError("LaBraM source-eval event lacks exact chunk reassembly")
    if row["input_binding_sha256"] != _input_binding_sha256(row):
        raise ValueError("LaBraM source-eval input binding SHA changed")
    return row


def _feature_receipt_payload(receipt: LaBraMFeatureReceipt) -> dict[str, object]:
    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("foundation_feature_receipt must be LaBraMFeatureReceipt")
    payload = asdict(receipt)
    for field in ("semantic_channels", "position_names", "position_ids"):
        payload[field] = list(payload[field])
    return payload


def _validate_feature_receipt(value: object) -> LaBraMFeatureReceipt:
    if not isinstance(value, Mapping):
        raise ValueError("foundation_feature_receipt must be an object")
    payload = dict(value)
    expected = frozenset(
        asdict(
            LaBraMFeatureReceipt(
                checkpoint_path="x",
                checkpoint_sha256="0" * 64,
                modeling_path="y",
                modeling_sha256="1" * 64,
                encoder_tensor_count=1,
                semantic_channels=STANDARD_19,
                position_names=STANDARD_19,
                position_ids=tuple(range(1, 20)),
                tile_seconds=4,
            )
        )
    )
    _require_exact_fields(payload, expected, field="foundation_feature_receipt")
    try:
        receipt = LaBraMFeatureReceipt(
            **{
                **payload,
                "semantic_channels": tuple(payload["semantic_channels"]),
                "position_names": tuple(payload["position_names"]),
                "position_ids": tuple(payload["position_ids"]),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("foundation_feature_receipt is invalid") from exc
    fixed = {
        "checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
        "semantic_channels": STANDARD_19,
        "tile_seconds": LABRAM_PEFT_SECONDS_PER_CALL,
        "samples_per_token": 200,
        "token_dim": LABRAM_PEFT_TOKEN_DIM,
        "input_scale_from_volts": 1e4,
    }
    changed = tuple(
        name
        for name, expected_value in fixed.items()
        if getattr(receipt, name) != expected_value
    )
    if changed:
        raise ValueError(f"LaBraM source-eval foundation boundary changed: {changed}")
    return receipt


def _validate_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("LaBraM source-eval prefix manifest must be an object")
    manifest = dict(value)
    _require_exact_fields(manifest, _MANIFEST_FIELDS, field="manifest")
    full_scope = manifest["full_scope"]
    if type(full_scope) is not bool:
        raise ValueError("LaBraM source-eval full_scope must be boolean")
    expected_schema = (
        LABRAM_SOURCE_EVAL_PREFIX_FULL_SCHEMA
        if full_scope
        else LABRAM_SOURCE_EVAL_PREFIX_SMOKE_SCHEMA
    )
    fixed = {
        "schema_version": expected_schema,
        "purpose": LABRAM_SOURCE_EVAL_PREFIX_PURPOSE,
        "serialization": LABRAM_SOURCE_EVAL_PREFIX_SERIALIZATION,
        "model_split": "source_eval",
        "official_split": "eval",
        "locked_evaluation": True,
        "inference_feature_only": True,
        "scope_kind": "full" if full_scope else "smoke_prefix",
        "smoke_only": not full_scope,
        "formal_evaluation_input_authorized": full_scope,
        "training_authorized": False,
        "model_selection_authorized": False,
        "threshold_tuning_authorized": False,
        "selection_policy": (
            "complete_locked_source_eval_event_order_v1"
            if full_scope
            else "first_n_locked_source_eval_event_order_smoke_only_v1"
        ),
        "source_roster_event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
        "source_roster_patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
        "foundation_backbone": "official_pretrained_LaBraM_Base_frozen_blocks_0_9",
        "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
        "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
        "foundation_prefix_blocks": list(range(LABRAM_PEFT_BLOCKS[0])),
        "foundation_prefix_block_stop_exclusive": LABRAM_PEFT_BLOCKS[0],
        "foundation_trainable_parameter_count": 0,
        "raw_event_shape": [19, 12_000],
        "sampling_frequency_hz": 200,
        "event_seconds": 60,
        "call_count_per_event": 15,
        "call_seconds": 4,
        "call_input_shape": [19, 4, 200],
        "call_output_shape": list(LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE[1:]),
        "chunk_policy": "nonoverlap_15x4s_channel_major_exact_reassembly_v1",
        "prefix_cut": "immediately_before_official_transformer_block_10_with_cls",
        "tensor_name": LABRAM_SOURCE_EVAL_PREFIX_TENSOR_NAME,
        "tensor_dtype": "torch.float32",
        "tensor_file": LABRAM_SOURCE_EVAL_PREFIX_TENSOR_FILENAME,
        "raw_replay_verified": True,
        "chunk_reassembly_verified": True,
        "source_train_eeg_loaded": False,
        "source_train_target_values_loaded": False,
        "source_dev_eeg_loaded": False,
        "source_dev_target_values_loaded": False,
        "source_eval_eeg_loaded": True,
        "source_eval_target_values_loaded": False,
        "deepsoz_target_values_loaded": False,
        "tusz_channel_target_values_loaded": False,
        "private_eeg_loaded": False,
        "private_target_values_loaded": False,
        "training_performed": False,
    }
    changed = tuple(
        name for name, expected_value in fixed.items() if manifest[name] != expected_value
    )
    if changed:
        raise ValueError(f"LaBraM source-eval scientific boundary changed: {changed}")

    for field in (
        "event_count",
        "patient_count",
        "source_eval_event_count_loaded",
        "tensor_file_size_bytes",
    ):
        item = manifest[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"LaBraM source-eval {field} must be positive")
    event_count = int(manifest["event_count"])
    if manifest["source_eval_event_count_loaded"] != event_count:
        raise ValueError("LaBraM source-eval loaded event count changed")
    if full_scope:
        if (
            event_count != EXPECTED_SOURCE_EVAL_EVENT_COUNT
            or manifest["patient_count"] != EXPECTED_SOURCE_EVAL_PATIENT_COUNT
        ):
            raise ValueError("Full source-eval prefix requires 185 events/21 patients")
    elif not 1 <= event_count < EXPECTED_SOURCE_EVAL_EVENT_COUNT:
        raise ValueError("Smoke source-eval prefix must be a strict roster prefix")
    if manifest["tensor_shape"] != [
        event_count,
        *LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE,
    ]:
        raise ValueError("LaBraM source-eval tensor shape disagrees with event_count")

    lineage = manifest["lineage"]
    if not isinstance(lineage, Mapping):
        raise ValueError("LaBraM source-eval lineage must be an object")
    lineage = dict(lineage)
    _require_exact_fields(lineage, _LINEAGE_FIELDS, field="lineage")
    for name, digest in lineage.items():
        _require_sha256(digest, field=f"lineage.{name}")
    pinned = {
        "signal_preflight_artifact_sha256": EXPECTED_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
        "signal_preflight_receipt_sha256": EXPECTED_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
        "preprocess_config_sha256": EXPECTED_PREPROCESS_CONFIG_SHA256,
    }
    changed_lineage = tuple(
        name for name, expected_value in pinned.items() if lineage[name] != expected_value
    )
    if changed_lineage:
        raise ValueError(f"LaBraM source-eval pinned lineage changed: {changed_lineage}")
    manifest["lineage"] = lineage

    receipt = _validate_feature_receipt(manifest["foundation_feature_receipt"])
    if manifest["foundation_feature_receipt_sha256"] != labram_feature_receipt_sha256(receipt):
        raise ValueError("LaBraM source-eval foundation receipt SHA changed")
    if manifest["foundation_prefix_implementation_sha256"] != source_eval_prefix_implementation_sha256():
        raise ValueError("LaBraM source-eval prefix implementation changed")
    for field in (
        "foundation_feature_receipt_sha256",
        "foundation_prefix_implementation_sha256",
        "event_order_sha256",
        "patient_roster_sha256",
        "input_binding_roster_sha256",
        "prefix_tensor_roster_sha256",
        "tensor_file_sha256",
    ):
        _require_sha256(manifest[field], field=field)

    values = manifest["events"]
    if not isinstance(values, list) or len(values) != event_count:
        raise ValueError("LaBraM source-eval events disagree with event_count")
    events = [_validate_event(row, ordinal=index) for index, row in enumerate(values)]
    event_ids = tuple(str(row["event_id"]) for row in events)
    if len(set(event_ids)) != event_count:
        raise ValueError("LaBraM source-eval event IDs repeat")
    patients = tuple(sorted({str(row["patient_id"]) for row in events}))
    if len(patients) != manifest["patient_count"]:
        raise ValueError("LaBraM source-eval patient_count disagrees with events")
    expected_hashes = {
        "event_order_sha256": _canonical_sha256(event_ids),
        "patient_roster_sha256": _canonical_sha256(patients),
        "input_binding_roster_sha256": _canonical_sha256(
            tuple((row["event_id"], row["input_binding_sha256"]) for row in events)
        ),
        "prefix_tensor_roster_sha256": _canonical_sha256(
            tuple((row["event_id"], row["prefix_tensor_sha256"]) for row in events)
        ),
    }
    for name, expected_value in expected_hashes.items():
        if manifest[name] != expected_value:
            raise ValueError(f"LaBraM source-eval {name} disagrees with events")
    if full_scope:
        if manifest["event_order_sha256"] != EXPECTED_SOURCE_EVAL_EVENT_ORDER_SHA256:
            raise ValueError("Full LaBraM source-eval event order changed")
        if manifest["patient_roster_sha256"] != EXPECTED_SOURCE_EVAL_PATIENT_ROSTER_SHA256:
            raise ValueError("Full LaBraM source-eval patient roster changed")
        if event_ids[0] != EXPECTED_SOURCE_EVAL_FIRST_EVENT_ID or event_ids[-1] != EXPECTED_SOURCE_EVAL_LAST_EVENT_ID:
            raise ValueError("Full LaBraM source-eval endpoint events changed")
    for field in ("elapsed_sec", "seconds_per_event"):
        item = manifest[field]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0
        ):
            raise ValueError(f"LaBraM source-eval {field} must be finite/non-negative")
    peak = manifest["peak_cuda_memory_bytes"]
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise ValueError("LaBraM source-eval peak CUDA memory is invalid")
    if not isinstance(manifest["materialization_device"], str) or not manifest[
        "materialization_device"
    ]:
        raise ValueError("LaBraM source-eval materialization device is invalid")
    manifest["events"] = events
    return manifest


@dataclass(frozen=True)
class LaBraMSourceEvalPrefixEvent:
    ordinal: int
    event_id: str
    patient_id: str
    local_patient_id: str
    relative_edf_path: str
    global_event_index: int
    global_t0_sec: float
    global_stop_sec: float
    edf_sha256: str
    processed_window_sha256: str
    labram_position_names: tuple[str, ...]
    labram_position_ids: tuple[int, ...]
    prefix_tensor_sha256: str


@dataclass(frozen=True)
class LoadedLaBraMSourceEvalPrefix:
    path: Path
    manifest_sha256: str
    manifest: Mapping[str, object]
    tokens: torch.Tensor
    events: tuple[LaBraMSourceEvalPrefixEvent, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_sha256, field="manifest_sha256")
        count = int(self.manifest["event_count"])
        if tuple(self.tokens.shape) != (
            count,
            *LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE,
        ):
            raise ValueError("Loaded LaBraM source-eval tensor shape changed")
        if (
            self.tokens.device.type != "cpu"
            or self.tokens.dtype != torch.float32
            or self.tokens.requires_grad
            or self.tokens.grad_fn is not None
        ):
            raise TypeError("Loaded LaBraM source-eval tensor must be detached CPU float32")
        if len(self.events) != count:
            raise ValueError("Loaded LaBraM source-eval event carrier changed")

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({event.patient_id for event in self.events}))

    @property
    def full_scope(self) -> bool:
        return bool(self.manifest["full_scope"])


def _runtime_events(
    rows: Sequence[Mapping[str, object]],
) -> tuple[LaBraMSourceEvalPrefixEvent, ...]:
    return tuple(
        LaBraMSourceEvalPrefixEvent(
            ordinal=int(row["ordinal"]),
            event_id=str(row["event_id"]),
            patient_id=str(row["patient_id"]),
            local_patient_id=str(row["local_patient_id"]),
            relative_edf_path=str(row["relative_edf_path"]),
            global_event_index=int(row["global_event_index"]),
            global_t0_sec=float(row["global_t0_sec"]),
            global_stop_sec=float(row["global_stop_sec"]),
            edf_sha256=str(row["edf_sha256"]),
            processed_window_sha256=str(row["processed_window_sha256"]),
            labram_position_names=tuple(row["labram_position_names"]),
            labram_position_ids=tuple(int(value) for value in row["labram_position_ids"]),
            prefix_tensor_sha256=str(row["prefix_tensor_sha256"]),
        )
        for row in rows
    )


def load_labram_source_eval_prefix(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    require_full_scope: bool = True,
) -> LoadedLaBraMSourceEvalPrefix:
    """Strictly load a full cache; smoke loading requires an explicit opt-in."""

    source = _absolute_no_symlink(directory, field="LaBraM source-eval prefix")
    if (
        source.is_symlink()
        or not source.is_dir()
        or {entry.name for entry in source.iterdir()} != set(_FILES)
    ):
        raise ValueError("LaBraM source-eval prefix violates its closed directory schema")
    raw, manifest_sha = _stable_manifest(
        source / LABRAM_SOURCE_EVAL_PREFIX_MANIFEST_FILENAME
    )
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("LaBraM source-eval manifest SHA mismatch")
    manifest = _validate_manifest(_strict_json(raw, field="manifest"))
    if require_full_scope and not manifest["full_scope"]:
        raise ValueError("Formal source-eval loading rejects a smoke prefix cache")
    tensor_path = source / LABRAM_SOURCE_EVAL_PREFIX_TENSOR_FILENAME
    tensor_sha, before = _stable_tensor_sha256(tensor_path)
    if (
        tensor_sha != manifest["tensor_file_sha256"]
        or before.st_size != manifest["tensor_file_size_bytes"]
    ):
        raise ValueError("LaBraM source-eval tensor file binding changed")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    tensors = load_file(str(tensor_path), device="cpu")
    after = tensor_path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError("LaBraM source-eval tensor changed while loaded")
    if set(tensors) != {LABRAM_SOURCE_EVAL_PREFIX_TENSOR_NAME}:
        raise ValueError("LaBraM source-eval safetensors keys changed")
    tokens = tensors[LABRAM_SOURCE_EVAL_PREFIX_TENSOR_NAME].detach().cpu().contiguous()
    if (
        tuple(tokens.shape) != tuple(manifest["tensor_shape"])
        or tokens.dtype != torch.float32
        or tokens.requires_grad
        or not torch.isfinite(tokens).all().item()
    ):
        raise ValueError("LaBraM source-eval prefix tensor metadata/values changed")
    for index, row in enumerate(manifest["events"]):
        if source_eval_prefix_event_tensor_sha256(tokens[index]) != row[
            "prefix_tensor_sha256"
        ]:
            raise ValueError(f"LaBraM source-eval tensor changed at ordinal {index}")
    return LoadedLaBraMSourceEvalPrefix(
        path=source,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        tokens=tokens,
        events=_runtime_events(manifest["events"]),
    )


def publish_labram_source_eval_prefix(
    output_directory: str | Path,
    *,
    tokens: torch.Tensor,
    event_rows: Sequence[Mapping[str, object]],
    source_event_ids: Sequence[str],
    source_patient_ids: Sequence[str],
    lineage: Mapping[str, str],
    foundation_feature_receipt: LaBraMFeatureReceipt,
    full_scope: bool,
    materialization_device: str,
    elapsed_sec: float,
    peak_cuda_memory_bytes: int | None,
) -> LoadedLaBraMSourceEvalPrefix:
    """Atomically publish a target-free full or strict-prefix smoke cache."""

    target = _safe_new_directory(output_directory)
    values = tokens.detach().cpu().contiguous()
    if (
        values.dtype != torch.float32
        or values.requires_grad
        or values.grad_fn is not None
        or not torch.isfinite(values).all().item()
    ):
        raise TypeError("Source-eval prefix publication requires finite detached float32")
    if values.ndim != 4 or tuple(values.shape[1:]) != LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE:
        raise ValueError("Source-eval prefix publication tensor shape changed")

    complete_event_ids = tuple(str(value) for value in source_event_ids)
    complete_patient_ids = tuple(str(value) for value in source_patient_ids)
    if (
        len(complete_event_ids) != EXPECTED_SOURCE_EVAL_EVENT_COUNT
        or len(set(complete_event_ids)) != EXPECTED_SOURCE_EVAL_EVENT_COUNT
        or _canonical_sha256(complete_event_ids) != EXPECTED_SOURCE_EVAL_EVENT_ORDER_SHA256
        or complete_event_ids[0] != EXPECTED_SOURCE_EVAL_FIRST_EVENT_ID
        or complete_event_ids[-1] != EXPECTED_SOURCE_EVAL_LAST_EVENT_ID
    ):
        raise ValueError("Source-eval publication requires the complete pinned event roster")
    if (
        len(complete_patient_ids) != EXPECTED_SOURCE_EVAL_PATIENT_COUNT
        or tuple(sorted(set(complete_patient_ids))) != complete_patient_ids
        or _canonical_sha256(complete_patient_ids)
        != EXPECTED_SOURCE_EVAL_PATIENT_ROSTER_SHA256
    ):
        raise ValueError("Source-eval publication requires the complete pinned patient roster")

    rows_input = tuple(dict(row) for row in event_rows)
    if len(rows_input) != values.shape[0]:
        raise ValueError("Source-eval event rows/tensor count differ")
    event_count = len(rows_input)
    if full_scope and event_count != EXPECTED_SOURCE_EVAL_EVENT_COUNT:
        raise ValueError("Full source-eval publication requires all 185 events")
    if not full_scope and not 1 <= event_count < EXPECTED_SOURCE_EVAL_EVENT_COUNT:
        raise ValueError("Smoke source-eval publication requires a strict prefix")
    if tuple(str(row["event_id"]) for row in rows_input) != complete_event_ids[:event_count]:
        raise ValueError("Source-eval event rows are not a strict pinned roster prefix")

    rows: list[dict[str, object]] = []
    for ordinal, (base, event_tensor) in enumerate(zip(rows_input, values)):
        _require_exact_fields(base, _EVENT_BASE_FIELDS, field=f"event_rows[{ordinal}]")
        row = dict(base)
        row["input_binding_sha256"] = _input_binding_sha256(row)
        row["prefix_tensor_sha256"] = source_eval_prefix_event_tensor_sha256(
            event_tensor
        )
        rows.append(_validate_event(row, ordinal=ordinal))

    patients = tuple(sorted({str(row["patient_id"]) for row in rows}))
    if full_scope and patients != complete_patient_ids:
        raise ValueError("Full source-eval patient roster changed")
    feature_payload = _feature_receipt_payload(foundation_feature_receipt)
    lineage_payload = dict(lineage)
    _require_exact_fields(lineage_payload, _LINEAGE_FIELDS, field="lineage")
    for name, digest in lineage_payload.items():
        _require_sha256(digest, field=f"lineage.{name}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required") from exc
        tensor_path = staging / LABRAM_SOURCE_EVAL_PREFIX_TENSOR_FILENAME
        save_file({LABRAM_SOURCE_EVAL_PREFIX_TENSOR_NAME: values}, str(tensor_path))
        tensor_sha, tensor_stat = _stable_tensor_sha256(tensor_path)
        event_ids = tuple(str(row["event_id"]) for row in rows)
        elapsed = float(elapsed_sec)
        manifest = {
            "schema_version": (
                LABRAM_SOURCE_EVAL_PREFIX_FULL_SCHEMA
                if full_scope
                else LABRAM_SOURCE_EVAL_PREFIX_SMOKE_SCHEMA
            ),
            "purpose": LABRAM_SOURCE_EVAL_PREFIX_PURPOSE,
            "serialization": LABRAM_SOURCE_EVAL_PREFIX_SERIALIZATION,
            "model_split": "source_eval",
            "official_split": "eval",
            "locked_evaluation": True,
            "inference_feature_only": True,
            "scope_kind": "full" if full_scope else "smoke_prefix",
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "formal_evaluation_input_authorized": full_scope,
            "training_authorized": False,
            "model_selection_authorized": False,
            "threshold_tuning_authorized": False,
            "selection_policy": (
                "complete_locked_source_eval_event_order_v1"
                if full_scope
                else "first_n_locked_source_eval_event_order_smoke_only_v1"
            ),
            "source_roster_event_count": EXPECTED_SOURCE_EVAL_EVENT_COUNT,
            "source_roster_patient_count": EXPECTED_SOURCE_EVAL_PATIENT_COUNT,
            "event_count": event_count,
            "patient_count": len(patients),
            "events": rows,
            "event_order_sha256": _canonical_sha256(event_ids),
            "patient_roster_sha256": _canonical_sha256(patients),
            "input_binding_roster_sha256": _canonical_sha256(
                tuple((row["event_id"], row["input_binding_sha256"]) for row in rows)
            ),
            "prefix_tensor_roster_sha256": _canonical_sha256(
                tuple((row["event_id"], row["prefix_tensor_sha256"]) for row in rows)
            ),
            "lineage": lineage_payload,
            "foundation_backbone": "official_pretrained_LaBraM_Base_frozen_blocks_0_9",
            "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "foundation_feature_receipt": feature_payload,
            "foundation_feature_receipt_sha256": labram_feature_receipt_sha256(
                foundation_feature_receipt
            ),
            "foundation_prefix_implementation_sha256": source_eval_prefix_implementation_sha256(),
            "foundation_prefix_blocks": list(range(LABRAM_PEFT_BLOCKS[0])),
            "foundation_prefix_block_stop_exclusive": LABRAM_PEFT_BLOCKS[0],
            "foundation_trainable_parameter_count": 0,
            "raw_event_shape": [19, 12_000],
            "sampling_frequency_hz": 200,
            "event_seconds": 60,
            "call_count_per_event": 15,
            "call_seconds": 4,
            "call_input_shape": [19, 4, 200],
            "call_output_shape": list(LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE[1:]),
            "chunk_policy": "nonoverlap_15x4s_channel_major_exact_reassembly_v1",
            "prefix_cut": "immediately_before_official_transformer_block_10_with_cls",
            "tensor_name": LABRAM_SOURCE_EVAL_PREFIX_TENSOR_NAME,
            "tensor_shape": list(values.shape),
            "tensor_dtype": "torch.float32",
            "tensor_file": LABRAM_SOURCE_EVAL_PREFIX_TENSOR_FILENAME,
            "tensor_file_sha256": tensor_sha,
            "tensor_file_size_bytes": tensor_stat.st_size,
            "materialization_device": str(materialization_device),
            "elapsed_sec": elapsed,
            "seconds_per_event": elapsed / event_count,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "raw_replay_verified": True,
            "chunk_reassembly_verified": True,
            "source_train_eeg_loaded": False,
            "source_train_target_values_loaded": False,
            "source_dev_eeg_loaded": False,
            "source_dev_target_values_loaded": False,
            "source_eval_eeg_loaded": True,
            "source_eval_event_count_loaded": event_count,
            "source_eval_target_values_loaded": False,
            "deepsoz_target_values_loaded": False,
            "tusz_channel_target_values_loaded": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "training_performed": False,
        }
        _validate_manifest(manifest)
        manifest_path = staging / LABRAM_SOURCE_EVAL_PREFIX_MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        with tensor_path.open("rb") as stream:
            os.fsync(stream.fileno())
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        staged = load_labram_source_eval_prefix(
            staging,
            expected_manifest_sha256=manifest_sha,
            require_full_scope=full_scope,
        )
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)
        return load_labram_source_eval_prefix(
            target,
            expected_manifest_sha256=staged.manifest_sha256,
            require_full_scope=full_scope,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "EXPECTED_PREPROCESS_CONFIG_SHA256",
    "EXPECTED_SIGNAL_PREFLIGHT_ARTIFACT_SHA256",
    "EXPECTED_SIGNAL_PREFLIGHT_RECEIPT_SHA256",
    "EXPECTED_SOURCE_EVAL_EVENT_COUNT",
    "EXPECTED_SOURCE_EVAL_EVENT_ORDER_SHA256",
    "EXPECTED_SOURCE_EVAL_PATIENT_COUNT",
    "EXPECTED_SOURCE_EVAL_PATIENT_ROSTER_SHA256",
    "LABRAM_SOURCE_EVAL_PREFIX_EVENT_SHAPE",
    "LABRAM_SOURCE_EVAL_PREFIX_FULL_SCHEMA",
    "LABRAM_SOURCE_EVAL_PREFIX_SMOKE_SCHEMA",
    "LaBraMSourceEvalPrefixEvent",
    "LoadedLaBraMSourceEvalPrefix",
    "load_labram_source_eval_prefix",
    "publish_labram_source_eval_prefix",
    "source_eval_prefix_event_tensor_sha256",
]
