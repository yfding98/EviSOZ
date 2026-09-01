"""Strict source-train-only cache for the frozen LaBraM PEFT prefix.

The cache contains the detached activation immediately before official
LaBraM block 10.  One 60 s event is represented by fifteen independent 4 s
calls, hence ``[15,77,200]`` (CLS plus 19 x 4 patch tokens).  This module owns
the closed full/smoke schemas and the only supported publisher/loader.

No SOZ, TUSZ-involvement, source-dev, source-eval, or private target is part
of this artifact.  The full schema is deliberately impossible to obtain via
the smoke/``--limit`` path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
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


LABRAM_PEFT_PREFIX_CACHE_FULL_SCHEMA = (
    "soz_labram_peft_prefix_cache_v8_full"
)
LABRAM_PEFT_PREFIX_CACHE_SMOKE_SCHEMA = (
    "soz_labram_peft_prefix_cache_v8_smoke"
)
LABRAM_PEFT_PREFIX_CACHE_PURPOSE = (
    "development_source_train_fold_local_labram_minimal_peft_prefix"
)
LABRAM_PEFT_PREFIX_CACHE_SERIALIZATION = (
    "canonical_json_plus_safetensors_no_pickle_v1"
)
LABRAM_PEFT_PREFIX_MANIFEST_FILENAME = "manifest.json"
LABRAM_PEFT_PREFIX_TENSOR_FILENAME = "prefix.safetensors"
LABRAM_PEFT_PREFIX_TENSOR_NAME = "prefix_tokens"
LABRAM_PEFT_PREFIX_EVENT_SHAPE = (
    15,
    LABRAM_PEFT_PREFIX_TOKENS,
    LABRAM_PEFT_TOKEN_DIM,
)
EXPECTED_SOURCE_TRAIN_EVENT_COUNT = 582
EXPECTED_SOURCE_TRAIN_PATIENT_COUNT = 65
EXPECTED_SOURCE_TRAIN_EVENT_ORDER_SHA256 = (
    "c45fe14fc4cdc1767710aa5bc22b3dce4cb08caa340f9e99a035bf134e59d434"
)
EXPECTED_SOURCE_TRAIN_IV_MANIFEST_SHA256 = (
    "ccd238b17e1da0aa24f2542a314c770900eeed71cbc31282a4acb76dcf957821"
)
EXPECTED_SOURCE_TRAIN_IV_RECEIPT_SHA256 = (
    "a977d692ae09ef5a37c131863f3544a9693834960ab50d7977d04ce4887d61d1"
)
EXPECTED_FROZEN_H_CROSSWALK_MANIFEST_SHA256 = (
    "f5a0b40e7d9ecc48ffb2f10a76128da4e110b791db47ac09ace54495bd2d797b"
)
EXPECTED_FROZEN_H_CROSSWALK_RECEIPT_SHA256 = (
    "4eec735065d93f761c1e17753977fe1f0e633d1fdbb6c6888f0af4eb78f6bbee"
)

_FILES = frozenset(
    {LABRAM_PEFT_PREFIX_MANIFEST_FILENAME, LABRAM_PEFT_PREFIX_TENSOR_FILENAME}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_TENSOR_BYTES = 2 * 1024 * 1024 * 1024

_LINEAGE_FIELDS = frozenset(
    {
        "source_train_iv_manifest_sha256",
        "source_train_iv_receipt_sha256",
        "source_train_iv_event_order_sha256",
        "crosswalk_manifest_sha256",
        "crosswalk_receipt_sha256",
        "crosswalk_parent_capability_manifest_sha256",
        "crosswalk_event_order_sha256",
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
    }
)

_EVENT_BASE_FIELDS = frozenset(
    {
        "ordinal",
        "evidence_event_id",
        "patient_id",
        "public_patient_id",
        "oof_fold",
        "token_event_id",
        "relative_edf_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "seizure_type",
        "event_record_sha256",
        "edf_sha256",
        "channel_annotation_sha256",
        "global_annotation_sha256",
        "annotation_pair_sha256",
        "processed_window_sha256",
        "processed_window_shape",
        "processed_window_dtype",
        "tusz_signal_preflight_receipt_sha256",
        "raw_replay_sha256",
        "labram_position_binding_policy",
        "labram_position_names",
        "labram_position_ids",
        "chunk_reassembly_exact",
    }
)
_EVENT_FIELDS = _EVENT_BASE_FIELDS | frozenset(
    {"input_binding_sha256", "prefix_tensor_sha256"}
)

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "serialization",
        "development_only",
        "model_split",
        "scope_kind",
        "full_scope",
        "smoke_only",
        "formal_training_input_authorized",
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
        "equivalence_control_event_id",
        "zero_adapter_official_equivalence_max_abs_error",
        "zero_adapter_official_equivalence_verified",
        "raw_replay_verified",
        "chunk_reassembly_verified",
        "source_train_eeg_loaded",
        "source_train_event_count_loaded",
        "source_train_iv_evidence_values_loaded",
        "source_train_iv_evidence_values_used",
        "deepsoz_target_values_loaded",
        "source_train_target_values_loaded",
        "tusz_involvement_target_values_loaded",
        "source_dev_eeg_loaded",
        "source_dev_target_values_loaded",
        "source_dev_used",
        "source_eval_eeg_loaded",
        "source_eval_target_values_loaded",
        "source_eval_used",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "private_used",
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
        raise ValueError("LaBraM prefix cache contains non-canonical data") from exc


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
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
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
    target = _absolute_no_symlink(path, field="LaBraM prefix output")
    if target.name in {"", ".", ".."}:
        raise ValueError("LaBraM prefix output requires a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise FileNotFoundError(target.parent)
    return target


def _stable_manifest(path: Path) -> tuple[bytes, str]:
    source = _absolute_no_symlink(path, field="LaBraM prefix manifest")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= _MAX_MANIFEST_BYTES:
            raise ValueError("LaBraM prefix manifest is not a bounded regular file")
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
        raise RuntimeError("LaBraM prefix manifest changed while read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError("LaBraM prefix manifest read was incomplete")
    return raw, hashlib.sha256(raw).hexdigest()


def _stable_tensor_sha256(path: Path) -> tuple[str, os.stat_result]:
    source = _absolute_no_symlink(path, field="LaBraM prefix tensor")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= _MAX_TENSOR_BYTES:
            raise ValueError("LaBraM prefix tensor is not a bounded regular file")
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
        raise RuntimeError("LaBraM prefix tensor changed while hashed")
    return digest.hexdigest(), after


def prefix_event_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one detached float32 ``[15,77,200]`` prefix activation."""

    values = tensor.detach().cpu().contiguous()
    if tuple(values.shape) != LABRAM_PEFT_PREFIX_EVENT_SHAPE:
        raise ValueError("LaBraM prefix event tensor has an invalid shape")
    if values.dtype != torch.float32 or values.requires_grad:
        raise TypeError("LaBraM prefix event tensor must be detached float32")
    metadata = _canonical_json_bytes(
        {
            "name": LABRAM_PEFT_PREFIX_TENSOR_NAME,
            "dtype": "torch.float32",
            "shape": list(values.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = values.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def prefix_implementation_sha256() -> str:
    path = Path(__file__).resolve().parent / "models" / "labram_peft.py"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_binding_sha256(row: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            name: row[name]
            for name in sorted(_EVENT_BASE_FIELDS - {"chunk_reassembly_exact"})
        }
    )


def _validate_event(value: object, *, ordinal: int) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"events[{ordinal}] must be an object")
    row = dict(value)
    _require_exact_fields(row, _EVENT_FIELDS, field=f"events[{ordinal}]")
    if row["ordinal"] != ordinal:
        raise ValueError("LaBraM prefix event ordinal/order changed")
    for field in (
        "evidence_event_id",
        "patient_id",
        "public_patient_id",
        "token_event_id",
        "relative_edf_path",
        "seizure_type",
        "processed_window_dtype",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"events[{ordinal}].{field} must be non-empty")
    for field in (
        "event_record_sha256",
        "edf_sha256",
        "channel_annotation_sha256",
        "global_annotation_sha256",
        "annotation_pair_sha256",
        "processed_window_sha256",
        "tusz_signal_preflight_receipt_sha256",
        "raw_replay_sha256",
        "input_binding_sha256",
        "prefix_tensor_sha256",
    ):
        _require_sha256(row[field], field=f"events[{ordinal}].{field}")
    for field in ("oof_fold", "global_event_index"):
        item = row[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"events[{ordinal}].{field} is invalid")
    if row["oof_fold"] not in range(5):
        raise ValueError("LaBraM prefix event OOF fold must be 0..4")
    try:
        t0 = float(row["global_t0_sec"])
        stop = float(row["global_stop_sec"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LaBraM prefix event interval is invalid") from exc
    if not math.isfinite(t0) or not math.isfinite(stop) or stop <= t0:
        raise ValueError("LaBraM prefix event interval is invalid")
    if row["processed_window_shape"] != [N_STANDARD_CHANNELS, 12_000]:
        raise ValueError("LaBraM prefix source window must be [19,12000]")
    if row["processed_window_dtype"] != "torch.float32":
        raise ValueError("LaBraM prefix source window must be replayed as float32")
    if row["labram_position_binding_policy"] != LABRAM_RAW_HEADER_POSITION_BINDING_POLICY:
        raise ValueError("LaBraM prefix position-binding policy changed")
    names = row["labram_position_names"]
    ids = row["labram_position_ids"]
    if not isinstance(names, list) or not isinstance(ids, list):
        raise ValueError("LaBraM prefix position binding must use JSON lists")
    binding = bind_labram_record_positions(names)
    if list(binding.position_names) != names or list(binding.position_ids) != ids:
        raise ValueError("LaBraM prefix position binding is not reproducible")
    if row["chunk_reassembly_exact"] is not True:
        raise ValueError("LaBraM prefix event lacks exact chunk reassembly")
    if row["input_binding_sha256"] != _input_binding_sha256(row):
        raise ValueError("LaBraM prefix input binding SHA changed")
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
    expected = frozenset(asdict(LaBraMFeatureReceipt(
        checkpoint_path="x",
        checkpoint_sha256="0" * 64,
        modeling_path="y",
        modeling_sha256="1" * 64,
        encoder_tensor_count=1,
        semantic_channels=STANDARD_19,
        position_names=STANDARD_19,
        position_ids=tuple(range(1, 20)),
        tile_seconds=4,
    )))
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
    changed = tuple(name for name, expected_value in fixed.items() if getattr(receipt, name) != expected_value)
    if changed:
        raise ValueError(f"foundation_feature_receipt boundary changed: {changed}")
    return receipt


def _validate_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("LaBraM prefix manifest must be an object")
    manifest = dict(value)
    _require_exact_fields(manifest, _MANIFEST_FIELDS, field="LaBraM prefix manifest")
    full_scope = manifest["full_scope"]
    if type(full_scope) is not bool:
        raise ValueError("LaBraM prefix full_scope must be boolean")
    expected_schema = (
        LABRAM_PEFT_PREFIX_CACHE_FULL_SCHEMA
        if full_scope
        else LABRAM_PEFT_PREFIX_CACHE_SMOKE_SCHEMA
    )
    fixed = {
        "schema_version": expected_schema,
        "purpose": LABRAM_PEFT_PREFIX_CACHE_PURPOSE,
        "serialization": LABRAM_PEFT_PREFIX_CACHE_SERIALIZATION,
        "development_only": True,
        "model_split": "source_train",
        "scope_kind": "full" if full_scope else "smoke_prefix",
        "smoke_only": not full_scope,
        "formal_training_input_authorized": full_scope,
        "selection_policy": (
            "complete_frozen_source_train_event_order_v1"
            if full_scope
            else "first_n_frozen_source_train_event_order_smoke_only_v1"
        ),
        "source_roster_event_count": EXPECTED_SOURCE_TRAIN_EVENT_COUNT,
        "source_roster_patient_count": EXPECTED_SOURCE_TRAIN_PATIENT_COUNT,
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
        "call_output_shape": list(LABRAM_PEFT_PREFIX_EVENT_SHAPE[1:]),
        "chunk_policy": "nonoverlap_15x4s_channel_major_exact_reassembly_v1",
        "prefix_cut": "immediately_before_official_transformer_block_10_with_cls",
        "tensor_name": LABRAM_PEFT_PREFIX_TENSOR_NAME,
        "tensor_dtype": "torch.float32",
        "tensor_file": LABRAM_PEFT_PREFIX_TENSOR_FILENAME,
        "zero_adapter_official_equivalence_verified": True,
        "raw_replay_verified": True,
        "chunk_reassembly_verified": True,
        "source_train_eeg_loaded": True,
        "source_train_iv_evidence_values_loaded": True,
        "source_train_iv_evidence_values_used": False,
        "deepsoz_target_values_loaded": False,
        "source_train_target_values_loaded": False,
        "tusz_involvement_target_values_loaded": False,
        "source_dev_eeg_loaded": False,
        "source_dev_target_values_loaded": False,
        "source_dev_used": False,
        "source_eval_eeg_loaded": False,
        "source_eval_target_values_loaded": False,
        "source_eval_used": False,
        "private_eeg_loaded": False,
        "private_target_values_loaded": False,
        "private_used": False,
        "training_performed": False,
    }
    changed = tuple(name for name, expected_value in fixed.items() if manifest[name] != expected_value)
    if changed:
        raise ValueError(f"LaBraM prefix scientific boundary changed: {changed}")

    for field in (
        "event_count",
        "patient_count",
        "source_train_event_count_loaded",
        "tensor_file_size_bytes",
    ):
        item = manifest[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"LaBraM prefix {field} must be a positive integer")
    event_count = int(manifest["event_count"])
    if manifest["source_train_event_count_loaded"] != event_count:
        raise ValueError("LaBraM prefix loaded event count changed")
    if full_scope:
        if event_count != EXPECTED_SOURCE_TRAIN_EVENT_COUNT or manifest["patient_count"] != EXPECTED_SOURCE_TRAIN_PATIENT_COUNT:
            raise ValueError("Full LaBraM prefix scope requires 582 events/65 patients")
    elif not 1 <= event_count < EXPECTED_SOURCE_TRAIN_EVENT_COUNT:
        raise ValueError("Smoke LaBraM prefix scope must be a strict roster prefix")
    if manifest["tensor_shape"] != [event_count, *LABRAM_PEFT_PREFIX_EVENT_SHAPE]:
        raise ValueError("LaBraM prefix tensor shape disagrees with event_count")

    lineage = manifest["lineage"]
    if not isinstance(lineage, Mapping):
        raise ValueError("LaBraM prefix lineage must be an object")
    lineage = dict(lineage)
    _require_exact_fields(lineage, _LINEAGE_FIELDS, field="LaBraM prefix lineage")
    for name, digest in lineage.items():
        _require_sha256(digest, field=f"lineage.{name}")
    if lineage["source_train_iv_event_order_sha256"] != EXPECTED_SOURCE_TRAIN_EVENT_ORDER_SHA256 or lineage["crosswalk_event_order_sha256"] != EXPECTED_SOURCE_TRAIN_EVENT_ORDER_SHA256:
        raise ValueError("LaBraM prefix upstream event order changed")
    pinned_lineage = {
        "source_train_iv_manifest_sha256": EXPECTED_SOURCE_TRAIN_IV_MANIFEST_SHA256,
        "source_train_iv_receipt_sha256": EXPECTED_SOURCE_TRAIN_IV_RECEIPT_SHA256,
        "crosswalk_manifest_sha256": EXPECTED_FROZEN_H_CROSSWALK_MANIFEST_SHA256,
        "crosswalk_receipt_sha256": EXPECTED_FROZEN_H_CROSSWALK_RECEIPT_SHA256,
    }
    changed_lineage = tuple(
        name
        for name, expected_value in pinned_lineage.items()
        if lineage[name] != expected_value
    )
    if changed_lineage:
        raise ValueError(
            f"LaBraM prefix pinned lineage changed: {changed_lineage}"
        )
    manifest["lineage"] = lineage

    receipt = _validate_feature_receipt(manifest["foundation_feature_receipt"])
    if manifest["foundation_feature_receipt_sha256"] != labram_feature_receipt_sha256(receipt):
        raise ValueError("LaBraM prefix foundation receipt SHA changed")
    if manifest["foundation_prefix_implementation_sha256"] != prefix_implementation_sha256():
        raise ValueError("LaBraM prefix implementation source changed")
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
        raise ValueError("LaBraM prefix event roster disagrees with event_count")
    events = [_validate_event(item, ordinal=index) for index, item in enumerate(values)]
    event_ids = tuple(str(row["evidence_event_id"]) for row in events)
    if len(set(event_ids)) != event_count:
        raise ValueError("LaBraM prefix evidence event IDs repeat")
    token_ids = tuple(str(row["token_event_id"]) for row in events)
    if len(set(token_ids)) != event_count:
        raise ValueError("LaBraM prefix token event IDs repeat")
    patients = tuple(sorted({str(row["patient_id"]) for row in events}))
    if len(patients) != manifest["patient_count"]:
        raise ValueError("LaBraM prefix patient_count disagrees with events")
    folds_by_patient: dict[str, set[int]] = {}
    for row in events:
        folds_by_patient.setdefault(str(row["patient_id"]), set()).add(int(row["oof_fold"]))
    if any(len(folds) != 1 for folds in folds_by_patient.values()):
        raise ValueError("A LaBraM prefix patient crosses OOF folds")
    expected_hashes = {
        "event_order_sha256": _canonical_sha256(event_ids),
        "patient_roster_sha256": _canonical_sha256(patients),
        "input_binding_roster_sha256": _canonical_sha256(tuple((row["evidence_event_id"], row["input_binding_sha256"]) for row in events)),
        "prefix_tensor_roster_sha256": _canonical_sha256(tuple((row["evidence_event_id"], row["prefix_tensor_sha256"]) for row in events)),
    }
    for name, expected_value in expected_hashes.items():
        if manifest[name] != expected_value:
            raise ValueError(f"LaBraM prefix {name} disagrees with events")
    if full_scope and manifest["event_order_sha256"] != EXPECTED_SOURCE_TRAIN_EVENT_ORDER_SHA256:
        raise ValueError("Full LaBraM prefix event order changed")
    for field in ("elapsed_sec", "seconds_per_event"):
        value_float = manifest[field]
        if isinstance(value_float, bool) or not isinstance(value_float, (int, float)) or not math.isfinite(float(value_float)) or float(value_float) < 0:
            raise ValueError(f"LaBraM prefix {field} must be finite/non-negative")
    peak = manifest["peak_cuda_memory_bytes"]
    if peak is not None and (isinstance(peak, bool) or not isinstance(peak, int) or peak < 0):
        raise ValueError("LaBraM prefix peak CUDA memory is invalid")
    if not isinstance(manifest["materialization_device"], str) or not manifest["materialization_device"]:
        raise ValueError("LaBraM prefix materialization device is invalid")
    if manifest["equivalence_control_event_id"] != events[0]["evidence_event_id"]:
        raise ValueError("LaBraM prefix equivalence control event changed")
    equivalence_error = manifest[
        "zero_adapter_official_equivalence_max_abs_error"
    ]
    if (
        isinstance(equivalence_error, bool)
        or not isinstance(equivalence_error, (int, float))
        or not math.isfinite(float(equivalence_error))
        or float(equivalence_error) < 0.0
        or float(equivalence_error) > 1e-6
    ):
        raise ValueError(
            "LaBraM prefix zero-adapter equivalence exceeds 1e-6"
        )
    manifest["events"] = events
    return manifest


@dataclass(frozen=True)
class LaBraMPEFTPrefixEvent:
    ordinal: int
    evidence_event_id: str
    patient_id: str
    public_patient_id: str
    oof_fold: int
    token_event_id: str
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
class LoadedLaBraMPEFTPrefixCache:
    path: Path
    manifest_sha256: str
    manifest: Mapping[str, object]
    tokens: torch.Tensor
    events: tuple[LaBraMPEFTPrefixEvent, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_sha256, field="manifest_sha256")
        event_count = int(self.manifest["event_count"])
        if tuple(self.tokens.shape) != (event_count, *LABRAM_PEFT_PREFIX_EVENT_SHAPE):
            raise ValueError("Loaded LaBraM prefix tensor shape changed")
        if self.tokens.device.type != "cpu" or self.tokens.dtype != torch.float32 or self.tokens.requires_grad or self.tokens.grad_fn is not None:
            raise TypeError("Loaded LaBraM prefix tensor must be detached CPU float32")
        if len(self.events) != event_count:
            raise ValueError("Loaded LaBraM prefix event carrier changed")

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.evidence_event_id for event in self.events)

    @property
    def patient_ids_by_event(self) -> tuple[str, ...]:
        return tuple(event.patient_id for event in self.events)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.patient_ids_by_event)))

    @property
    def oof_folds(self) -> tuple[int, ...]:
        return tuple(event.oof_fold for event in self.events)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def full_scope(self) -> bool:
        return bool(self.manifest["full_scope"])


def _runtime_events(rows: Sequence[Mapping[str, object]]) -> tuple[LaBraMPEFTPrefixEvent, ...]:
    return tuple(
        LaBraMPEFTPrefixEvent(
            ordinal=int(row["ordinal"]),
            evidence_event_id=str(row["evidence_event_id"]),
            patient_id=str(row["patient_id"]),
            public_patient_id=str(row["public_patient_id"]),
            oof_fold=int(row["oof_fold"]),
            token_event_id=str(row["token_event_id"]),
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


def load_labram_peft_prefix_cache(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    require_full_scope: bool = True,
) -> LoadedLaBraMPEFTPrefixCache:
    """Strictly load and hash-bind one full or smoke prefix cache."""

    source = _absolute_no_symlink(directory, field="LaBraM prefix cache")
    if source.is_symlink() or not source.is_dir() or {entry.name for entry in source.iterdir()} != set(_FILES):
        raise ValueError("LaBraM prefix cache violates its closed directory schema")
    manifest_raw, manifest_sha = _stable_manifest(source / LABRAM_PEFT_PREFIX_MANIFEST_FILENAME)
    if manifest_sha != _require_sha256(expected_manifest_sha256, field="expected_manifest_sha256"):
        raise ValueError("LaBraM prefix manifest SHA mismatch")
    manifest = _validate_manifest(_strict_json(manifest_raw, field="LaBraM prefix manifest"))
    if require_full_scope and not manifest["full_scope"]:
        raise ValueError("Formal PEFT loading rejects a smoke prefix cache")
    tensor_path = source / LABRAM_PEFT_PREFIX_TENSOR_FILENAME
    tensor_sha, before = _stable_tensor_sha256(tensor_path)
    if tensor_sha != manifest["tensor_file_sha256"] or before.st_size != manifest["tensor_file_size_bytes"]:
        raise ValueError("LaBraM prefix tensor file binding changed")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    tensors = load_file(str(tensor_path), device="cpu")
    after = tensor_path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_identity != after_identity:
        raise RuntimeError("LaBraM prefix tensor changed while loaded")
    if set(tensors) != {LABRAM_PEFT_PREFIX_TENSOR_NAME}:
        raise ValueError("LaBraM prefix safetensors keys changed")
    tokens = tensors[LABRAM_PEFT_PREFIX_TENSOR_NAME].detach().cpu().contiguous()
    if tuple(tokens.shape) != tuple(manifest["tensor_shape"]) or tokens.dtype != torch.float32 or tokens.requires_grad:
        raise ValueError("LaBraM prefix tensor metadata changed")
    if not torch.isfinite(tokens).all().item():
        raise ValueError("LaBraM prefix tensor contains non-finite values")
    for index, row in enumerate(manifest["events"]):
        if prefix_event_tensor_sha256(tokens[index]) != row["prefix_tensor_sha256"]:
            raise ValueError(f"LaBraM prefix event tensor changed at ordinal {index}")
    return LoadedLaBraMPEFTPrefixCache(
        path=source,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        tokens=tokens,
        events=_runtime_events(manifest["events"]),
    )


def publish_labram_peft_prefix_cache(
    output_directory: str | Path,
    *,
    tokens: torch.Tensor,
    event_rows: Sequence[Mapping[str, object]],
    lineage: Mapping[str, str],
    foundation_feature_receipt: LaBraMFeatureReceipt,
    full_scope: bool,
    materialization_device: str,
    elapsed_sec: float,
    peak_cuda_memory_bytes: int | None,
    zero_adapter_official_equivalence_max_abs_error: float,
) -> LoadedLaBraMPEFTPrefixCache:
    """Atomically publish detached prefix values under the closed v8 schema."""

    target = _safe_new_directory(output_directory)
    values = tokens.detach().cpu().contiguous()
    if values.dtype != torch.float32 or values.requires_grad or values.grad_fn is not None:
        raise TypeError("LaBraM prefix publication requires detached float32")
    if values.ndim != 4 or tuple(values.shape[1:]) != LABRAM_PEFT_PREFIX_EVENT_SHAPE:
        raise ValueError("LaBraM prefix publication tensor shape changed")
    rows_input = tuple(dict(row) for row in event_rows)
    if len(rows_input) != values.shape[0]:
        raise ValueError("LaBraM prefix event rows/tensor count differ")
    rows: list[dict[str, object]] = []
    for ordinal, (base, event_tensor) in enumerate(zip(rows_input, values)):
        _require_exact_fields(base, _EVENT_BASE_FIELDS, field=f"event_rows[{ordinal}]")
        row = dict(base)
        row["input_binding_sha256"] = _input_binding_sha256(row)
        row["prefix_tensor_sha256"] = prefix_event_tensor_sha256(event_tensor)
        rows.append(_validate_event(row, ordinal=ordinal))
    event_count = len(rows)
    if full_scope and event_count != EXPECTED_SOURCE_TRAIN_EVENT_COUNT:
        raise ValueError("Full LaBraM prefix publication requires all 582 events")
    if not full_scope and not 1 <= event_count < EXPECTED_SOURCE_TRAIN_EVENT_COUNT:
        raise ValueError("Smoke LaBraM prefix publication requires a strict prefix")
    patients = tuple(sorted({str(row["patient_id"]) for row in rows}))
    feature_payload = _feature_receipt_payload(foundation_feature_receipt)
    lineage_payload = dict(lineage)
    _require_exact_fields(lineage_payload, _LINEAGE_FIELDS, field="LaBraM prefix lineage")
    for name, digest in lineage_payload.items():
        _require_sha256(digest, field=f"lineage.{name}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required") from exc
        tensor_path = staging / LABRAM_PEFT_PREFIX_TENSOR_FILENAME
        save_file({LABRAM_PEFT_PREFIX_TENSOR_NAME: values}, str(tensor_path))
        tensor_sha, tensor_stat = _stable_tensor_sha256(tensor_path)
        event_ids = tuple(str(row["evidence_event_id"]) for row in rows)
        elapsed = float(elapsed_sec)
        manifest = {
            "schema_version": LABRAM_PEFT_PREFIX_CACHE_FULL_SCHEMA if full_scope else LABRAM_PEFT_PREFIX_CACHE_SMOKE_SCHEMA,
            "purpose": LABRAM_PEFT_PREFIX_CACHE_PURPOSE,
            "serialization": LABRAM_PEFT_PREFIX_CACHE_SERIALIZATION,
            "development_only": True,
            "model_split": "source_train",
            "scope_kind": "full" if full_scope else "smoke_prefix",
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "formal_training_input_authorized": full_scope,
            "selection_policy": "complete_frozen_source_train_event_order_v1" if full_scope else "first_n_frozen_source_train_event_order_smoke_only_v1",
            "source_roster_event_count": EXPECTED_SOURCE_TRAIN_EVENT_COUNT,
            "source_roster_patient_count": EXPECTED_SOURCE_TRAIN_PATIENT_COUNT,
            "event_count": event_count,
            "patient_count": len(patients),
            "events": rows,
            "event_order_sha256": _canonical_sha256(event_ids),
            "patient_roster_sha256": _canonical_sha256(patients),
            "input_binding_roster_sha256": _canonical_sha256(tuple((row["evidence_event_id"], row["input_binding_sha256"]) for row in rows)),
            "prefix_tensor_roster_sha256": _canonical_sha256(tuple((row["evidence_event_id"], row["prefix_tensor_sha256"]) for row in rows)),
            "lineage": lineage_payload,
            "foundation_backbone": "official_pretrained_LaBraM_Base_frozen_blocks_0_9",
            "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "foundation_feature_receipt": feature_payload,
            "foundation_feature_receipt_sha256": labram_feature_receipt_sha256(foundation_feature_receipt),
            "foundation_prefix_implementation_sha256": prefix_implementation_sha256(),
            "foundation_prefix_blocks": list(range(LABRAM_PEFT_BLOCKS[0])),
            "foundation_prefix_block_stop_exclusive": LABRAM_PEFT_BLOCKS[0],
            "foundation_trainable_parameter_count": 0,
            "raw_event_shape": [19, 12_000],
            "sampling_frequency_hz": 200,
            "event_seconds": 60,
            "call_count_per_event": 15,
            "call_seconds": 4,
            "call_input_shape": [19, 4, 200],
            "call_output_shape": list(LABRAM_PEFT_PREFIX_EVENT_SHAPE[1:]),
            "chunk_policy": "nonoverlap_15x4s_channel_major_exact_reassembly_v1",
            "prefix_cut": "immediately_before_official_transformer_block_10_with_cls",
            "tensor_name": LABRAM_PEFT_PREFIX_TENSOR_NAME,
            "tensor_shape": list(values.shape),
            "tensor_dtype": "torch.float32",
            "tensor_file": LABRAM_PEFT_PREFIX_TENSOR_FILENAME,
            "tensor_file_sha256": tensor_sha,
            "tensor_file_size_bytes": tensor_stat.st_size,
            "materialization_device": str(materialization_device),
            "elapsed_sec": elapsed,
            "seconds_per_event": elapsed / event_count,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "equivalence_control_event_id": rows[0]["evidence_event_id"],
            "zero_adapter_official_equivalence_max_abs_error": float(
                zero_adapter_official_equivalence_max_abs_error
            ),
            "zero_adapter_official_equivalence_verified": True,
            "raw_replay_verified": True,
            "chunk_reassembly_verified": True,
            "source_train_eeg_loaded": True,
            "source_train_event_count_loaded": event_count,
            "source_train_iv_evidence_values_loaded": True,
            "source_train_iv_evidence_values_used": False,
            "deepsoz_target_values_loaded": False,
            "source_train_target_values_loaded": False,
            "tusz_involvement_target_values_loaded": False,
            "source_dev_eeg_loaded": False,
            "source_dev_target_values_loaded": False,
            "source_dev_used": False,
            "source_eval_eeg_loaded": False,
            "source_eval_target_values_loaded": False,
            "source_eval_used": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_used": False,
            "training_performed": False,
        }
        _validate_manifest(manifest)
        manifest_path = staging / LABRAM_PEFT_PREFIX_MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        with tensor_path.open("rb") as stream:
            os.fsync(stream.fileno())
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        loaded = load_labram_peft_prefix_cache(staging, expected_manifest_sha256=manifest_sha, require_full_scope=full_scope)
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(staging, target)
        published = True
        return load_labram_peft_prefix_cache(target, expected_manifest_sha256=loaded.manifest_sha256, require_full_scope=full_scope)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "EXPECTED_SOURCE_TRAIN_EVENT_COUNT",
    "EXPECTED_SOURCE_TRAIN_EVENT_ORDER_SHA256",
    "EXPECTED_SOURCE_TRAIN_IV_MANIFEST_SHA256",
    "EXPECTED_SOURCE_TRAIN_IV_RECEIPT_SHA256",
    "EXPECTED_SOURCE_TRAIN_PATIENT_COUNT",
    "LABRAM_PEFT_PREFIX_CACHE_FULL_SCHEMA",
    "LABRAM_PEFT_PREFIX_CACHE_SMOKE_SCHEMA",
    "LABRAM_PEFT_PREFIX_EVENT_SHAPE",
    "LaBraMPEFTPrefixEvent",
    "LoadedLaBraMPEFTPrefixCache",
    "load_labram_peft_prefix_cache",
    "prefix_event_tensor_sha256",
    "prefix_implementation_sha256",
    "publish_labram_peft_prefix_cache",
]
