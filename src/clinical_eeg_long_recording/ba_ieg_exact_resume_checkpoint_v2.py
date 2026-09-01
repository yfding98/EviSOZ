"""Deterministic, content-bound exact-resume checkpoints for BA-IEG v2.

The checkpoint stores the complete composite model/head state, optimizer,
scheduler, Python/NumPy/Torch CPU and CUDA RNG states, patient-token sampler
state, epoch/step/cursor, and the frozen disk manifest identity.  A small
deterministic ZIP codec is used instead of pickle so identical state produces
byte-identical files and loading never executes serialized Python objects.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Final, Mapping
import zipfile

import numpy as np
import torch

from .ba_ieg_permission_split_segmental_disk_training_v1 import (
    BAIEGPatientTokenBucketBatchSamplerV1,
    BAIEGSegmentalDiskBatchV1,
    BAIEGSegmentalDiskDatasetV1,
    collate_ba_ieg_segmental_disk_patient_bags_v1,
)
from .ba_ieg_permission_split_segmental_disk_training_v2 import (
    BA_IEG_V2_DISK_COMPOSITE_ID,
    BAIEGPermissionSplitSegmentalCompositeTrainerV2,
)


BA_IEG_V2_EXACT_RESUME_CHECKPOINT_SCHEMA: Final[str] = (
    "ba_ieg_v2_full_state_deterministic_exact_resume_checkpoint_v1"
)
_ZIP_TIMESTAMP: Final[tuple[int, int, int, int, int, int]] = (1980, 1, 1, 0, 0, 0)
_MAX_CHECKPOINT_BYTES: Final[int] = 16 * 1024 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _state_digest(value: object) -> str:
    def project(item: object) -> object:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return {
                "type": "torch_tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
            }
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return {
                "type": "numpy_array",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        if isinstance(item, dict):
            rows = [(project(key), project(subvalue)) for key, subvalue in item.items()]
            rows.sort(key=lambda row: _canonical_json(row[0]))
            return {"type": "dict", "items": rows}
        if isinstance(item, tuple):
            return {"type": "tuple", "items": [project(value) for value in item]}
        if isinstance(item, list):
            return {"type": "list", "items": [project(value) for value in item]}
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("checkpoint state contains a non-finite scalar")
            return {"type": "float", "hex": item.hex()}
        raise TypeError(f"unsupported exact-resume state type: {type(item)!r}")

    return hashlib.sha256(_canonical_json(project(value))).hexdigest()


def _encode_tree(value: object) -> tuple[object, tuple[bytes, ...]]:
    blobs: list[bytes] = []

    def encode(item: object) -> object:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            index = len(blobs)
            blobs.append(_tensor_bytes(tensor))
            return {
                "type": "torch_tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "blob": index,
            }
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            index = len(blobs)
            blobs.append(array.tobytes(order="C"))
            return {
                "type": "numpy_array",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "blob": index,
            }
        if isinstance(item, dict):
            # Sort before visiting values: visiting a tensor allocates its blob
            # index, so sorting only after recursion would preserve a hidden
            # dependency on Python/OrderedDict insertion order.
            ordered = sorted(
                item.items(), key=lambda row: _state_digest(row[0])
            )
            rows = [(encode(key), encode(subvalue)) for key, subvalue in ordered]
            return {"type": "dict", "items": rows}
        if isinstance(item, tuple):
            return {"type": "tuple", "items": [encode(value) for value in item]}
        if isinstance(item, list):
            return {"type": "list", "items": [encode(value) for value in item]}
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("checkpoint state contains a non-finite scalar")
            return {"type": "float", "hex": item.hex()}
        raise TypeError(f"unsupported exact-resume state type: {type(item)!r}")

    return encode(value), tuple(blobs)


_TORCH_DTYPES: Final[dict[str, torch.dtype]] = {
    str(value): value
    for value in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


def _decode_tree(metadata: object, blobs: tuple[bytes, ...]) -> object:
    def decode(item: object) -> object:
        if not isinstance(item, dict) or "type" not in item:
            if item is None or isinstance(item, (bool, int, str)):
                return item
            raise ValueError("exact-resume metadata contains an untyped object")
        kind = item["type"]
        if kind == "float":
            return float.fromhex(item["hex"])
        if kind in {"torch_tensor", "numpy_array"}:
            if set(item) != {"type", "dtype", "shape", "blob"}:
                raise ValueError("exact-resume tensor metadata fields drifted")
            blob_index = item["blob"]
            if (
                isinstance(blob_index, bool)
                or not isinstance(blob_index, int)
                or blob_index < 0
                or blob_index >= len(blobs)
            ):
                raise ValueError("exact-resume tensor blob index is invalid")
            shape = tuple(item["shape"])
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape):
                raise ValueError("exact-resume tensor shape is invalid")
            raw = blobs[blob_index]
            if kind == "numpy_array":
                dtype = np.dtype(item["dtype"])
                expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if len(raw) != expected:
                    raise ValueError("exact-resume NumPy blob size drifted")
                return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
            dtype = _TORCH_DTYPES.get(item["dtype"])
            if dtype is None:
                raise ValueError("exact-resume Torch dtype is unsupported")
            element_size = torch.empty((), dtype=dtype).element_size()
            expected = math.prod(shape) * element_size
            if len(raw) != expected:
                raise ValueError("exact-resume Torch blob size drifted")
            if expected == 0:
                return torch.empty(shape, dtype=dtype)
            return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
        if kind in {"tuple", "list"}:
            if set(item) != {"type", "items"} or not isinstance(item["items"], list):
                raise ValueError("exact-resume sequence metadata drifted")
            values = [decode(value) for value in item["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind == "dict":
            if set(item) != {"type", "items"} or not isinstance(item["items"], list):
                raise ValueError("exact-resume mapping metadata drifted")
            result: dict[object, object] = {}
            for row in item["items"]:
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError("exact-resume mapping row drifted")
                key, value = decode(row[0]), decode(row[1])
                if key in result:
                    raise ValueError("exact-resume mapping repeats a key")
                result[key] = value
            return result
        raise ValueError("exact-resume metadata has an unknown type")

    return decode(metadata)


def _zip_entry(name: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    return info, data


def _serialize_payload(payload: Mapping[str, Any]) -> bytes:
    metadata, blobs = _encode_tree(dict(payload))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        info, data = _zip_entry("metadata.json", _canonical_json(metadata))
        archive.writestr(info, data)
        for index, blob in enumerate(blobs):
            info, data = _zip_entry(f"blobs/{index:08d}.bin", blob)
            archive.writestr(info, data)
    result = stream.getvalue()
    if len(result) > _MAX_CHECKPOINT_BYTES:
        raise ValueError("exact-resume checkpoint exceeds the maximum size")
    return result


def _deserialize_payload(data: bytes) -> dict[str, Any]:
    if not data or len(data) > _MAX_CHECKPOINT_BYTES:
        raise ValueError("exact-resume checkpoint size is invalid")
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
        names = archive.namelist()
        if not names or names[0] != "metadata.json" or len(names) != len(set(names)):
            raise ValueError("exact-resume ZIP inventory drifted")
        if any(
            name.startswith("/") or ".." in Path(name).parts or "\\" in name
            for name in names
        ):
            raise ValueError("exact-resume ZIP has an unsafe member")
        expected_blobs = [f"blobs/{index:08d}.bin" for index in range(len(names) - 1)]
        if names[1:] != expected_blobs:
            raise ValueError("exact-resume blob roster is not canonical")
        if any(info.compress_type != zipfile.ZIP_STORED for info in archive.infolist()):
            raise ValueError("exact-resume ZIP compression drifted")
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
        blobs = tuple(archive.read(name) for name in expected_blobs)
    payload = _decode_tree(metadata, blobs)
    if type(payload) is not dict:
        raise ValueError("exact-resume root payload must be an object")
    return payload


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "cuda_available": torch.cuda.is_available(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "cuda_available"}
    if type(state) is not dict or set(state) != required:
        raise ValueError("exact-resume RNG state fields drifted")
    if bool(state["cuda_available"]) != torch.cuda.is_available():
        raise ValueError("exact-resume CUDA availability changed")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        current_devices = torch.cuda.device_count()
        if len(state["torch_cuda"]) != current_devices:
            raise ValueError("exact-resume CUDA device roster changed")
        torch.cuda.set_rng_state_all(list(state["torch_cuda"]))


def _sampler_state(sampler: BAIEGPatientTokenBucketBatchSamplerV1) -> dict[str, Any]:
    if not isinstance(sampler, BAIEGPatientTokenBucketBatchSamplerV1):
        raise TypeError("exact resume requires the registered patient-token sampler")
    return {
        "class": type(sampler).__name__,
        "epoch": sampler._epoch,
        "seed": sampler._seed,
        "shuffle": sampler._shuffle,
        "maximum_tokens": sampler._maximum_tokens,
        "maximum_patients": sampler._maximum_patients,
        "bucket_multiplier": sampler._bucket_multiplier,
        "dataset_manifest_id": sampler._dataset.manifest_id,
        "dataset_manifest_file_sha256": sampler._dataset.manifest_file_sha256,
        "patient_uids": sampler._dataset.patient_uids,
        "patient_event_token_counts": sampler._dataset.patient_event_token_counts,
    }


def _restore_sampler_state(
    sampler: BAIEGPatientTokenBucketBatchSamplerV1, state: Mapping[str, Any]
) -> None:
    expected = _sampler_state(sampler)
    if type(state) is not dict or set(state) != set(expected):
        raise ValueError("exact-resume sampler fields drifted")
    for name, value in expected.items():
        if name == "epoch":
            continue
        if state[name] != value:
            raise ValueError(f"exact-resume sampler static state changed: {name}")
    sampler.set_epoch(state["epoch"])


def remaining_ba_ieg_v2_sampler_batches(
    trainer: BAIEGPermissionSplitSegmentalCompositeTrainerV2,
    sampler: BAIEGPatientTokenBucketBatchSamplerV1,
) -> tuple[tuple[int, ...], ...]:
    """Return the exact stored-cursor suffix; callers never guess skip count."""

    if not isinstance(trainer, BAIEGPermissionSplitSegmentalCompositeTrainerV2):
        raise TypeError("remaining-order replay requires the v2 trainer")
    if not isinstance(sampler, BAIEGPatientTokenBucketBatchSamplerV1):
        raise TypeError("remaining-order replay requires the patient-token sampler")
    if (
        sampler._dataset.manifest_id != trainer.manifest_id
        or sampler._dataset.manifest_file_sha256 != trainer.manifest_file_sha256
        or sampler._dataset.candidate_roster_receipt_sha256
        != trainer.candidate_roster_receipt_sha256
    ):
        raise ValueError("exact-resume sampler crosses trainer manifest/roster")
    if sampler._epoch != trainer.epoch:
        raise ValueError("exact-resume sampler epoch differs from trainer epoch")
    full = tuple(tuple(int(index) for index in row) for row in sampler)
    cursor = trainer.batches_consumed_in_epoch
    if (
        isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or cursor < 0
        or cursor > len(full)
    ):
        raise ValueError("exact-resume batch cursor lies outside sampler order")
    return full[cursor:]


def iter_ba_ieg_v2_remaining_disk_batches(
    trainer: BAIEGPermissionSplitSegmentalCompositeTrainerV2,
    dataset: BAIEGSegmentalDiskDatasetV1,
    sampler: BAIEGPatientTokenBucketBatchSamplerV1,
):
    """Yield registered disk batches from the stored exact-resume cursor."""

    if not isinstance(dataset, BAIEGSegmentalDiskDatasetV1):
        raise TypeError("remaining disk iterator requires a segmental dataset")
    if dataset is not sampler._dataset:
        raise ValueError("remaining disk iterator dataset is not the sampler dataset")
    for indices in remaining_ba_ieg_v2_sampler_batches(trainer, sampler):
        yield collate_ba_ieg_segmental_disk_patient_bags_v1(
            tuple(dataset[index] for index in indices)
        )


def _data_order_state(
    trainer: BAIEGPermissionSplitSegmentalCompositeTrainerV2,
    sampler: BAIEGPatientTokenBucketBatchSamplerV1,
) -> dict[str, Any]:
    full = tuple(tuple(int(index) for index in row) for row in sampler)
    remaining = remaining_ba_ieg_v2_sampler_batches(trainer, sampler)
    return {
        "epoch": trainer.epoch,
        "batches_consumed_in_epoch": trainer.batches_consumed_in_epoch,
        "full_batch_index_order": full,
        "remaining_batch_index_order": remaining,
        "full_order_receipt_sha256": hashlib.sha256(_canonical_json(full)).hexdigest(),
        "remaining_order_receipt_sha256": hashlib.sha256(
            _canonical_json(remaining)
        ).hexdigest(),
    }


def build_ba_ieg_v2_exact_resume_payload(
    trainer: BAIEGPermissionSplitSegmentalCompositeTrainerV2,
    *,
    sampler: BAIEGPatientTokenBucketBatchSamplerV1,
) -> dict[str, Any]:
    if not isinstance(trainer, BAIEGPermissionSplitSegmentalCompositeTrainerV2):
        raise TypeError("exact-resume payload requires the v2 composite trainer")
    core = {
        "schema_version": BA_IEG_V2_EXACT_RESUME_CHECKPOINT_SCHEMA,
        "implementation_id": BA_IEG_V2_DISK_COMPOSITE_ID,
        "manifest": {
            "manifest_id": trainer.manifest_id,
            "manifest_file_sha256": trainer.manifest_file_sha256,
            "candidate_roster_receipt_sha256": trainer.candidate_roster_receipt_sha256,
            "prediction_roster_receipt_sha256": trainer.prediction_roster_receipt_sha256,
            "acquisition_support_lineage_receipt_sha256": (
                trainer.acquisition_support_lineage_receipt_sha256
            ),
            "stable_origin_registry_receipt_sha256": (
                trainer.stable_origin_registry_receipt_sha256
            ),
            "training_authority_receipt_sha256": (
                trainer.training_authority_receipt_sha256
            ),
        },
        "trainer_cursor": {
            "epoch": trainer.epoch,
            "optimizer_step": trainer.optimizer_step,
            "batches_consumed_in_epoch": trainer.batches_consumed_in_epoch,
        },
        "k3_policy_receipt_sha256": trainer.model.k3_policy.receipt_sha256,
        "loss_weights": trainer.loss_weights,
        "maximum_gradient_norm": trainer.maximum_gradient_norm,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "scheduler_present": trainer.scheduler is not None,
        "scheduler_state": (
            None if trainer.scheduler is None else trainer.scheduler.state_dict()
        ),
        "rng_state": _rng_state(),
        "sampler_state": _sampler_state(sampler),
        "data_order_state": _data_order_state(trainer, sampler),
    }
    receipt = _state_digest(core)
    return {
        "core": core,
        "checkpoint_receipt_sha256": receipt,
        "content_address_semantics": "sha256_of_typed_tree_and_raw_tensor_bytes",
    }


def validate_ba_ieg_v2_exact_resume_payload(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != {
        "core",
        "checkpoint_receipt_sha256",
        "content_address_semantics",
    }:
        raise ValueError("exact-resume payload fields drifted")
    core = payload["core"]
    if type(core) is not dict or core.get("schema_version") != BA_IEG_V2_EXACT_RESUME_CHECKPOINT_SCHEMA:
        raise ValueError("exact-resume checkpoint schema drifted")
    if core.get("implementation_id") != BA_IEG_V2_DISK_COMPOSITE_ID:
        raise ValueError("exact-resume implementation identity drifted")
    if payload["content_address_semantics"] != "sha256_of_typed_tree_and_raw_tensor_bytes":
        raise ValueError("exact-resume content-address semantics drifted")
    expected = _state_digest(core)
    if payload["checkpoint_receipt_sha256"] != expected:
        raise ValueError("exact-resume checkpoint receipt does not replay")
    return payload


def save_ba_ieg_v2_exact_resume_checkpoint(
    path: str | Path,
    trainer: BAIEGPermissionSplitSegmentalCompositeTrainerV2,
    *,
    sampler: BAIEGPatientTokenBucketBatchSamplerV1,
) -> dict[str, str]:
    payload = build_ba_ieg_v2_exact_resume_payload(trainer, sampler=sampler)
    validate_ba_ieg_v2_exact_resume_payload(payload)
    data = _serialize_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if existing != data:
            raise FileExistsError("exact-resume checkpoint path has different bytes")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {
        "checkpoint_receipt_sha256": payload["checkpoint_receipt_sha256"],
        "checkpoint_file_sha256": hashlib.sha256(data).hexdigest(),
    }


def load_ba_ieg_v2_exact_resume_checkpoint(
    path: str | Path,
    trainer: BAIEGPermissionSplitSegmentalCompositeTrainerV2,
    *,
    sampler: BAIEGPatientTokenBucketBatchSamplerV1,
) -> dict[str, str]:
    if not isinstance(trainer, BAIEGPermissionSplitSegmentalCompositeTrainerV2):
        raise TypeError("exact-resume load requires the v2 composite trainer")
    source = Path(path)
    data = source.read_bytes()
    payload = validate_ba_ieg_v2_exact_resume_payload(_deserialize_payload(data))
    core = payload["core"]
    expected_manifest = {
        "manifest_id": trainer.manifest_id,
        "manifest_file_sha256": trainer.manifest_file_sha256,
        "candidate_roster_receipt_sha256": trainer.candidate_roster_receipt_sha256,
        "prediction_roster_receipt_sha256": trainer.prediction_roster_receipt_sha256,
        "acquisition_support_lineage_receipt_sha256": (
            trainer.acquisition_support_lineage_receipt_sha256
        ),
        "stable_origin_registry_receipt_sha256": (
            trainer.stable_origin_registry_receipt_sha256
        ),
        "training_authority_receipt_sha256": (
            trainer.training_authority_receipt_sha256
        ),
    }
    if core["manifest"] != expected_manifest:
        raise ValueError("exact-resume checkpoint crosses disk manifest/roster")
    if (
        core["k3_policy_receipt_sha256"] != trainer.model.k3_policy.receipt_sha256
        or tuple(core["loss_weights"]) != trainer.loss_weights
        or core["maximum_gradient_norm"] != trainer.maximum_gradient_norm
    ):
        raise ValueError("exact-resume model/loss policy changed")
    if bool(core["scheduler_present"]) != (trainer.scheduler is not None):
        raise ValueError("exact-resume scheduler presence changed")
    trainer.model.load_state_dict(core["model_state"], strict=True)
    trainer.optimizer.load_state_dict(core["optimizer_state"])
    if trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(core["scheduler_state"])
    _restore_sampler_state(sampler, core["sampler_state"])
    cursor = core["trainer_cursor"]
    if type(cursor) is not dict or set(cursor) != {
        "epoch",
        "optimizer_step",
        "batches_consumed_in_epoch",
    }:
        raise ValueError("exact-resume trainer cursor fields drifted")
    trainer.epoch = int(cursor["epoch"])
    trainer.optimizer_step = int(cursor["optimizer_step"])
    trainer.batches_consumed_in_epoch = int(cursor["batches_consumed_in_epoch"])
    replayed_order = _data_order_state(trainer, sampler)
    if core.get("data_order_state") != replayed_order:
        raise ValueError("exact-resume sampler full/remaining order does not replay")
    _restore_rng_state(core["rng_state"])
    return {
        "checkpoint_receipt_sha256": payload["checkpoint_receipt_sha256"],
        "checkpoint_file_sha256": hashlib.sha256(data).hexdigest(),
    }


__all__ = [
    "BA_IEG_V2_EXACT_RESUME_CHECKPOINT_SCHEMA",
    "build_ba_ieg_v2_exact_resume_payload",
    "load_ba_ieg_v2_exact_resume_checkpoint",
    "iter_ba_ieg_v2_remaining_disk_batches",
    "remaining_ba_ieg_v2_sampler_batches",
    "save_ba_ieg_v2_exact_resume_checkpoint",
    "validate_ba_ieg_v2_exact_resume_payload",
]
