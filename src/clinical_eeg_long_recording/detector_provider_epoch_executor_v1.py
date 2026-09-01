"""Actual-byte disk epoch and exact-resume executor for clean-room detectors.

This additive module closes a software-path gap shared by the independent
EventNet EN19/EN17 and SeizureTransformer ST18/ST16 variants.  It does *not*
materialize a trained checkpoint or make a performance claim.  Its formal
inputs remain the process-sealed fold-phase, variant-roster, record-pool and
target authorities issued by the provider registries.

The execution boundary is deliberately strict:

* a disk bundle is written from opaque provider objects and admitted only
  after every referenced safetensors byte is replayed against those objects;
* an epoch plan is regenerated from the complete opaque record-pool roster,
  so caller-owned subsets or reordered batches are not sampler authority;
* model input is EEG only; patient keys and targets remain loss/sampler
  control plane and never enter ``model.forward``;
* a checkpoint is emitted only after a complete epoch and stores tensor-only
  model bytes plus a deterministic, non-pickle optimizer/RNG archive;
* same-process resume reloads and replays phase, variant roster, epoch sampler,
  disk manifest, model, optimizer and RNG bytes while retaining the opaque
  completed-epoch seal; cross-process formal resume is not admitted in v1.

The private conformance-fixture runtime exists solely to exercise the executor
on CPU in unit tests.  Its receipts are permanently non-promotable and formal
checkpoint publication rejects it by default.  No synthetic execution is
presented as real EEG training.  The current materializer also retains all
opaque full-record sources while constructing safetensors, so it is a software
closure fixture rather than a scalable whole-fold streaming materializer.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import tempfile
from typing import Any, Final, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from safetensors.torch import load as load_safetensors_bytes
from safetensors.torch import save as save_safetensors_bytes
import torch
from torch import Tensor, nn
import torch.nn.functional as torch_functional

from . import eventnet_cleanroom_registry_v1 as _eventnet
from . import seizuretransformer_cleanroom_registry_v1 as _st


SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_cleanroom_provider_disk_epoch_exact_resume_executor_v1"
)
EXECUTOR_ID: Final[str] = (
    "CLINICAL-EEG-CLEANROOM-PROVIDER-DISK-EPOCH-EXACT-RESUME-V1-20260824"
)
DISK_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_cleanroom_provider_training_disk_bundle_v1"
)
EPOCH_EXECUTION_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_cleanroom_provider_completed_epoch_execution_v1"
)
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_cleanroom_provider_completed_epoch_checkpoint_v1"
)

EVENTNET_PROVIDER_FAMILY: Final[str] = "eventnet"
SEIZURETRANSFORMER_PROVIDER_FAMILY: Final[str] = "seizuretransformer"
PROVIDER_FAMILIES: Final[frozenset[str]] = frozenset(
    {EVENTNET_PROVIDER_FAMILY, SEIZURETRANSFORMER_PROVIDER_FAMILY}
)

_CONTENT_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
_ZIP_TIMESTAMP: Final[tuple[int, int, int, int, int, int]] = (
    1980,
    1,
    1,
    0,
    0,
    0,
)
_MAX_SAFE_TREE_BYTES: Final[int] = 16 * 1024 * 1024 * 1024
_MAX_ARTIFACT_BYTES: Final[int] = 16 * 1024 * 1024 * 1024
_DISK_AUTHORITY_SEAL = object()
_EPOCH_AUTHORITY_SEAL = object()
_RUNTIME_AUTHORITY_SEAL = object()
_COMPLETED_EPOCH_SEAL = object()
_CHECKPOINT_AUTHORITY_SEAL = object()


@dataclass(frozen=True)
class EventNetAuthorizedDiskRecordV1:
    """One complete eligible EventNet record and all trainable target tiles."""

    transform_result: _eventnet.EventNetTransformResult
    record_pool_authority: _eventnet.AuthorizedEventNetRecordPool
    target_bundles: tuple[_eventnet.AuthorizedEventNetTargetBundle, ...]


@dataclass(frozen=True)
class SeizureTransformerAuthorizedDiskRecordV1:
    """One complete eligible ST record and all trainable target tiles."""

    transform_result: _st.SeizureTransformerTransformResult
    record_pool_authority: _st.AuthorizedSeizureTransformerRecordPool
    target_bundles: tuple[_st.AuthorizedSeizureTransformerTargetBundle, ...]


@dataclass(frozen=True)
class AdmittedProviderTrainingDiskBundleV1:
    """Opaque actual-byte-replayed provider training bundle."""

    _root: str
    _manifest_json: str
    _phase_snapshot_json: str
    _roster_snapshot_json: str
    _admission_receipt_json: str
    _phase_authority: object
    _variant_roster_authority: object
    _record_sources: tuple[object, ...]
    _target_by_tile: Mapping[str, object]
    _validation_seal: object

    @property
    def root(self) -> Path:
        return Path(self._root)

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._admission_receipt_json)


@dataclass(frozen=True)
class AuthorizedProviderEpochExecutionV1:
    """Opaque complete sampler plan bound to one admitted disk bundle."""

    disk_bundle: AdmittedProviderTrainingDiskBundleV1
    _epoch_plan_json: str
    _class_weight_authority: object | None
    _receipt_json: str
    _validation_seal: object

    @property
    def epoch_plan(self) -> dict[str, Any]:
        return json.loads(self._epoch_plan_json)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class ProviderModelRuntimeAuthorityV1:
    """Process-local model/optimizer/RNG authority for one provider run."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    _rng_state: dict[str, Any]
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class CompletedProviderEpochV1:
    """One fully consumed epoch; partial epochs cannot construct this type."""

    epoch_authority: AuthorizedProviderEpochExecutionV1
    runtime_authority: ProviderModelRuntimeAuthorityV1
    _receipt_json: str
    _validation_seal: object

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class AdmittedProviderEpochCheckpointV1:
    """Opaque checkpoint returned only after all bundle bytes replay."""

    _root: str
    disk_bundle: AdmittedProviderTrainingDiskBundleV1
    runtime_authority: ProviderModelRuntimeAuthorityV1
    _manifest_json: str
    _receipt_json: str
    _validation_seal: object

    @property
    def root(self) -> Path:
        return Path(self._root)

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def detector_provider_epoch_executor_source_sha256_v1() -> str:
    return _bytes_sha256(Path(__file__).read_bytes())


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _CONTENT_PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(value: object, *, context: str) -> dict[str, Any]:
    if type(value) is not dict or "receipt_sha256" not in value:
        raise ValueError(f"{context} must be a content-addressed object")
    result = deepcopy(value)
    supplied = result["receipt_sha256"]
    result["receipt_sha256"] = _CONTENT_PENDING
    if supplied != _canonical_sha256(result):
        raise ValueError(f"{context} content address drifted")
    result["receipt_sha256"] = supplied
    return result


def _provider_family_for_variant(variant_id: str) -> str:
    if variant_id in {_eventnet.EN19_VARIANT_ID, _eventnet.EN17_VARIANT_ID}:
        return EVENTNET_PROVIDER_FAMILY
    if variant_id in {_st.ST18_VARIANT_ID, _st.ST16_VARIANT_ID}:
        return SEIZURETRANSFORMER_PROVIDER_FAMILY
    raise ValueError("unknown clean-room detector variant")


def _expected_channel_count(variant_id: str) -> int:
    return {
        _eventnet.EN19_VARIANT_ID: 19,
        _eventnet.EN17_VARIANT_ID: 17,
        _st.ST18_VARIANT_ID: 18,
        _st.ST16_VARIANT_ID: 16,
    }[variant_id]


def _provider_context(
    phase_authority: object, variant_roster_authority: object
) -> tuple[str, dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay provider-private opaque authorities without accepting mappings."""

    if isinstance(phase_authority, _eventnet.AuthorizedEventNetFoldPhase):
        phase, patients, phase_receipt = (
            _eventnet._require_authorized_eventnet_fold_phase(phase_authority)
        )
        roster, roster_receipt = (
            _eventnet._require_authorized_eventnet_variant_training_roster(
                variant_roster_authority
            )
        )
        family = EVENTNET_PROVIDER_FAMILY
    elif isinstance(
        phase_authority, _st.AuthorizedSeizureTransformerFoldPhase
    ):
        phase, patients, phase_receipt = (
            _st._require_authorized_seizuretransformer_fold_phase(phase_authority)
        )
        roster, roster_receipt = _st._require_authorized_st_variant_training_roster(
            variant_roster_authority
        )
        family = SEIZURETRANSFORMER_PROVIDER_FAMILY
    else:
        raise TypeError(
            "provider disk execution requires an opaque EventNet or "
            "SeizureTransformer fold-phase authority"
        )
    if (
        roster_receipt["registry_sha256"] != phase_receipt["registry_sha256"]
        or roster_receipt["outer_fold"] != phase_receipt["outer_fold"]
        or roster_receipt["phase"] != phase_receipt["phase"]
        or roster_receipt["detector_fold_phase_receipt_sha256"]
        != phase_receipt["detector_fold_phase_receipt_sha256"]
    ):
        raise PermissionError("phase and variant-roster authorities cross a fold phase")
    if _provider_family_for_variant(str(roster_receipt["variant_id"])) != family:
        raise PermissionError("variant roster belongs to a different provider family")
    return family, phase, patients, phase_receipt, roster, roster_receipt


def _authority_snapshots(
    phase_authority: object, variant_roster_authority: object
) -> tuple[bytes, bytes, tuple[Any, ...]]:
    context = _provider_context(phase_authority, variant_roster_authority)
    family, phase, patients, phase_receipt, roster, roster_receipt = context
    phase_bytes = _canonical_json_bytes(
        {
            "schema_version": "provider_fold_phase_authority_snapshot_v1",
            "provider_family": family,
            "phase_payload": phase,
            "fold_owned_patient_mapping": patients,
            "authority_receipt": phase_receipt,
        }
    )
    roster_bytes = _canonical_json_bytes(
        {
            "schema_version": "provider_variant_roster_authority_snapshot_v1",
            "provider_family": family,
            "roster_payload": roster,
            "authority_receipt": roster_receipt,
        }
    )
    return phase_bytes, roster_bytes, context


def _tensor_payload_receipt(value: Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    raw = (
        b""
        if tensor.numel() == 0
        else tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "payload_sha256": _bytes_sha256(raw),
    }


def _model_state_bytes(model: nn.Module) -> bytes:
    state: dict[str, Tensor] = {}
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("model state must be a string-to-tensor mapping")
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("model state contains nonfinite values")
        state[name] = tensor
    if not state:
        raise ValueError("model state may not be empty")
    return bytes(save_safetensors_bytes(state))


def _model_state_ledger(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, **_tensor_payload_receipt(value)}
        for name, value in sorted(model.state_dict().items())
    ]


def _state_digest(value: object) -> str:
    """Digest a typed state tree without pickle or lossy scalar conversion."""

    def project(item: object) -> object:
        if isinstance(item, Tensor):
            return {"type": "torch_tensor", **_tensor_payload_receipt(item)}
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return {
                "type": "numpy_array",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "payload_sha256": _bytes_sha256(array.tobytes(order="C")),
            }
        if isinstance(item, Mapping):
            rows = [(project(key), project(value)) for key, value in item.items()]
            rows.sort(key=lambda row: _canonical_json_bytes(row[0]))
            return {"type": "mapping", "items": rows}
        if isinstance(item, tuple):
            return {"type": "tuple", "items": [project(value) for value in item]}
        if isinstance(item, list):
            return {"type": "list", "items": [project(value) for value in item]}
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("state tree contains a nonfinite scalar")
            return {"type": "float", "hex": item.hex()}
        raise TypeError(f"unsupported deterministic state type: {type(item)!r}")

    return _canonical_sha256(project(value))


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


def _encode_state_tree(value: object) -> tuple[object, tuple[bytes, ...]]:
    blobs: list[bytes] = []

    def encode(item: object) -> object:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            raw = (
                b""
                if tensor.numel() == 0
                else tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            )
            index = len(blobs)
            blobs.append(raw)
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
        if isinstance(item, Mapping):
            ordered = sorted(item.items(), key=lambda row: _state_digest(row[0]))
            return {
                "type": "mapping",
                "items": [(encode(key), encode(value)) for key, value in ordered],
            }
        if isinstance(item, tuple):
            return {"type": "tuple", "items": [encode(value) for value in item]}
        if isinstance(item, list):
            return {"type": "list", "items": [encode(value) for value in item]}
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("state tree contains a nonfinite scalar")
            return {"type": "float", "hex": item.hex()}
        raise TypeError(f"unsupported deterministic state type: {type(item)!r}")

    return encode(value), tuple(blobs)


def _decode_state_tree(metadata: object, blobs: tuple[bytes, ...]) -> object:
    referenced_blobs: list[int] = []

    def decode(item: object) -> object:
        if not isinstance(item, dict) or "type" not in item:
            if item is None or isinstance(item, (bool, int, str)):
                return item
            raise ValueError("state metadata contains an untyped object")
        kind = item["type"]
        if kind == "float":
            if set(item) != {"type", "hex"}:
                raise ValueError("state float metadata drifted")
            value = float.fromhex(item["hex"])
            if not math.isfinite(value):
                raise ValueError("decoded state float is nonfinite")
            return value
        if kind in {"torch_tensor", "numpy_array"}:
            if set(item) != {"type", "dtype", "shape", "blob"}:
                raise ValueError("state tensor metadata drifted")
            blob_index = item["blob"]
            if (
                isinstance(blob_index, bool)
                or not isinstance(blob_index, int)
                or not 0 <= blob_index < len(blobs)
            ):
                raise ValueError("state tensor blob index is invalid")
            shape = tuple(item["shape"])
            if any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            ):
                raise ValueError("state tensor shape is invalid")
            raw = blobs[blob_index]
            referenced_blobs.append(blob_index)
            if kind == "numpy_array":
                dtype = np.dtype(item["dtype"])
                if (
                    dtype.hasobject
                    or dtype.fields is not None
                    or dtype.subdtype is not None
                    or dtype.kind not in {"b", "i", "u", "f"}
                ):
                    raise ValueError("NumPy state dtype is not allowlisted")
                expected = math.prod(shape) * dtype.itemsize
                if len(raw) != expected:
                    raise ValueError("NumPy state blob size drifted")
                return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
            dtype = _TORCH_DTYPES.get(item["dtype"])
            if dtype is None:
                raise ValueError("Torch state dtype is unsupported")
            expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if len(raw) != expected:
                raise ValueError("Torch state blob size drifted")
            if expected == 0:
                return torch.empty(shape, dtype=dtype)
            return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
        if kind in {"tuple", "list"}:
            if set(item) != {"type", "items"} or not isinstance(item["items"], list):
                raise ValueError("state sequence metadata drifted")
            values = [decode(value) for value in item["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind == "mapping":
            if set(item) != {"type", "items"} or not isinstance(item["items"], list):
                raise ValueError("state mapping metadata drifted")
            result: dict[object, object] = {}
            for row in item["items"]:
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError("state mapping row drifted")
                key, value = decode(row[0]), decode(row[1])
                if key in result:
                    raise ValueError("state mapping repeats a key")
                result[key] = value
            return result
        raise ValueError("state metadata has an unknown type")

    result = decode(metadata)
    if sorted(referenced_blobs) != list(range(len(blobs))):
        raise ValueError("state blob inventory is unused or multiply referenced")
    return result


def _zip_entry(name: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    return info, data


def _serialize_state_tree(value: object) -> bytes:
    metadata, blobs = _encode_state_tree(value)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        info, data = _zip_entry("metadata.json", _canonical_json_bytes(metadata))
        archive.writestr(info, data)
        for index, blob in enumerate(blobs):
            info, data = _zip_entry(f"blobs/{index:08d}.bin", blob)
            archive.writestr(info, data)
    result = stream.getvalue()
    if not result or len(result) > _MAX_SAFE_TREE_BYTES:
        raise ValueError("deterministic state archive size is invalid")
    return result


def _deserialize_state_tree(data: bytes) -> object:
    if not data or len(data) > _MAX_SAFE_TREE_BYTES:
        raise ValueError("deterministic state archive size is invalid")
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
        names = archive.namelist()
        if not names or names[0] != "metadata.json" or len(names) != len(set(names)):
            raise ValueError("deterministic state ZIP inventory drifted")
        expected = [f"blobs/{index:08d}.bin" for index in range(len(names) - 1)]
        if names[1:] != expected:
            raise ValueError("deterministic state blob roster is not canonical")
        if any(
            info.compress_type != zipfile.ZIP_STORED
            or info.date_time != _ZIP_TIMESTAMP
            for info in archive.infolist()
        ):
            raise ValueError("deterministic state ZIP metadata drifted")
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
        blobs = tuple(archive.read(name) for name in expected)
    return _decode_state_tree(metadata, blobs)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _secure_read_file(
    root: Path,
    relative_path: str,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
        or "\\" in relative_path
    ):
        raise ValueError("artifact path is unsafe")
    path = root / relative_path
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("artifact must be a single-link regular file")
        if before.st_size <= 0 or before.st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError("artifact size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("artifact was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("artifact grew during read")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("artifact changed during read")
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if expected_size is not None and len(data) != expected_size:
        raise ValueError("artifact size differs from manifest")
    if expected_sha256 is not None and _bytes_sha256(data) != expected_sha256:
        raise ValueError("artifact SHA-256 differs from manifest")
    return data


def _strict_new_directory(path: str | Path) -> tuple[Path, Path]:
    destination = Path(path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    return destination, temporary


def _phase_name_to_stage(phase: str) -> str:
    mapping = {"selection_fit": "selection", "final_refit": "final_refit"}
    if phase not in mapping:
        raise PermissionError("provider epoch execution accepts gradient phases only")
    return mapping[phase]


def _validate_transform_payload(
    family: str, transform: object
) -> tuple[np.ndarray, dict[str, Any]]:
    if family == EVENTNET_PROVIDER_FAMILY:
        if (
            not isinstance(transform, _eventnet.EventNetTransformResult)
            or transform._validation_seal is not _eventnet._TRANSFORM_RESULT_SEAL
        ):
            raise TypeError("disk materialization requires an opaque EventNet transform")
        signal = np.asarray(transform.signal_uv)
        receipt = deepcopy(transform.receipt)
        semantic = "EventNet_cleanroom_provider_native_full_record_uV"
        expected = _eventnet._payload_receipt(signal, semantic=semantic)
    else:
        if (
            not isinstance(transform, _st.SeizureTransformerTransformResult)
            or transform._validation_seal is not _st._TRANSFORM_RESULT_SEAL
        ):
            raise TypeError(
                "disk materialization requires an opaque SeizureTransformer transform"
            )
        signal = np.asarray(transform.signal)
        receipt = deepcopy(transform.receipt)
        semantic = "SeizureTransformer_provider_native_full_record"
        expected = _st._payload_receipt(signal, semantic=semantic)
    if (
        signal.dtype != np.dtype("float32")
        or signal.ndim != 2
        or signal.shape[0] != _expected_channel_count(str(receipt.get("variant_id")))
        or signal.shape[1] <= 0
        or not np.isfinite(signal).all()
        or receipt.get("output", {}).get("sample_count") != signal.shape[1]
        or receipt.get("output", {}).get("payload_receipt") != expected
    ):
        raise ValueError("opaque provider transform payload drifted")
    return np.ascontiguousarray(signal), receipt


def _record_source_components(
    family: str, source: object
) -> tuple[np.ndarray, dict[str, Any], object, dict[str, Any], dict[str, Any], tuple[object, ...]]:
    if family == EVENTNET_PROVIDER_FAMILY:
        if not isinstance(source, EventNetAuthorizedDiskRecordV1):
            raise TypeError("EventNet disk bundle received a foreign record source")
        signal, transform_receipt = _validate_transform_payload(
            family, source.transform_result
        )
        pool, pool_receipt = _eventnet._require_authorized_eventnet_record_pool(
            source.record_pool_authority
        )
        targets = tuple(
            _eventnet._require_authorized_eventnet_target_bundle(value)
            for value in source.target_bundles
        )
        return (
            signal,
            transform_receipt,
            source.record_pool_authority,
            pool,
            pool_receipt,
            targets,
        )
    if not isinstance(source, SeizureTransformerAuthorizedDiskRecordV1):
        raise TypeError("SeizureTransformer disk bundle received a foreign record source")
    signal, transform_receipt = _validate_transform_payload(
        family, source.transform_result
    )
    pool, pool_receipt = _st._require_authorized_st_record_pool(
        source.record_pool_authority
    )
    targets = tuple(
        _st._require_authorized_st_target_bundle(value)
        for value in source.target_bundles
    )
    return (
        signal,
        transform_receipt,
        source.record_pool_authority,
        pool,
        pool_receipt,
        targets,
    )


def _record_artifact_and_manifest_row(
    *,
    family: str,
    source: object,
    phase_receipt: Mapping[str, Any],
    roster_receipt: Mapping[str, Any],
    expected_roster_row: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any], dict[str, object]]:
    signal, transform_receipt, _pool_authority, pool, pool_receipt, targets = (
        _record_source_components(family, source)
    )
    identity = str(pool_receipt["analysis_identity_id"])
    patient = str(pool_receipt["fold_owned_patient_key"])
    variant_id = str(roster_receipt["variant_id"])
    common_matches = (
        pool_receipt["registry_sha256"] == phase_receipt["registry_sha256"]
        and pool_receipt["variant_id"] == variant_id
        and pool_receipt["outer_fold"] == phase_receipt["outer_fold"]
        and pool_receipt["phase"] == phase_receipt["phase"]
        and pool_receipt["detector_fold_phase_receipt_sha256"]
        == phase_receipt["detector_fold_phase_receipt_sha256"]
        and pool_receipt["variant_training_roster_receipt_sha256"]
        == roster_receipt["receipt_sha256"]
        and transform_receipt["variant_id"] == variant_id
        and transform_receipt["receipt_sha256"]
        == pool_receipt["transform_receipt_sha256"]
        and identity == expected_roster_row["analysis_identity_id"]
        and patient == expected_roster_row["fold_owned_patient_key"]
        and transform_receipt["receipt_sha256"]
        == expected_roster_row["transform_receipt_sha256"]
    )
    if not common_matches:
        raise PermissionError("record source crosses phase, roster, pool or transform authority")

    if family == EVENTNET_PROVIDER_FAMILY:
        eligible_tile_ids = tuple(pool["positive"] + pool["background"])
        tile_row_by_id = {
            row["tile_id"]: row
            for row in pool["tile_rows"]
            if row["pool"] in {"positive", "background"}
        }
    else:
        eligible_tile_ids = tuple(pool["all"])
        tile_row_by_id = {row["tile_id"]: row for row in pool["tile_rows"]}
    if set(tile_row_by_id) != set(eligible_tile_ids):
        raise ValueError("record pool tile ledger disagrees with its eligible roster")

    target_by_tile: dict[str, object] = {}
    for target in targets:
        receipt = target.receipt
        if (
            receipt["registry_sha256"] != phase_receipt["registry_sha256"]
            or receipt["variant_id"] != variant_id
            or receipt["outer_fold"] != phase_receipt["outer_fold"]
            or receipt["phase"] != phase_receipt["phase"]
            or receipt["detector_fold_phase_receipt_sha256"]
            != phase_receipt["detector_fold_phase_receipt_sha256"]
            or receipt["variant_training_roster_receipt_sha256"]
            != roster_receipt["receipt_sha256"]
            or receipt["analysis_identity_id"] != identity
            or receipt["fold_owned_patient_key"] != patient
            or receipt["transform_receipt_sha256"]
            != transform_receipt["receipt_sha256"]
            or receipt["transform_output_payload_sha256"]
            != transform_receipt["output"]["payload_receipt"]["payload_sha256"]
        ):
            raise PermissionError("target bundle crosses opaque provider authorities")
        start = int(receipt["target_start_sample"])
        if family == EVENTNET_PROVIDER_FAMILY:
            candidates = [
                tile_id
                for tile_id, row in tile_row_by_id.items()
                if int(row["target_start_sample"]) == start
            ]
            model_tile = _eventnet.materialize_model_tile(
                signal, target_start_sample=start
            )
            if (
                model_tile.receipt["receipt_sha256"]
                != receipt["model_tile_receipt_sha256"]
                or model_tile.receipt["model_input_payload_receipt"]["payload_sha256"]
                != receipt["model_input_payload_sha256"]
            ):
                raise ValueError("EventNet disk target is bound to a different model tile")
        else:
            stop = int(receipt["target_stop_sample_exclusive"])
            candidates = [
                tile_id
                for tile_id, row in tile_row_by_id.items()
                if int(row["start_sample"]) == start
                and int(row["stop_sample_exclusive"]) == stop
            ]
        if len(candidates) != 1 or candidates[0] in target_by_tile:
            raise ValueError("target bundle does not map one-to-one onto the record pool")
        target_by_tile[candidates[0]] = target
    if set(target_by_tile) != set(eligible_tile_ids):
        missing = sorted(set(eligible_tile_ids).difference(target_by_tile))
        extra = sorted(set(target_by_tile).difference(eligible_tile_ids))
        raise PermissionError(
            "disk bundle must contain every eligible target tile; "
            f"missing={missing}, extra={extra}"
        )

    tensors: dict[str, Tensor] = {"signal": torch.from_numpy(signal.copy())}
    manifest_tiles: list[dict[str, Any]] = []
    for index, tile_id in enumerate(sorted(target_by_tile)):
        target = target_by_tile[tile_id]
        prefix = f"tile_{index:08d}"
        if family == EVENTNET_PROVIDER_FAMILY:
            keys = {
                "center_target": f"{prefix}.center_target",
                "duration_target": f"{prefix}.duration_target",
                "center_loss_mask": f"{prefix}.center_loss_mask",
                "duration_loss_mask": f"{prefix}.duration_loss_mask",
            }
            tensors[keys["center_target"]] = torch.from_numpy(
                np.asarray(target.center_target).copy()
            )
            tensors[keys["duration_target"]] = torch.from_numpy(
                np.asarray(target.duration_target).copy()
            )
            tensors[keys["center_loss_mask"]] = torch.from_numpy(
                np.asarray(target.center_loss_mask, dtype=np.uint8).copy()
            )
            tensors[keys["duration_loss_mask"]] = torch.from_numpy(
                np.asarray(target.duration_loss_mask, dtype=np.uint8).copy()
            )
            pool_value = tile_row_by_id[tile_id]["pool"]
            distinct_center_count = int(target.distinct_center_count)
        else:
            keys = {
                "target": f"{prefix}.target",
                "observed_mask": f"{prefix}.observed_mask",
            }
            tensors[keys["target"]] = torch.from_numpy(
                np.asarray(target.target).copy()
            )
            tensors[keys["observed_mask"]] = torch.from_numpy(
                np.asarray(target.observed_mask).copy()
            )
            pool_value = "positive" if tile_row_by_id[tile_id]["positive"] else "all"
            distinct_center_count = None
        row = {
            "tile_id": tile_id,
            "patient_key": patient,
            "target_start_sample": int(target.receipt["target_start_sample"]),
            "target_stop_sample_exclusive": int(
                target.receipt["target_stop_sample_exclusive"]
            ),
            "pool": pool_value,
            "target_bundle_receipt_sha256": target.receipt["receipt_sha256"],
            "tensor_keys": keys,
        }
        if distinct_center_count is not None:
            row["distinct_center_count"] = distinct_center_count
        manifest_tiles.append(row)

    artifact = bytes(
        save_safetensors_bytes(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
        )
    )
    artifact_name = f"records/{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.safetensors"
    manifest_row = {
        "analysis_identity_id": identity,
        "fold_owned_patient_key": patient,
        "transform_receipt_sha256": transform_receipt["receipt_sha256"],
        "transform_output_payload_sha256": transform_receipt["output"][
            "payload_receipt"
        ]["payload_sha256"],
        "record_pool_receipt_sha256": pool_receipt["receipt_sha256"],
        "artifact_path": artifact_name,
        "artifact_size_bytes": len(artifact),
        "artifact_sha256": _bytes_sha256(artifact),
        "signal_shape": list(signal.shape),
        "signal_dtype": signal.dtype.str,
        "tiles": manifest_tiles,
    }
    return artifact, manifest_row, target_by_tile


def _assemble_bundle_payload(
    *,
    phase_authority: object,
    variant_roster_authority: object,
    record_sources: Sequence[object],
) -> tuple[dict[str, Any], bytes, bytes, dict[str, bytes], dict[str, object]]:
    phase_bytes, roster_bytes, context = _authority_snapshots(
        phase_authority, variant_roster_authority
    )
    family, _phase, _patients, phase_receipt, roster, roster_receipt = context
    if not isinstance(record_sources, Sequence) or not record_sources:
        raise ValueError("provider disk bundle needs at least one eligible record")
    roster_by_identity = {
        str(row["analysis_identity_id"]): row for row in roster["eligible_records"]
    }
    records: list[dict[str, Any]] = []
    artifact_by_path: dict[str, bytes] = {}
    target_by_tile: dict[str, object] = {}
    for source in record_sources:
        _, _, _, pool, pool_receipt, _ = _record_source_components(family, source)
        identity = str(pool_receipt["analysis_identity_id"])
        if identity not in roster_by_identity:
            raise PermissionError("disk record lies outside the opaque eligible roster")
        if any(row["analysis_identity_id"] == identity for row in records):
            raise ValueError("disk bundle repeats an eligible record")
        artifact, row, targets = _record_artifact_and_manifest_row(
            family=family,
            source=source,
            phase_receipt=phase_receipt,
            roster_receipt=roster_receipt,
            expected_roster_row=roster_by_identity[identity],
        )
        if row["artifact_path"] in artifact_by_path:
            raise ValueError("disk record artifact path collides")
        artifact_by_path[row["artifact_path"]] = artifact
        for tile_id, target in targets.items():
            if tile_id in target_by_tile:
                raise ValueError("disk tile identity collides across records")
            target_by_tile[tile_id] = target
        records.append(row)
    if {row["analysis_identity_id"] for row in records} != set(roster_by_identity):
        missing = sorted(
            set(roster_by_identity).difference(
                row["analysis_identity_id"] for row in records
            )
        )
        raise PermissionError(
            f"complete opaque variant-eligible record denominator was deleted: {missing}"
        )
    records.sort(key=lambda row: row["analysis_identity_id"])
    pool_roster_sha = _canonical_sha256(
        [row["record_pool_receipt_sha256"] for row in records]
    )
    manifest = _content_address(
        {
            "schema_version": DISK_BUNDLE_SCHEMA_VERSION,
            "executor_id": EXECUTOR_ID,
            "executor_source_sha256": detector_provider_epoch_executor_source_sha256_v1(),
            "provider_family": family,
            "variant_id": roster_receipt["variant_id"],
            "registry_sha256": roster_receipt["registry_sha256"],
            "outer_fold": phase_receipt["outer_fold"],
            "stage": _phase_name_to_stage(str(phase_receipt["phase"])),
            "phase": phase_receipt["phase"],
            "detector_fold_phase_receipt_sha256": phase_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "provider_phase_authority_receipt_sha256": phase_receipt[
                "receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": roster_receipt[
                "receipt_sha256"
            ],
            "phase_snapshot_path": "phase_authority.json",
            "phase_snapshot_size_bytes": len(phase_bytes),
            "phase_snapshot_sha256": _bytes_sha256(phase_bytes),
            "roster_snapshot_path": "variant_roster_authority.json",
            "roster_snapshot_size_bytes": len(roster_bytes),
            "roster_snapshot_sha256": _bytes_sha256(roster_bytes),
            "authorized_record_pool_receipt_roster_sha256": pool_roster_sha,
            "record_count": len(records),
            "tile_count": len(target_by_tile),
            "records": records,
            "serialization": "canonical_json_plus_safetensors_no_pickle",
            "model_forward_input_allowlist": ["provider_preprocessed_EEG_tensor"],
            "patient_identity_target_and_reference_used_as_model_features": False,
            "complete_variant_eligible_denominator_materialized": True,
            "whole_fold_streaming_materializer_implemented": False,
            "full_record_sources_retained_process_local": True,
            "large_real_fold_scalability_admitted": False,
            "real_EEG_training_or_accuracy_claim_materialized": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return manifest, phase_bytes, roster_bytes, artifact_by_path, target_by_tile


def materialize_provider_training_disk_bundle_v1(
    output_directory: str | Path,
    *,
    phase_authority: object,
    variant_roster_authority: object,
    record_sources: Sequence[object],
) -> AdmittedProviderTrainingDiskBundleV1:
    """Atomically publish and immediately actual-byte-admit one disk bundle."""

    manifest, phase_bytes, roster_bytes, artifacts, _targets = _assemble_bundle_payload(
        phase_authority=phase_authority,
        variant_roster_authority=variant_roster_authority,
        record_sources=record_sources,
    )
    destination, temporary = _strict_new_directory(output_directory)
    try:
        _atomic_write_bytes(temporary / "phase_authority.json", phase_bytes)
        _atomic_write_bytes(temporary / "variant_roster_authority.json", roster_bytes)
        for relative_path, data in sorted(artifacts.items()):
            _atomic_write_bytes(temporary / relative_path, data)
        manifest_bytes = _canonical_json_bytes(manifest)
        _atomic_write_bytes(temporary / "manifest.json", manifest_bytes)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return admit_provider_training_disk_bundle_v1(
        destination,
        phase_authority=phase_authority,
        variant_roster_authority=variant_roster_authority,
        record_sources=record_sources,
    )


def _validate_loaded_record_tensors(
    *,
    family: str,
    tensors: Mapping[str, Tensor],
    manifest_row: Mapping[str, Any],
    source: object,
) -> None:
    signal, _transform, _pool_authority, _pool, _pool_receipt, targets = (
        _record_source_components(family, source)
    )
    expected_keys = {"signal"}
    target_by_receipt = {
        target.receipt["receipt_sha256"]: target for target in targets
    }
    for tile in manifest_row["tiles"]:
        expected_keys.update(tile["tensor_keys"].values())
    if set(tensors) != expected_keys:
        raise ValueError("record safetensors key roster drifted")
    loaded_signal = tensors["signal"].detach().cpu().numpy()
    if not np.array_equal(loaded_signal, signal):
        raise ValueError("record safetensors EEG differs from opaque transform")
    for tile in manifest_row["tiles"]:
        target = target_by_receipt.get(tile["target_bundle_receipt_sha256"])
        if target is None:
            raise ValueError("record safetensors names an unknown opaque target")
        keys = tile["tensor_keys"]
        if family == EVENTNET_PROVIDER_FAMILY:
            expected = {
                "center_target": np.asarray(target.center_target),
                "duration_target": np.asarray(target.duration_target),
                "center_loss_mask": np.asarray(
                    target.center_loss_mask, dtype=np.uint8
                ),
                "duration_loss_mask": np.asarray(
                    target.duration_loss_mask, dtype=np.uint8
                ),
            }
        else:
            expected = {
                "target": np.asarray(target.target),
                "observed_mask": np.asarray(target.observed_mask),
            }
        for name, value in expected.items():
            loaded = tensors[keys[name]].detach().cpu().numpy()
            if not np.array_equal(loaded, value):
                raise ValueError("record safetensors target differs from opaque bundle")


def admit_provider_training_disk_bundle_v1(
    directory: str | Path,
    *,
    phase_authority: object,
    variant_roster_authority: object,
    record_sources: Sequence[object],
) -> AdmittedProviderTrainingDiskBundleV1:
    """Replay every disk byte against current process-sealed authorities."""

    root = Path(directory).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("provider training disk bundle must be a real directory")
    (
        expected_manifest,
        expected_phase_bytes,
        expected_roster_bytes,
        _expected_artifacts,
        target_by_tile,
    ) = _assemble_bundle_payload(
        phase_authority=phase_authority,
        variant_roster_authority=variant_roster_authority,
        record_sources=record_sources,
    )
    manifest_raw = _secure_read_file(root, "manifest.json")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider disk manifest is unreadable") from exc
    if manifest != expected_manifest or manifest_raw != _canonical_json_bytes(manifest):
        raise ValueError("provider disk manifest differs from opaque authorities")
    phase_raw = _secure_read_file(
        root,
        manifest["phase_snapshot_path"],
        expected_size=manifest["phase_snapshot_size_bytes"],
        expected_sha256=manifest["phase_snapshot_sha256"],
    )
    roster_raw = _secure_read_file(
        root,
        manifest["roster_snapshot_path"],
        expected_size=manifest["roster_snapshot_size_bytes"],
        expected_sha256=manifest["roster_snapshot_sha256"],
    )
    if phase_raw != expected_phase_bytes or roster_raw != expected_roster_bytes:
        raise PermissionError("disk authority snapshot differs from opaque authority")
    source_by_identity: dict[str, object] = {}
    family = str(manifest["provider_family"])
    for source in record_sources:
        _signal, _transform, _pool_authority, _pool, pool_receipt, _targets = (
            _record_source_components(family, source)
        )
        source_by_identity[str(pool_receipt["analysis_identity_id"])] = source
    expected_inventory = {
        "manifest.json",
        manifest["phase_snapshot_path"],
        manifest["roster_snapshot_path"],
        *[row["artifact_path"] for row in manifest["records"]],
    }
    observed_inventory = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_inventory != expected_inventory:
        raise ValueError("provider disk bundle file inventory drifted")
    for row in manifest["records"]:
        raw = _secure_read_file(
            root,
            row["artifact_path"],
            expected_size=row["artifact_size_bytes"],
            expected_sha256=row["artifact_sha256"],
        )
        try:
            tensors = load_safetensors_bytes(raw)
        except Exception as exc:
            raise ValueError("provider record artifact is invalid safetensors") from exc
        _validate_loaded_record_tensors(
            family=family,
            tensors=tensors,
            manifest_row=row,
            source=source_by_identity[row["analysis_identity_id"]],
        )
    artifact_actual_byte_roster = [
        {
            "artifact_path": row["artifact_path"],
            "artifact_size_bytes": row["artifact_size_bytes"],
            "artifact_sha256": row["artifact_sha256"],
        }
        for row in manifest["records"]
    ]
    admission_receipt = _content_address(
        {
            "schema_version": "provider_training_disk_bundle_admission_v1",
            "executor_id": EXECUTOR_ID,
            "executor_source_sha256": detector_provider_epoch_executor_source_sha256_v1(),
            "provider_family": manifest["provider_family"],
            "variant_id": manifest["variant_id"],
            "outer_fold": manifest["outer_fold"],
            "stage": manifest["stage"],
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "manifest_file_sha256": _bytes_sha256(manifest_raw),
            "phase_snapshot_file_sha256": manifest["phase_snapshot_sha256"],
            "roster_snapshot_file_sha256": manifest["roster_snapshot_sha256"],
            "record_artifact_actual_byte_roster_sha256": _canonical_sha256(
                artifact_actual_byte_roster
            ),
            "record_count": manifest["record_count"],
            "tile_count": manifest["tile_count"],
            "actual_record_artifact_bytes_replayed": True,
            "opaque_phase_and_variant_roster_replayed": True,
            "trusted_local_filesystem_required": True,
            "hostile_concurrent_directory_replacement_defended": False,
            "filesystem_scope": "trusted_local_filesystem_only",
            "large_real_fold_scalability_admitted": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AdmittedProviderTrainingDiskBundleV1(
        _root=str(root),
        _manifest_json=_canonical_json_bytes(manifest).decode("utf-8"),
        _phase_snapshot_json=phase_raw.decode("utf-8"),
        _roster_snapshot_json=roster_raw.decode("utf-8"),
        _admission_receipt_json=_canonical_json_bytes(admission_receipt).decode(
            "utf-8"
        ),
        _phase_authority=phase_authority,
        _variant_roster_authority=variant_roster_authority,
        _record_sources=tuple(record_sources),
        _target_by_tile=dict(target_by_tile),
        _validation_seal=_DISK_AUTHORITY_SEAL,
    )


def _require_disk_authority(
    value: object,
) -> AdmittedProviderTrainingDiskBundleV1:
    if (
        not isinstance(value, AdmittedProviderTrainingDiskBundleV1)
        or value._validation_seal is not _DISK_AUTHORITY_SEAL
    ):
        raise TypeError("provider epoch requires an opaque admitted disk bundle")
    manifest = value.manifest
    manifest = _validate_content_address(
        manifest, context="provider training disk manifest"
    )
    admission = _validate_content_address(
        value.receipt, context="provider training disk admission"
    )
    if (
        manifest.get("schema_version") != DISK_BUNDLE_SCHEMA_VERSION
        or manifest.get("executor_id") != EXECUTOR_ID
        or manifest.get("executor_source_sha256")
        != detector_provider_epoch_executor_source_sha256_v1()
        or manifest.get("provider_family") not in PROVIDER_FAMILIES
        or manifest.get("complete_variant_eligible_denominator_materialized")
        is not True
        or manifest.get(
            "patient_identity_target_and_reference_used_as_model_features"
        )
        is not False
        or manifest.get("real_EEG_training_or_accuracy_claim_materialized")
        is not False
        or admission.get("schema_version")
        != "provider_training_disk_bundle_admission_v1"
        or admission.get("executor_id") != EXECUTOR_ID
        or admission.get("executor_source_sha256")
        != detector_provider_epoch_executor_source_sha256_v1()
        or admission.get("manifest_receipt_sha256")
        != manifest["receipt_sha256"]
        or admission.get("actual_record_artifact_bytes_replayed") is not True
        or admission.get("opaque_phase_and_variant_roster_replayed") is not True
        or admission.get("hostile_concurrent_directory_replacement_defended")
        is not False
    ):
        raise ValueError("opaque provider disk manifest semantics drifted")
    return value


def authorize_provider_epoch_execution_v1(
    disk_bundle: AdmittedProviderTrainingDiskBundleV1,
    *,
    epoch_index: int,
    class_weight_authority: object | None = None,
) -> AuthorizedProviderEpochExecutionV1:
    """Regenerate and seal one complete provider epoch plan.

    A caller cannot supply a plan mapping.  The provider registry rebuilds the
    exact plan from the opaque phase, variant roster and complete record pools
    retained by the actual-byte-admitted disk authority.
    """

    disk = _require_disk_authority(disk_bundle)
    if (
        isinstance(epoch_index, bool)
        or not isinstance(epoch_index, int)
        or epoch_index < 0
    ):
        raise ValueError("epoch_index must be a nonnegative integer")
    manifest = disk.manifest
    phase = disk._phase_authority
    roster = disk._variant_roster_authority
    pools = tuple(source.record_pool_authority for source in disk._record_sources)
    common = {
        "variant_id": manifest["variant_id"],
        "outer_fold": manifest["outer_fold"],
        "stage": manifest["stage"],
        "epoch_index": epoch_index,
    }
    if manifest["provider_family"] == EVENTNET_PROVIDER_FAMILY:
        if class_weight_authority is not None:
            raise TypeError("EventNet epoch does not accept an ST class-weight authority")
        plan = _eventnet.build_authorized_patient_balanced_epoch_plan(
            phase, roster, pools, **common
        )
        class_weight_receipt = None
    else:
        plan = _st.build_authorized_seizuretransformer_epoch_plan(
            phase, roster, pools, **common
        )
        class_weight_receipt = _st._require_authorized_st_class_weight(
            class_weight_authority
        )
        if (
            class_weight_receipt["registry_sha256"] != manifest["registry_sha256"]
            or class_weight_receipt["variant_id"] != manifest["variant_id"]
            or class_weight_receipt["outer_fold"] != manifest["outer_fold"]
            or class_weight_receipt["phase"] != manifest["phase"]
            or class_weight_receipt[
                "detector_fold_phase_receipt_sha256"
            ]
            != manifest["detector_fold_phase_receipt_sha256"]
            or class_weight_receipt[
                "variant_training_roster_receipt_sha256"
            ]
            != manifest["variant_training_roster_receipt_sha256"]
        ):
            raise PermissionError("ST class weight crosses the epoch authority")

    plan = _validate_content_address(plan, context="authorized provider epoch plan")
    primitive = _validate_content_address(
        plan.get("primitive_plan"), context="provider primitive epoch plan"
    )
    if (
        plan["authorized_record_pool_receipt_roster_sha256"]
        != manifest["authorized_record_pool_receipt_roster_sha256"]
        or plan["complete_variant_eligible_record_count"]
        != manifest["record_count"]
        or plan["epoch_index"] != epoch_index
        or primitive["epoch_index"] != epoch_index
        or plan["variant_id"] != manifest["variant_id"]
        or primitive["variant_id"] != manifest["variant_id"]
        or plan["outer_fold"] != manifest["outer_fold"]
        or primitive["outer_fold"] != manifest["outer_fold"]
        or plan["stage"] != manifest["stage"]
        or primitive["stage"] != manifest["stage"]
        or plan["detector_fold_phase_receipt_sha256"]
        != manifest["detector_fold_phase_receipt_sha256"]
        or plan["variant_training_roster_receipt_sha256"]
        != manifest["variant_training_roster_receipt_sha256"]
        or plan["eligible_record_or_patient_deletion_allowed"] is not False
        or plan["prediction_first_denominator_preserved"] is not True
    ):
        raise PermissionError("regenerated provider epoch plan crosses disk authority")

    manifest_tile_rows = {
        tile["tile_id"]: tile
        for record in manifest["records"]
        for tile in record["tiles"]
    }
    ordered_rows: list[dict[str, str]] = []
    seen_by_batch: list[list[str]] = []
    for batch in primitive["batches"]:
        if not isinstance(batch, list) or not batch:
            raise ValueError("provider epoch contains an empty or malformed batch")
        batch_patients: list[str] = []
        for row in batch:
            if type(row) is not dict or set(row) - {"patient_key", "tile_id", "pool"}:
                raise ValueError("provider epoch sampler row fields drifted")
            patient = row.get("patient_key")
            tile_id = row.get("tile_id")
            if (
                not isinstance(patient, str)
                or not patient
                or not isinstance(tile_id, str)
                or tile_id not in manifest_tile_rows
                or manifest_tile_rows[tile_id]["patient_key"] != patient
                or tile_id not in disk._target_by_tile
            ):
                raise PermissionError("epoch sampler row is not bound to a disk target")
            batch_patients.append(patient)
            ordered_rows.append({"patient_key": patient, "tile_id": tile_id})
        if len(batch_patients) != len(set(batch_patients)):
            raise ValueError("provider epoch batch repeats a patient")
        seen_by_batch.append(batch_patients)
    if len(ordered_rows) != sum(len(batch) for batch in primitive["batches"]):
        raise RuntimeError("provider epoch sampler traversal drifted")

    plan_bytes = _canonical_json_bytes(plan)
    receipt = _content_address(
        {
            "schema_version": "authorized_provider_epoch_execution_authority_v1",
            "executor_id": EXECUTOR_ID,
            "executor_source_sha256": detector_provider_epoch_executor_source_sha256_v1(),
            "provider_family": manifest["provider_family"],
            "variant_id": manifest["variant_id"],
            "registry_sha256": manifest["registry_sha256"],
            "outer_fold": manifest["outer_fold"],
            "stage": manifest["stage"],
            "epoch_index": epoch_index,
            "detector_fold_phase_receipt_sha256": manifest[
                "detector_fold_phase_receipt_sha256"
            ],
            "provider_phase_authority_receipt_sha256": manifest[
                "provider_phase_authority_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": manifest[
                "variant_training_roster_receipt_sha256"
            ],
            "training_disk_manifest_receipt_sha256": manifest["receipt_sha256"],
            "training_disk_manifest_file_sha256": disk.receipt[
                "manifest_file_sha256"
            ],
            "authorized_record_pool_receipt_roster_sha256": manifest[
                "authorized_record_pool_receipt_roster_sha256"
            ],
            "class_weight_receipt_sha256": (
                None
                if class_weight_receipt is None
                else class_weight_receipt["receipt_sha256"]
            ),
            "epoch_plan_receipt_sha256": plan["receipt_sha256"],
            "epoch_plan_actual_bytes_sha256": _bytes_sha256(plan_bytes),
            "primitive_sampler_receipt_sha256": primitive["receipt_sha256"],
            "batch_count": len(primitive["batches"]),
            "sampler_row_count": len(ordered_rows),
            "complete_sampler_regenerated_from_opaque_record_pools": True,
            "caller_owned_plan_or_subset_accepted": False,
            "model_forward_control_plane_fields": [],
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AuthorizedProviderEpochExecutionV1(
        disk_bundle=disk,
        _epoch_plan_json=plan_bytes.decode("utf-8"),
        _class_weight_authority=class_weight_authority,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_EPOCH_AUTHORITY_SEAL,
    )


def _require_epoch_authority(
    value: object,
) -> AuthorizedProviderEpochExecutionV1:
    if (
        not isinstance(value, AuthorizedProviderEpochExecutionV1)
        or value._validation_seal is not _EPOCH_AUTHORITY_SEAL
    ):
        raise TypeError("training requires an opaque authorized provider epoch")
    _require_disk_authority(value.disk_bundle)
    receipt = _validate_content_address(
        value.receipt, context="authorized provider epoch execution"
    )
    plan_bytes = value._epoch_plan_json.encode("utf-8")
    plan = _validate_content_address(
        json.loads(value._epoch_plan_json), context="authorized provider epoch plan"
    )
    if (
        receipt["schema_version"]
        != "authorized_provider_epoch_execution_authority_v1"
        or receipt["executor_id"] != EXECUTOR_ID
        or receipt["executor_source_sha256"]
        != detector_provider_epoch_executor_source_sha256_v1()
        or receipt["epoch_plan_receipt_sha256"] != plan["receipt_sha256"]
        or receipt["epoch_plan_actual_bytes_sha256"] != _bytes_sha256(plan_bytes)
        or receipt["epoch_index"] != plan["epoch_index"]
        or receipt["caller_owned_plan_or_subset_accepted"] is not False
        or receipt["complete_sampler_regenerated_from_opaque_record_pools"]
        is not True
        or receipt["model_forward_control_plane_fields"] != []
    ):
        raise ValueError("opaque provider epoch authority drifted")
    return value


def _device_from_string(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("training device is invalid") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("provider executor supports CPU or CUDA only")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA runtime is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "formal provider runtime requires exactly one visible CUDA device"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError("CUDA device index is invalid")
        device = torch.device("cuda", index)
    return device


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
    }


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    if type(state) is not dict or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "device",
    }:
        raise ValueError("provider runtime RNG state fields drifted")
    if state["device"] != str(device):
        raise ValueError("provider runtime RNG device changed")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if not isinstance(state["torch_cuda"], Tensor):
            raise ValueError("provider CUDA RNG state is absent")
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    elif state["torch_cuda"] is not None:
        raise ValueError("CPU runtime carries an unexpected CUDA RNG state")


def _seeded_rng_state(seed: int, device: torch.device) -> dict[str, Any]:
    outer = _capture_rng_state(device)
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        return _capture_rng_state(device)
    finally:
        _restore_rng_state(outer, device)


def _optimizer_parameter_names(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[str, ...]:
    model_rows = list(model.named_parameters())
    if not model_rows:
        raise ValueError("provider model has no trainable parameters")
    name_by_id = {id(parameter): name for name, parameter in model_rows}
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if (
        len(optimizer_parameters) != len(model_rows)
        or len({id(parameter) for parameter in optimizer_parameters})
        != len(optimizer_parameters)
        or set(map(id, optimizer_parameters)) != set(name_by_id)
    ):
        raise PermissionError(
            "optimizer parameters must equal the model parameter roster exactly once"
        )
    observed = tuple(name_by_id[id(parameter)] for parameter in optimizer_parameters)
    expected = tuple(name for name, _parameter in model_rows)
    if observed != expected:
        raise PermissionError("optimizer parameter order differs from named_parameters")
    return observed


def _optimizer_contract(
    family: str, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    expected_type: type[torch.optim.Optimizer]
    expected_type = (
        torch.optim.AdamW
        if family == EVENTNET_PROVIDER_FAMILY
        else torch.optim.RAdam
    )
    if type(optimizer) is not expected_type:
        raise TypeError(
            f"{family} formal optimizer must be {expected_type.__module__}.{expected_type.__name__}"
        )
    if len(optimizer.param_groups) != 1:
        raise ValueError("provider optimizer must contain exactly one parameter group")
    for group in optimizer.param_groups:
        expected = {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 2e-5,
        }
        for name, value in expected.items():
            observed = group.get(name)
            if isinstance(value, tuple):
                if tuple(observed) != value:
                    raise ValueError(f"optimizer {name} drifted")
            elif float(observed) != value:
                raise ValueError(f"optimizer {name} drifted")
        if group.get("maximize", False) is not False:
            raise ValueError("optimizer maximize flag drifted")
        if family == EVENTNET_PROVIDER_FAMILY:
            for name, expected_value in {
                "amsgrad": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
            }.items():
                if group.get(name) is not expected_value:
                    raise ValueError(f"EventNet optimizer {name} drifted")
        else:
            for name, expected_value in {
                "decoupled_weight_decay": False,
                "foreach": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
            }.items():
                if group.get(name) is not expected_value:
                    raise ValueError(f"ST optimizer {name} drifted")
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 2e-5,
        "gradient_clip_global_L2_norm": 1.0,
    }


def _load_st_architecture_class() -> type[nn.Module]:
    from third_party.SeizureTransformer.time_step_level.model import (
        SeizureTransformer,
    )

    return SeizureTransformer


def _formal_architecture_receipt(
    family: str, variant_id: str, model: nn.Module
) -> dict[str, Any]:
    channels = _expected_channel_count(variant_id)
    if family == EVENTNET_PROVIDER_FAMILY:
        if type(model) is not _eventnet.EventNetCleanroomUNet:
            raise TypeError("formal EventNet runtime requires EventNetCleanroomUNet")
        if int(model.input_channels) != channels:
            raise ValueError("EventNet runtime input width crosses variant")
        architecture_id = "EventNetCleanroomUNet"
        source_sha256 = _eventnet.eventnet_cleanroom_registry_code_sha256()
    else:
        architecture_type = _load_st_architecture_class()
        if type(model) is not architecture_type:
            raise TypeError("formal ST runtime requires the vendored SeizureTransformer")
        if int(model.in_channels) != channels or int(model.in_samples) != _st.TILE_SAMPLES:
            raise ValueError("ST runtime input geometry crosses variant")
        architecture_id = (
            "third_party.SeizureTransformer.time_step_level.model.SeizureTransformer"
        )
        source_sha256 = (
            "0c3fd38a5350bb293e5337c26bb01c83945624b6eb8000da50e955e54174c7b2"
        )
    return {
        "scope": "registered_cleanroom_architecture_software_execution_unadmitted",
        "architecture_id": architecture_id,
        "architecture_source_sha256": source_sha256,
        "variant_id": variant_id,
        "input_channels": channels,
        "registered_architecture_software_execution": True,
        "strict_ordered_module_hyperparameter_ledger_admitted": False,
        "promotable_architecture": False,
        "nonpromotion_reason": (
            "strict_ordered_architecture_execution_ledger_not_yet_materialized"
        ),
    }


def _make_runtime_authority(
    epoch_authority: AuthorizedProviderEpochExecutionV1,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    initialization_receipt: Mapping[str, Any],
    device: str,
    architecture_receipt: Mapping[str, Any],
    rng_state: Mapping[str, Any] | None = None,
) -> ProviderModelRuntimeAuthorityV1:
    epoch = _require_epoch_authority(epoch_authority)
    context = epoch.receipt
    torch_device = _device_from_string(device)
    if any(parameter.device != torch_device for parameter in model.parameters()):
        raise ValueError("provider model parameters do not share the declared device")
    parameter_names = _optimizer_parameter_names(model, optimizer)
    optimizer_contract = _optimizer_contract(context["provider_family"], optimizer)
    initialization = _validate_content_address(
        dict(initialization_receipt), context="provider random initialization"
    )
    if (
        initialization.get("variant_id") != context["variant_id"]
        or initialization.get("outer_fold") != context["outer_fold"]
        or initialization.get("stage") != context["stage"]
        or initialization.get("derived_seed")
        != (
            _eventnet.derive_training_seed(
                variant_id=context["variant_id"],
                outer_fold=context["outer_fold"],
                stage=context["stage"],
            )
            if context["provider_family"] == EVENTNET_PROVIDER_FAMILY
            else _st.derive_training_seed(
                variant_id=context["variant_id"],
                outer_fold=context["outer_fold"],
                stage=context["stage"],
            )
        )
    ):
        raise PermissionError("initialization receipt crosses provider run identity")
    if rng_state is None:
        rng_state = _seeded_rng_state(int(initialization["derived_seed"]), torch_device)
    if type(rng_state) is not dict or rng_state.get("device") != str(torch_device):
        raise ValueError("provider runtime RNG state is invalid")
    model_bytes = _model_state_bytes(model)
    optimizer_state = optimizer.state_dict()
    receipt = _content_address(
        {
            "schema_version": "provider_model_optimizer_runtime_authority_v1",
            "executor_id": EXECUTOR_ID,
            "executor_source_sha256": detector_provider_epoch_executor_source_sha256_v1(),
            "provider_family": context["provider_family"],
            "variant_id": context["variant_id"],
            "registry_sha256": context["registry_sha256"],
            "outer_fold": context["outer_fold"],
            "stage": context["stage"],
            "detector_fold_phase_receipt_sha256": context[
                "detector_fold_phase_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": context[
                "variant_training_roster_receipt_sha256"
            ],
            "training_disk_manifest_receipt_sha256": context[
                "training_disk_manifest_receipt_sha256"
            ],
            "initialization_receipt": initialization,
            "architecture_receipt": deepcopy(dict(architecture_receipt)),
            "optimizer_contract": optimizer_contract,
            "optimizer_parameter_names": list(parameter_names),
            "initial_model_safetensors_sha256": _bytes_sha256(model_bytes),
            "current_model_state_sha256": _state_digest(model.state_dict()),
            "current_optimizer_state_sha256": _state_digest(optimizer_state),
            "current_rng_state_sha256": _state_digest(dict(rng_state)),
            "device": str(torch_device),
            "completed_epoch_count": 0,
            "next_epoch_index": 0,
            "published_or_other_variant_checkpoint_loaded": False,
            "raw_model_optimizer_or_hash_is_checkpoint_authority": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return ProviderModelRuntimeAuthorityV1(
        model=model,
        optimizer=optimizer,
        _rng_state=deepcopy(dict(rng_state)),
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_RUNTIME_AUTHORITY_SEAL,
    )


def build_registered_provider_runtime_v1(
    epoch_authority: AuthorizedProviderEpochExecutionV1,
    *,
    device: str = "cpu",
) -> ProviderModelRuntimeAuthorityV1:
    """Randomly initialize the exact registered model and frozen optimizer."""

    epoch = _require_epoch_authority(epoch_authority)
    context = epoch.receipt
    if context["epoch_index"] != 0:
        raise PermissionError("a fresh provider runtime must begin at epoch index zero")
    torch_device = _device_from_string(device)
    variant_id = str(context["variant_id"])
    if context["provider_family"] == EVENTNET_PROVIDER_FAMILY:
        model, upstream_init = _eventnet.build_randomly_initialized_model(
            variant_id=variant_id,
            outer_fold=int(context["outer_fold"]),
            stage=str(context["stage"]),
        )
        initialization = _content_address(
            {
                "schema_version": "provider_random_initialization_receipt_v1",
                "provider_family": EVENTNET_PROVIDER_FAMILY,
                "variant_id": variant_id,
                "outer_fold": context["outer_fold"],
                "stage": context["stage"],
                "derived_seed": upstream_init["derived_seed"],
                "upstream_initialization_receipt": upstream_init,
                "published_or_other_variant_checkpoint_loaded": False,
                "receipt_sha256": _CONTENT_PENDING,
            }
        )
        model = model.to(torch_device)
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=2e-5,
        )
    else:
        seed = _st.derive_training_seed(
            variant_id=variant_id,
            outer_fold=int(context["outer_fold"]),
            stage=str(context["stage"]),
        )
        architecture_type = _load_st_architecture_class()
        devices = [torch_device.index] if torch_device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if torch_device.type == "cuda":
                torch.cuda.manual_seed(seed)
            model = architecture_type(
                in_channels=_expected_channel_count(variant_id),
                in_samples=_st.TILE_SAMPLES,
                dim_feedforward=2048,
                num_layers=8,
                num_heads=4,
                drop_rate=0.1,
            ).to(torch_device)
        initialization = _content_address(
            {
                "schema_version": "provider_random_initialization_receipt_v1",
                "provider_family": SEIZURETRANSFORMER_PROVIDER_FAMILY,
                "variant_id": variant_id,
                "outer_fold": context["outer_fold"],
                "stage": context["stage"],
                "derived_seed": seed,
                "initial_state_dict_sha256": _state_digest(model.state_dict()),
                "published_or_other_variant_checkpoint_loaded": False,
                "receipt_sha256": _CONTENT_PENDING,
            }
        )
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=2e-5,
            foreach=False,
        )
    architecture = _formal_architecture_receipt(
        context["provider_family"], variant_id, model
    )
    return _make_runtime_authority(
        epoch,
        model=model,
        optimizer=optimizer,
        initialization_receipt=initialization,
        device=str(torch_device),
        architecture_receipt=architecture,
    )


def _authorize_nonpromotable_conformance_runtime_v1(
    epoch_authority: AuthorizedProviderEpochExecutionV1,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> ProviderModelRuntimeAuthorityV1:
    """Test-only runtime; receipts permanently deny checkpoint promotion."""

    epoch = _require_epoch_authority(epoch_authority)
    context = epoch.receipt
    seed = (
        _eventnet.derive_training_seed(
            variant_id=context["variant_id"],
            outer_fold=context["outer_fold"],
            stage=context["stage"],
        )
        if context["provider_family"] == EVENTNET_PROVIDER_FAMILY
        else _st.derive_training_seed(
            variant_id=context["variant_id"],
            outer_fold=context["outer_fold"],
            stage=context["stage"],
        )
    )
    initialization = _content_address(
        {
            "schema_version": "provider_random_initialization_receipt_v1",
            "provider_family": context["provider_family"],
            "variant_id": context["variant_id"],
            "outer_fold": context["outer_fold"],
            "stage": context["stage"],
            "derived_seed": seed,
            "initial_state_dict_sha256": _state_digest(model.state_dict()),
            "published_or_other_variant_checkpoint_loaded": False,
            "nonpromotable_executor_conformance_fixture": True,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    architecture = {
        "scope": "nonpromotable_executor_conformance_fixture",
        "architecture_id": type(model).__qualname__,
        "architecture_source_sha256": None,
        "variant_id": context["variant_id"],
        "input_channels": _expected_channel_count(context["variant_id"]),
        "promotable_architecture": False,
    }
    return _make_runtime_authority(
        epoch,
        model=model,
        optimizer=optimizer,
        initialization_receipt=initialization,
        device=device,
        architecture_receipt=architecture,
    )


def _require_runtime_authority(
    value: object,
) -> ProviderModelRuntimeAuthorityV1:
    if (
        not isinstance(value, ProviderModelRuntimeAuthorityV1)
        or value._validation_seal is not _RUNTIME_AUTHORITY_SEAL
    ):
        raise TypeError("provider execution requires an opaque model runtime")
    receipt = _validate_content_address(
        value.receipt, context="provider model runtime authority"
    )
    device = _device_from_string(receipt["device"])
    parameter_devices = {parameter.device for parameter in value.model.parameters()}
    buffer_devices = {buffer.device for buffer in value.model.buffers()}
    optimizer_tensor_devices = {
        tensor.device
        for state in value.optimizer.state.values()
        for tensor in state.values()
        if isinstance(tensor, Tensor) and tensor.ndim > 0
    }
    if (
        receipt["schema_version"] != "provider_model_optimizer_runtime_authority_v1"
        or receipt["executor_id"] != EXECUTOR_ID
        or receipt["executor_source_sha256"]
        != detector_provider_epoch_executor_source_sha256_v1()
        or receipt["current_model_state_sha256"]
        != _state_digest(value.model.state_dict())
        or receipt["current_optimizer_state_sha256"]
        != _state_digest(value.optimizer.state_dict())
        or receipt["current_rng_state_sha256"] != _state_digest(value._rng_state)
        or tuple(receipt["optimizer_parameter_names"])
        != _optimizer_parameter_names(value.model, value.optimizer)
        or value._rng_state.get("device") != str(device)
        or parameter_devices != {device}
        or (buffer_devices and buffer_devices != {device})
        or (
            optimizer_tensor_devices
            and optimizer_tensor_devices != {device}
        )
    ):
        raise ValueError("opaque provider runtime state drifted")
    _optimizer_contract(receipt["provider_family"], value.optimizer)
    if receipt["architecture_receipt"].get(
        "registered_architecture_software_execution"
    ) is True:
        expected = _formal_architecture_receipt(
            receipt["provider_family"], receipt["variant_id"], value.model
        )
        if receipt["architecture_receipt"] != expected:
            raise ValueError("registered provider runtime architecture drifted")
    return value


def seizuretransformer_authorized_differentiable_patient_macro_bce_v1(
    probabilities: Tensor,
    *,
    target_bundles: Sequence[_st.AuthorizedSeizureTransformerTargetBundle],
    class_weight_authority: _st.AuthorizedSeizureTransformerClassWeight,
) -> Tensor:
    """Differentiable Torch realization of the frozen ST patient-macro BCE.

    The existing registry function is an independent NumPy/float replay
    oracle.  This function keeps the same reduction and epsilon while
    accepting targets/patient grouping only from opaque provider authorities.
    """

    if not isinstance(probabilities, Tensor) or not probabilities.is_floating_point():
        raise TypeError("ST probabilities must be a floating-point Torch tensor")
    if probabilities.ndim == 3 and probabilities.shape[1] == 1:
        probabilities = probabilities[:, 0, :]
    probabilities = probabilities.float()
    if probabilities.ndim != 2 or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("ST probabilities must be finite [tiles,samples]")
    bundles = [
        _st._require_authorized_st_target_bundle(bundle)
        for bundle in target_bundles
    ]
    if len(bundles) != probabilities.shape[0]:
        raise ValueError("ST probability/opaque-target batch geometry drifted")
    class_weight = _st._require_authorized_st_class_weight(class_weight_authority)
    first = bundles[0].receipt if bundles else None
    if first is None:
        raise ValueError("ST differentiable loss needs at least one target bundle")
    if any(
        bundle.receipt["registry_sha256"] != first["registry_sha256"]
        or bundle.receipt["variant_id"] != first["variant_id"]
        or bundle.receipt["outer_fold"] != first["outer_fold"]
        or bundle.receipt["phase"] != first["phase"]
        or bundle.receipt["variant_training_roster_receipt_sha256"]
        != first["variant_training_roster_receipt_sha256"]
        for bundle in bundles
    ) or (
        class_weight["registry_sha256"] != first["registry_sha256"]
        or class_weight["variant_id"] != first["variant_id"]
        or class_weight["outer_fold"] != first["outer_fold"]
        or class_weight["phase"] != first["phase"]
        or class_weight["variant_training_roster_receipt_sha256"]
        != first["variant_training_roster_receipt_sha256"]
    ):
        raise PermissionError("ST differentiable loss authority binding drifted")
    target = torch.stack(
        [
            torch.from_numpy(np.asarray(bundle.target).copy())
            for bundle in bundles
        ]
    ).to(device=probabilities.device, dtype=probabilities.dtype)
    mask = torch.stack(
        [
            torch.from_numpy(np.asarray(bundle.observed_mask).copy())
            for bundle in bundles
        ]
    ).to(device=probabilities.device, dtype=torch.bool)
    if target.shape != probabilities.shape or mask.shape != probabilities.shape:
        raise ValueError("ST probability and opaque target shapes differ")
    if bool(torch.any(probabilities < 0)) or bool(torch.any(probabilities > 1)):
        raise ValueError("ST probabilities must lie in [0,1]")
    positive_weight = float(
        class_weight["primitive_class_weight_receipt"]["positive_weight"]
    )
    clipped = probabilities.clamp(1e-7, 1.0 - 1e-7)
    per_sample = -(
        positive_weight * target * torch.log(clipped)
        + (1.0 - target) * torch.log1p(-clipped)
    )
    sample_weight = torch.where(
        target == 1,
        torch.as_tensor(
            positive_weight, device=target.device, dtype=target.dtype
        ),
        torch.ones((), device=target.device, dtype=target.dtype),
    )
    patient_rows: dict[str, list[int]] = {}
    for index, bundle in enumerate(bundles):
        patient = str(bundle.receipt["fold_owned_patient_key"])
        patient_rows.setdefault(patient, []).append(index)
    patient_losses: list[Tensor] = []
    for patient in sorted(patient_rows):
        rows = patient_rows[patient]
        numerator = (per_sample[rows] * mask[rows]).sum()
        denominator = (sample_weight[rows] * mask[rows]).sum()
        if not bool(denominator > 0):
            raise ValueError("ST opaque target has no observed loss opportunity")
        patient_losses.append(numerator / denominator)
    result = torch.stack(patient_losses).mean()
    if not bool(torch.isfinite(result)):
        raise ValueError("ST differentiable loss became nonfinite")
    return result


def _load_record_artifact_for_execution(
    disk: AdmittedProviderTrainingDiskBundleV1,
    manifest_row: Mapping[str, Any],
) -> dict[str, Tensor]:
    raw = _secure_read_file(
        disk.root,
        manifest_row["artifact_path"],
        expected_size=manifest_row["artifact_size_bytes"],
        expected_sha256=manifest_row["artifact_sha256"],
    )
    try:
        tensors = load_safetensors_bytes(raw)
    except Exception as exc:
        raise ValueError("execution record artifact is invalid safetensors") from exc
    return {name: value.detach().cpu().contiguous() for name, value in tensors.items()}


def _disk_target_bundle_for_execution(
    *,
    family: str,
    original: object,
    tensors: Mapping[str, Tensor],
    tile_row: Mapping[str, Any],
) -> object:
    keys = tile_row["tensor_keys"]
    if family == EVENTNET_PROVIDER_FAMILY:
        opaque = _eventnet._require_authorized_eventnet_target_bundle(original)
        result = _eventnet.AuthorizedEventNetTargetBundle(
            center_target=tensors[keys["center_target"]].numpy(),
            duration_target=tensors[keys["duration_target"]].numpy(),
            center_loss_mask=tensors[keys["center_loss_mask"]].numpy().astype(
                np.bool_, copy=False
            ),
            duration_loss_mask=tensors[keys["duration_loss_mask"]].numpy().astype(
                np.bool_, copy=False
            ),
            distinct_center_count=int(opaque.distinct_center_count),
            _receipt_json=opaque._receipt_json,
            _validation_seal=_eventnet._TARGET_BUNDLE_SEAL,
        )
        return _eventnet._require_authorized_eventnet_target_bundle(result)
    opaque = _st._require_authorized_st_target_bundle(original)
    result = _st.AuthorizedSeizureTransformerTargetBundle(
        target=tensors[keys["target"]].numpy(),
        observed_mask=tensors[keys["observed_mask"]].numpy(),
        _receipt_json=opaque._receipt_json,
        _validation_seal=_st._TARGET_BUNDLE_SEAL,
    )
    return _st._require_authorized_st_target_bundle(result)


def _execution_batch(
    epoch: AuthorizedProviderEpochExecutionV1,
    batch: Sequence[Mapping[str, Any]],
    *,
    record_cache: OrderedDict[str, dict[str, Tensor]],
    maximum_cached_records: int,
) -> tuple[Tensor, list[object], list[object]]:
    disk = epoch.disk_bundle
    manifest = disk.manifest
    family = manifest["provider_family"]
    record_by_tile: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in manifest["records"]:
        for tile in record["tiles"]:
            record_by_tile[tile["tile_id"]] = (record, tile)
    model_inputs: list[Tensor] = []
    target_bundles: list[object] = []
    model_tiles: list[object] = []
    for sampler_row in batch:
        tile_id = sampler_row["tile_id"]
        if tile_id not in record_by_tile:
            raise PermissionError("sampler tile is absent from the disk manifest")
        record, tile = record_by_tile[tile_id]
        artifact_path = record["artifact_path"]
        if artifact_path not in record_cache:
            tensors = _load_record_artifact_for_execution(disk, record)
            source = next(
                value
                for value in disk._record_sources
                if value.record_pool_authority.receipt["analysis_identity_id"]
                == record["analysis_identity_id"]
            )
            _validate_loaded_record_tensors(
                family=family,
                tensors=tensors,
                manifest_row=record,
                source=source,
            )
            record_cache[artifact_path] = tensors
            record_cache.move_to_end(artifact_path)
            while len(record_cache) > maximum_cached_records:
                record_cache.popitem(last=False)
        tensors = record_cache[artifact_path]
        record_cache.move_to_end(artifact_path)
        signal = tensors["signal"].numpy()
        opaque_target = _disk_target_bundle_for_execution(
            family=family,
            original=disk._target_by_tile[tile_id],
            tensors=tensors,
            tile_row=tile,
        )
        start = int(tile["target_start_sample"])
        stop = int(tile["target_stop_sample_exclusive"])
        if family == EVENTNET_PROVIDER_FAMILY:
            model_tile = _eventnet.materialize_model_tile(
                signal, target_start_sample=start
            )
            if (
                model_tile.receipt["receipt_sha256"]
                != opaque_target.receipt["model_tile_receipt_sha256"]
            ):
                raise ValueError("disk EEG reconstructed the wrong EventNet tile")
            model_inputs.append(torch.from_numpy(model_tile.model_input_uv.copy()))
            model_tiles.append(model_tile)
        else:
            if stop - start != _st.TILE_SAMPLES or stop > signal.shape[1]:
                raise ValueError("disk EEG cannot reconstruct the ST tile")
            model_inputs.append(torch.from_numpy(signal[:, start:stop].copy()))
        target_bundles.append(opaque_target)
    return torch.stack(model_inputs), target_bundles, model_tiles


def _advance_runtime_receipt(
    runtime: ProviderModelRuntimeAuthorityV1,
    *,
    rng_state: Mapping[str, Any],
    completed_epoch_count: int,
    next_epoch_index: int,
) -> ProviderModelRuntimeAuthorityV1:
    original = runtime.receipt
    updated = deepcopy(original)
    updated["current_model_state_sha256"] = _state_digest(
        runtime.model.state_dict()
    )
    updated["current_optimizer_state_sha256"] = _state_digest(
        runtime.optimizer.state_dict()
    )
    updated["current_rng_state_sha256"] = _state_digest(dict(rng_state))
    updated["completed_epoch_count"] = completed_epoch_count
    updated["next_epoch_index"] = next_epoch_index
    updated["receipt_sha256"] = _CONTENT_PENDING
    updated["receipt_sha256"] = _canonical_sha256(updated)
    return ProviderModelRuntimeAuthorityV1(
        model=runtime.model,
        optimizer=runtime.optimizer,
        _rng_state=deepcopy(dict(rng_state)),
        _receipt_json=_canonical_json_bytes(updated).decode("utf-8"),
        _validation_seal=_RUNTIME_AUTHORITY_SEAL,
    )


def execute_authorized_provider_epoch_v1(
    epoch_authority: AuthorizedProviderEpochExecutionV1,
    runtime_authority: ProviderModelRuntimeAuthorityV1,
    *,
    maximum_cached_records: int = 4,
) -> CompletedProviderEpochV1:
    """Consume every batch exactly once and return a completed-epoch seal."""

    epoch = _require_epoch_authority(epoch_authority)
    runtime = _require_runtime_authority(runtime_authority)
    epoch_receipt = epoch.receipt
    runtime_receipt = runtime.receipt
    if (
        isinstance(maximum_cached_records, bool)
        or not isinstance(maximum_cached_records, int)
        or maximum_cached_records <= 0
    ):
        raise ValueError("maximum_cached_records must be a positive integer")
    for field in (
        "provider_family",
        "variant_id",
        "registry_sha256",
        "outer_fold",
        "stage",
        "detector_fold_phase_receipt_sha256",
        "variant_training_roster_receipt_sha256",
        "training_disk_manifest_receipt_sha256",
    ):
        if runtime_receipt[field] != epoch_receipt[field]:
            raise PermissionError(f"provider runtime crosses epoch field: {field}")
    if runtime_receipt["next_epoch_index"] != epoch_receipt["epoch_index"]:
        raise PermissionError("provider runtime may resume only at the exact next epoch")
    device = _device_from_string(runtime_receipt["device"])
    model = runtime.model
    optimizer = runtime.optimizer
    before_model = _state_digest(model.state_dict())
    before_optimizer = _state_digest(optimizer.state_dict())
    before_rng = _state_digest(runtime._rng_state)
    primitive = epoch.epoch_plan["primitive_plan"]
    expected_batch_count = len(primitive["batches"])
    if (
        epoch_receipt["provider_family"] == SEIZURETRANSFORMER_PROVIDER_FAMILY
        and int(primitive["batch_count"]) != expected_batch_count
    ):
        raise ValueError("provider epoch batch-count receipt drifted")

    outer_rng = _capture_rng_state(device)
    outer_deterministic = torch.are_deterministic_algorithms_enabled()
    outer_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    outer_cudnn_benchmark = torch.backends.cudnn.benchmark
    outer_cudnn_deterministic = torch.backends.cudnn.deterministic
    outer_cuda_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    outer_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    loss_rows: list[dict[str, Any]] = []
    record_cache: OrderedDict[str, dict[str, Tensor]] = OrderedDict()
    completed_batches = 0
    model.train()
    try:
        if (
            device.type == "cuda"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise RuntimeError(
                "CUDA provider training requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _restore_rng_state(runtime._rng_state, device)
        for batch_index, batch in enumerate(primitive["batches"]):
            inputs, targets, model_tiles = _execution_batch(
                epoch,
                batch,
                record_cache=record_cache,
                maximum_cached_records=maximum_cached_records,
            )
            inputs = inputs.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = (
                device.type == "cuda"
                and epoch_receipt["provider_family"]
                == SEIZURETRANSFORMER_PROVIDER_FAMILY
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                if epoch_receipt["provider_family"] == EVENTNET_PROVIDER_FAMILY:
                    if not hasattr(model, "forward_logits"):
                        raise TypeError("EventNet runtime lacks forward_logits")
                    center_logits, duration_logits = model.forward_logits(inputs)
                    bound = _eventnet.bind_eventnet_training_logits(
                        center_logits,
                        duration_logits,
                        target_bundles=targets,
                        model_tiles=model_tiles,
                    )
                    loss_result = _eventnet.eventnet_authorized_multitask_loss(
                        bound, target_bundles=targets
                    )
                    loss = loss_result.loss
                    metrics = {
                        "loss_hex": float(loss.detach().cpu().double()).hex(),
                        "center_loss_hex": float(
                            loss_result.center_loss.detach().cpu().double()
                        ).hex(),
                        "duration_loss_hex": float(
                            loss_result.duration_loss.detach().cpu().double()
                        ).hex(),
                    }
                else:
                    probabilities = model(inputs)
                    loss = seizuretransformer_authorized_differentiable_patient_macro_bce_v1(
                        probabilities,
                        target_bundles=targets,
                        class_weight_authority=epoch._class_weight_authority,
                    )
                    metrics = {
                        "loss_hex": float(loss.detach().cpu().double()).hex()
                    }
            if not isinstance(loss, Tensor) or not bool(torch.isfinite(loss)):
                raise ValueError("provider training loss is nonfinite")
            loss.backward()
            parameters_with_grad = [
                parameter
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not parameters_with_grad:
                raise RuntimeError("provider training batch produced no gradients")
            if any(
                not bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters_with_grad
            ):
                raise ValueError("provider training gradients are nonfinite")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            for value in model.state_dict().values():
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError("optimizer step produced nonfinite model state")
            for state in optimizer.state.values():
                for value in state.values():
                    if (
                        isinstance(value, Tensor)
                        and value.is_floating_point()
                        and not bool(torch.isfinite(value).all())
                    ):
                        raise ValueError(
                            "optimizer step produced nonfinite optimizer state"
                        )
            completed_batches += 1
            loss_rows.append(
                {
                    "batch_index": batch_index,
                    "ordered_tile_ids": [row["tile_id"] for row in batch],
                    "ordered_patient_keys_sha256": _canonical_sha256(
                        [row["patient_key"] for row in batch]
                    ),
                    **metrics,
                    "preclip_gradient_global_L2_norm_hex": float(
                        gradient_norm.detach().cpu().double()
                    ).hex(),
                }
            )
        if completed_batches != expected_batch_count:
            raise RuntimeError("provider epoch ended before every sampler batch")
        final_rng = _capture_rng_state(device)
    finally:
        _restore_rng_state(outer_rng, device)
        torch.use_deterministic_algorithms(
            outer_deterministic, warn_only=outer_warn_only
        )
        torch.backends.cudnn.benchmark = outer_cudnn_benchmark
        torch.backends.cudnn.deterministic = outer_cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = outer_cuda_matmul_tf32
        torch.backends.cudnn.allow_tf32 = outer_cudnn_tf32
    completed_count = int(epoch_receipt["epoch_index"]) + 1
    advanced_runtime = _advance_runtime_receipt(
        runtime,
        rng_state=final_rng,
        completed_epoch_count=completed_count,
        next_epoch_index=completed_count,
    )
    receipt = _content_address(
        {
            "schema_version": EPOCH_EXECUTION_SCHEMA_VERSION,
            "executor_id": EXECUTOR_ID,
            "executor_source_sha256": detector_provider_epoch_executor_source_sha256_v1(),
            "provider_family": epoch_receipt["provider_family"],
            "variant_id": epoch_receipt["variant_id"],
            "registry_sha256": epoch_receipt["registry_sha256"],
            "outer_fold": epoch_receipt["outer_fold"],
            "stage": epoch_receipt["stage"],
            "epoch_index": epoch_receipt["epoch_index"],
            "epoch_completed_one_based": completed_count,
            "next_epoch_index": completed_count,
            "detector_fold_phase_receipt_sha256": epoch_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "provider_phase_authority_receipt_sha256": epoch_receipt[
                "provider_phase_authority_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": epoch_receipt[
                "variant_training_roster_receipt_sha256"
            ],
            "training_disk_manifest_receipt_sha256": epoch_receipt[
                "training_disk_manifest_receipt_sha256"
            ],
            "training_disk_manifest_file_sha256": epoch_receipt[
                "training_disk_manifest_file_sha256"
            ],
            "epoch_plan_receipt_sha256": epoch_receipt[
                "epoch_plan_receipt_sha256"
            ],
            "epoch_plan_actual_bytes_sha256": epoch_receipt[
                "epoch_plan_actual_bytes_sha256"
            ],
            "class_weight_receipt_sha256": epoch_receipt[
                "class_weight_receipt_sha256"
            ],
            "initialization_receipt_sha256": runtime_receipt[
                "initialization_receipt"
            ]["receipt_sha256"],
            "architecture_receipt": runtime_receipt["architecture_receipt"],
            "model_state_before_sha256": before_model,
            "optimizer_state_before_sha256": before_optimizer,
            "rng_state_before_sha256": before_rng,
            "model_state_after_sha256": _state_digest(model.state_dict()),
            "optimizer_state_after_sha256": _state_digest(optimizer.state_dict()),
            "rng_state_after_sha256": _state_digest(final_rng),
            "batch_count_expected": expected_batch_count,
            "batch_count_completed": completed_batches,
            "optimizer_step_count_this_epoch": completed_batches,
            "ordered_batch_loss_and_gradient_ledger": loss_rows,
            "all_sampler_batches_consumed_exactly_once": True,
            "checkpoint_save_boundary": "completed_epoch_only",
            "resume_boundary": "next_epoch_only",
            "numeric_execution": {
                "torch_deterministic_algorithms_enforced": True,
                "deterministic_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "CUDA_TF32_allowed": False,
                "CUBLAS_WORKSPACE_CONFIG_required_on_CUDA": ":4096:8",
                "single_visible_CUDA_device_required": True,
                "autocast": (
                    "cuda_bfloat16"
                    if device.type == "cuda"
                    and epoch_receipt["provider_family"]
                    == SEIZURETRANSFORMER_PROVIDER_FAMILY
                    else (
                        "cuda_float32"
                        if device.type == "cuda"
                        else "cpu_float32_conformance_only"
                    )
                ),
            },
            "ST_loss_execution_scope": (
                None
                if epoch_receipt["provider_family"] == EVENTNET_PROVIDER_FAMILY
                else (
                    "frozen_patient_macro_loss_within_each_distinct_patient_"
                    "batch_stochastic_estimator_not_full_epoch_joint_objective"
                )
            ),
            "real_EEG_or_performance_claim_materialized": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return CompletedProviderEpochV1(
        epoch_authority=epoch,
        runtime_authority=advanced_runtime,
        _receipt_json=_canonical_json_bytes(receipt).decode("utf-8"),
        _validation_seal=_COMPLETED_EPOCH_SEAL,
    )


def _require_completed_epoch(value: object) -> CompletedProviderEpochV1:
    if (
        not isinstance(value, CompletedProviderEpochV1)
        or value._validation_seal is not _COMPLETED_EPOCH_SEAL
    ):
        raise TypeError("checkpoint publication requires a completed-epoch authority")
    epoch = _require_epoch_authority(value.epoch_authority)
    runtime = _require_runtime_authority(value.runtime_authority)
    receipt = _validate_content_address(
        value.receipt, context="completed provider epoch"
    )
    if (
        receipt["schema_version"] != EPOCH_EXECUTION_SCHEMA_VERSION
        or receipt["executor_id"] != EXECUTOR_ID
        or receipt["executor_source_sha256"]
        != detector_provider_epoch_executor_source_sha256_v1()
        or receipt["epoch_index"] != epoch.receipt["epoch_index"]
        or receipt["epoch_completed_one_based"] != receipt["epoch_index"] + 1
        or receipt["next_epoch_index"] != receipt["epoch_completed_one_based"]
        or runtime.receipt["next_epoch_index"] != receipt["next_epoch_index"]
        or receipt["model_state_after_sha256"]
        != runtime.receipt["current_model_state_sha256"]
        or receipt["optimizer_state_after_sha256"]
        != runtime.receipt["current_optimizer_state_sha256"]
        or receipt["rng_state_after_sha256"]
        != runtime.receipt["current_rng_state_sha256"]
        or receipt["batch_count_completed"] != receipt["batch_count_expected"]
        or receipt["all_sampler_batches_consumed_exactly_once"] is not True
        or receipt["checkpoint_save_boundary"] != "completed_epoch_only"
        or receipt["resume_boundary"] != "next_epoch_only"
    ):
        raise ValueError("completed provider epoch authority drifted")
    return value


def _checkpoint_file_descriptor(path: str, data: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": len(data),
        "sha256": _bytes_sha256(data),
    }


def _checkpoint_static_payloads(
    completed: CompletedProviderEpochV1,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    completed = _require_completed_epoch(completed)
    epoch = completed.epoch_authority
    runtime = completed.runtime_authority
    disk = epoch.disk_bundle
    runtime_receipt = runtime.receipt
    completed_receipt = completed.receipt
    model_bytes = _model_state_bytes(runtime.model)
    optimizer_bytes = _serialize_state_tree(runtime.optimizer.state_dict())
    rng_bytes = _serialize_state_tree(runtime._rng_state)
    initialization_bytes = _canonical_json_bytes(
        runtime_receipt["initialization_receipt"]
    )
    architecture_bytes = _canonical_json_bytes(
        runtime_receipt["architecture_receipt"]
    )
    runtime_receipt_bytes = _canonical_json_bytes(runtime_receipt)
    completed_bytes = _canonical_json_bytes(completed_receipt)
    epoch_plan_bytes = epoch._epoch_plan_json.encode("utf-8")
    phase_bytes = disk._phase_snapshot_json.encode("utf-8")
    roster_bytes = disk._roster_snapshot_json.encode("utf-8")
    disk_admission_bytes = _canonical_json_bytes(disk.receipt)
    if epoch._class_weight_authority is None:
        class_weight_bytes = _canonical_json_bytes(None)
    else:
        class_weight_bytes = _canonical_json_bytes(
            _st._require_authorized_st_class_weight(
                epoch._class_weight_authority
            )
        )
    numeric_environment = {
        "schema_version": "provider_epoch_numeric_environment_v1",
        "python_version": ".".join(
            str(value) for value in os.sys.version_info[:3]
        ),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": runtime_receipt["device"],
        "training_autocast": (
            "cuda_bfloat16"
            if runtime_receipt["device"].startswith("cuda")
            and runtime_receipt["provider_family"]
            == SEIZURETRANSFORMER_PROVIDER_FAMILY
            else (
                "cuda_float32"
                if runtime_receipt["device"].startswith("cuda")
                else "cpu_float32_conformance_only"
            )
        ),
        "CUDA_TF32_allowed": False,
        "cudnn_benchmark": False,
        "deterministic_algorithms_required_during_epoch": True,
        "dataloader_workers": 0,
        "automatic_microbatch_backoff_allowed": False,
    }
    numeric_bytes = _canonical_json_bytes(numeric_environment)
    payloads = {
        "model.safetensors": model_bytes,
        "optimizer.statezip": optimizer_bytes,
        "rng.statezip": rng_bytes,
        "initialization.json": initialization_bytes,
        "architecture.json": architecture_bytes,
        "runtime_receipt.json": runtime_receipt_bytes,
        "completed_epoch.json": completed_bytes,
        "epoch_plan.json": epoch_plan_bytes,
        "phase_authority.json": phase_bytes,
        "variant_roster_authority.json": roster_bytes,
        "class_weight_authority.json": class_weight_bytes,
        "training_disk_admission.json": disk_admission_bytes,
        "numeric_environment.json": numeric_bytes,
    }
    model_ledger = _model_state_ledger(runtime.model)
    metadata = {
        "model_state_ledger": model_ledger,
        "model_state_ledger_sha256": _canonical_sha256(model_ledger),
        "optimizer_state_sha256": _state_digest(runtime.optimizer.state_dict()),
        "rng_state_sha256": _state_digest(runtime._rng_state),
        "numeric_environment": numeric_environment,
    }
    return payloads, metadata


def _build_checkpoint_manifest(
    completed: CompletedProviderEpochV1,
    payloads: Mapping[str, bytes],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    epoch = completed.epoch_authority
    runtime = completed.runtime_authority
    epoch_receipt = epoch.receipt
    runtime_receipt = runtime.receipt
    completed_receipt = completed.receipt
    file_rows = [
        _checkpoint_file_descriptor(path, payloads[path])
        for path in sorted(payloads)
    ]
    return _content_address(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "executor_id": EXECUTOR_ID,
            "executor_source_sha256": detector_provider_epoch_executor_source_sha256_v1(),
            "provider_family": epoch_receipt["provider_family"],
            "variant_id": epoch_receipt["variant_id"],
            "registry_sha256": epoch_receipt["registry_sha256"],
            "outer_fold": epoch_receipt["outer_fold"],
            "stage": epoch_receipt["stage"],
            "epoch_index": epoch_receipt["epoch_index"],
            "epoch_completed_one_based": completed_receipt[
                "epoch_completed_one_based"
            ],
            "next_epoch_index": completed_receipt["next_epoch_index"],
            "detector_fold_phase_receipt_sha256": epoch_receipt[
                "detector_fold_phase_receipt_sha256"
            ],
            "provider_phase_authority_receipt_sha256": epoch_receipt[
                "provider_phase_authority_receipt_sha256"
            ],
            "variant_training_roster_receipt_sha256": epoch_receipt[
                "variant_training_roster_receipt_sha256"
            ],
            "training_disk_manifest_receipt_sha256": epoch_receipt[
                "training_disk_manifest_receipt_sha256"
            ],
            "training_disk_manifest_file_sha256": epoch_receipt[
                "training_disk_manifest_file_sha256"
            ],
            "epoch_plan_receipt_sha256": epoch_receipt[
                "epoch_plan_receipt_sha256"
            ],
            "epoch_plan_actual_bytes_sha256": epoch_receipt[
                "epoch_plan_actual_bytes_sha256"
            ],
            "class_weight_receipt_sha256": epoch_receipt[
                "class_weight_receipt_sha256"
            ],
            "initialization_receipt_sha256": runtime_receipt[
                "initialization_receipt"
            ]["receipt_sha256"],
            "architecture_receipt": runtime_receipt["architecture_receipt"],
            "optimizer_contract": runtime_receipt["optimizer_contract"],
            "optimizer_parameter_names": runtime_receipt[
                "optimizer_parameter_names"
            ],
            "model_state_ledger_sha256": metadata[
                "model_state_ledger_sha256"
            ],
            "model_state_sha256": runtime_receipt[
                "current_model_state_sha256"
            ],
            "optimizer_state_sha256": metadata["optimizer_state_sha256"],
            "rng_state_sha256": metadata["rng_state_sha256"],
            "completed_epoch_receipt_sha256": completed_receipt[
                "receipt_sha256"
            ],
            "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
            "files": file_rows,
            "serialization": (
                "tensor_only_safetensors_plus_deterministic_typed_tree_"
                "archives_and_canonical_JSON_no_pickle"
            ),
            "save_boundary": "completed_epoch_only",
            "resume_boundary": "next_epoch_only",
            "partial_epoch_checkpoint_allowed": False,
            "architecture_promotable": runtime_receipt[
                "architecture_receipt"
            ]["promotable_architecture"],
            "numeric_execution_promotable": False,
            "nonpromotion_reasons": [
                "strict_ordered_architecture_execution_ledger_not_admitted",
                "whole_fold_streaming_materializer_not_implemented",
                "cross_process_signed_completed_epoch_authority_not_implemented",
            ],
            "test_fixture_checkpoint": not runtime_receipt[
                "architecture_receipt"
            ]["promotable_architecture"],
            "real_EEG_checkpoint_or_performance_claim_materialized": False,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )


def write_completed_provider_epoch_checkpoint_v1(
    output_directory: str | Path,
    completed_epoch: CompletedProviderEpochV1,
    *,
    _test_only_allow_nonpromotable: bool = False,
) -> AdmittedProviderEpochCheckpointV1:
    """Publish a byte-deterministic checkpoint at a completed-epoch boundary."""

    completed = _require_completed_epoch(completed_epoch)
    promotable = completed.runtime_authority.receipt["architecture_receipt"][
        "promotable_architecture"
    ]
    numeric_promotable = False
    if (not promotable or not numeric_promotable) and not (
        _test_only_allow_nonpromotable is True
    ):
        raise PermissionError(
            "v1 checkpoint publication is non-promotable pending the strict "
            "architecture ledger, streaming fold materializer and cross-process "
            "completed-epoch authority"
        )
    payloads, metadata = _checkpoint_static_payloads(completed)
    manifest = _build_checkpoint_manifest(completed, payloads, metadata)
    destination, temporary = _strict_new_directory(output_directory)
    try:
        for relative_path, data in sorted(payloads.items()):
            _atomic_write_bytes(temporary / relative_path, data)
        _atomic_write_bytes(
            temporary / "checkpoint.json", _canonical_json_bytes(manifest)
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return admit_provider_epoch_checkpoint_v1(
        destination,
        completed_epoch_authority=completed,
        runtime_authority=completed.runtime_authority,
        _test_only_allow_nonpromotable=_test_only_allow_nonpromotable,
    )


def _validate_model_payload_against_runtime(
    raw: bytes,
    *,
    runtime: ProviderModelRuntimeAuthorityV1,
    manifest: Mapping[str, Any],
) -> dict[str, Tensor]:
    try:
        loaded = load_safetensors_bytes(raw)
    except Exception as exc:
        raise ValueError("checkpoint model payload is invalid safetensors") from exc
    expected = runtime.model.state_dict()
    if set(loaded) != set(expected):
        raise ValueError("checkpoint model state-key roster drifted")
    for name in expected:
        if (
            loaded[name].dtype != expected[name].dtype
            or tuple(loaded[name].shape) != tuple(expected[name].shape)
            or (
                loaded[name].is_floating_point()
                and not bool(torch.isfinite(loaded[name]).all())
            )
        ):
            raise ValueError("checkpoint model tensor metadata drifted")
    ledger = [
        {"name": name, **_tensor_payload_receipt(value)}
        for name, value in sorted(loaded.items())
    ]
    if _canonical_sha256(ledger) != manifest["model_state_ledger_sha256"]:
        raise ValueError("checkpoint model tensor ledger drifted")
    if _state_digest(loaded) != manifest["model_state_sha256"]:
        raise ValueError("checkpoint model state digest drifted")
    return loaded


def _validate_optimizer_state_payload(
    state: object, runtime: ProviderModelRuntimeAuthorityV1
) -> dict[str, Any]:
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer root fields drifted")
    if type(state["state"]) is not dict or type(state["param_groups"]) is not list:
        raise ValueError("checkpoint optimizer state containers drifted")
    if len(state["param_groups"]) != 1:
        raise ValueError("checkpoint optimizer parameter-group count drifted")
    parameters = list(runtime.model.parameters())
    group = state["param_groups"][0]
    if type(group) is not dict or group.get("params") != list(range(len(parameters))):
        raise ValueError("checkpoint optimizer parameter index/order drifted")
    fresh_group = runtime.optimizer.state_dict()["param_groups"][0]
    if set(group) != set(fresh_group):
        raise ValueError("checkpoint optimizer group fields drifted")
    for name in group:
        if name != "params" and group[name] != fresh_group[name]:
            raise ValueError(f"checkpoint optimizer static group field drifted: {name}")
    if set(state["state"]) != set(range(len(parameters))):
        raise ValueError("checkpoint optimizer moment roster is incomplete or extra")
    for index, parameter in enumerate(parameters):
        row = state["state"][index]
        if type(row) is not dict or not row:
            raise ValueError("checkpoint optimizer parameter state is malformed")
        for name, value in row.items():
            if isinstance(value, Tensor):
                if value.numel() != 1 and tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"checkpoint optimizer tensor shape drifted: {name}"
                    )
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError("checkpoint optimizer tensor is nonfinite")
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("checkpoint optimizer scalar is nonfinite")
            elif not isinstance(value, (bool, int, str)) and value is not None:
                raise TypeError("checkpoint optimizer state contains an unsupported value")
    return state


def _validate_rng_state_replay(
    state: object, *, device: torch.device
) -> dict[str, Any]:
    if type(state) is not dict:
        raise ValueError("checkpoint RNG state root drifted")
    outer = _capture_rng_state(device)
    try:
        _restore_rng_state(state, device)
        replayed = _capture_rng_state(device)
        if _state_digest(replayed) != _state_digest(state):
            raise ValueError("checkpoint RNG state did not replay exactly")
    finally:
        _restore_rng_state(outer, device)
    return state


def admit_provider_epoch_checkpoint_v1(
    directory: str | Path,
    *,
    completed_epoch_authority: CompletedProviderEpochV1,
    runtime_authority: ProviderModelRuntimeAuthorityV1,
    _test_only_allow_nonpromotable: bool = False,
) -> AdmittedProviderEpochCheckpointV1:
    """Replay a completed-epoch checkpoint and restore exact next-epoch state."""

    completed = _require_completed_epoch(completed_epoch_authority)
    epoch = completed.epoch_authority
    runtime = _require_runtime_authority(runtime_authority)
    root = Path(directory).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("provider checkpoint must be a real directory")
    manifest_raw = _secure_read_file(root, "checkpoint.json")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider checkpoint manifest is unreadable") from exc
    manifest = _validate_content_address(
        manifest, context="provider completed-epoch checkpoint manifest"
    )
    if manifest_raw != _canonical_json_bytes(manifest):
        raise ValueError("provider checkpoint manifest JSON is noncanonical")
    epoch_receipt = epoch.receipt
    for field in (
        "provider_family",
        "variant_id",
        "registry_sha256",
        "outer_fold",
        "stage",
        "epoch_index",
        "detector_fold_phase_receipt_sha256",
        "provider_phase_authority_receipt_sha256",
        "variant_training_roster_receipt_sha256",
        "training_disk_manifest_receipt_sha256",
        "training_disk_manifest_file_sha256",
        "epoch_plan_receipt_sha256",
        "epoch_plan_actual_bytes_sha256",
        "class_weight_receipt_sha256",
    ):
        if manifest.get(field) != epoch_receipt.get(field):
            raise PermissionError(f"checkpoint crosses current epoch authority: {field}")
    if (
        manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or manifest["executor_id"] != EXECUTOR_ID
        or manifest["executor_source_sha256"]
        != detector_provider_epoch_executor_source_sha256_v1()
        or manifest["epoch_completed_one_based"] != manifest["epoch_index"] + 1
        or manifest["next_epoch_index"] != manifest["epoch_completed_one_based"]
        or manifest["save_boundary"] != "completed_epoch_only"
        or manifest["resume_boundary"] != "next_epoch_only"
        or manifest["partial_epoch_checkpoint_allowed"] is not False
        or manifest["real_EEG_checkpoint_or_performance_claim_materialized"]
        is not False
    ):
        raise ValueError("provider checkpoint boundary semantics drifted")
    if (
        payloads_completed_sha := manifest["completed_epoch_receipt_sha256"]
    ) != completed.receipt["receipt_sha256"]:
        raise PermissionError(
            "checkpoint does not bind the process-sealed completed epoch"
        )
    if (
        not manifest["architecture_promotable"]
        or not manifest["numeric_execution_promotable"]
    ) and not (_test_only_allow_nonpromotable is True):
        raise PermissionError("non-promotable conformance checkpoint cannot be admitted")

    file_rows = manifest.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise ValueError("provider checkpoint file ledger is missing")
    file_by_path: dict[str, dict[str, Any]] = {}
    for row in file_rows:
        if (
            type(row) is not dict
            or set(row) != {"path", "size_bytes", "sha256"}
            or not isinstance(row["path"], str)
            or row["path"] in file_by_path
        ):
            raise ValueError("provider checkpoint file ledger drifted")
        file_by_path[row["path"]] = row
    expected_inventory = {"checkpoint.json", *file_by_path}
    observed_inventory = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_inventory != expected_inventory:
        raise ValueError("provider checkpoint file inventory drifted")
    payloads = {
        path: _secure_read_file(
            root,
            path,
            expected_size=row["size_bytes"],
            expected_sha256=row["sha256"],
        )
        for path, row in sorted(file_by_path.items())
    }
    required_payloads = {
        "model.safetensors",
        "optimizer.statezip",
        "rng.statezip",
        "initialization.json",
        "architecture.json",
        "runtime_receipt.json",
        "completed_epoch.json",
        "epoch_plan.json",
        "phase_authority.json",
        "variant_roster_authority.json",
        "class_weight_authority.json",
        "training_disk_admission.json",
        "numeric_environment.json",
    }
    if set(payloads) != required_payloads:
        raise ValueError("provider checkpoint payload roster drifted")
    disk = epoch.disk_bundle
    exact_snapshots = {
        "epoch_plan.json": epoch._epoch_plan_json.encode("utf-8"),
        "phase_authority.json": disk._phase_snapshot_json.encode("utf-8"),
        "variant_roster_authority.json": disk._roster_snapshot_json.encode(
            "utf-8"
        ),
        "training_disk_admission.json": _canonical_json_bytes(disk.receipt),
    }
    for name, expected in exact_snapshots.items():
        if payloads[name] != expected:
            raise PermissionError(f"checkpoint {name} crosses current opaque authority")
    if epoch._class_weight_authority is None:
        expected_class_weight = _canonical_json_bytes(None)
    else:
        expected_class_weight = _canonical_json_bytes(
            _st._require_authorized_st_class_weight(epoch._class_weight_authority)
        )
    if payloads["class_weight_authority.json"] != expected_class_weight:
        raise PermissionError("checkpoint ST class-weight authority drifted")

    try:
        initialization = json.loads(payloads["initialization.json"].decode("utf-8"))
        architecture = json.loads(payloads["architecture.json"].decode("utf-8"))
        saved_runtime_receipt = json.loads(
            payloads["runtime_receipt.json"].decode("utf-8")
        )
        completed_receipt = json.loads(
            payloads["completed_epoch.json"].decode("utf-8")
        )
        numeric_environment = json.loads(
            payloads["numeric_environment.json"].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint JSON payload is unreadable") from exc
    for name, value in (
        ("initialization", initialization),
        ("saved runtime", saved_runtime_receipt),
        ("completed epoch", completed_receipt),
    ):
        _validate_content_address(value, context=f"checkpoint {name}")
    current_runtime = runtime.receipt
    if (
        initialization != current_runtime["initialization_receipt"]
        or architecture != current_runtime["architecture_receipt"]
        or manifest["initialization_receipt_sha256"]
        != initialization["receipt_sha256"]
        or manifest["architecture_receipt"] != architecture
        or manifest["optimizer_contract"] != current_runtime["optimizer_contract"]
        or tuple(manifest["optimizer_parameter_names"])
        != _optimizer_parameter_names(runtime.model, runtime.optimizer)
        or manifest["completed_epoch_receipt_sha256"]
        != completed_receipt["receipt_sha256"]
        or completed_receipt != completed.receipt
        or manifest["runtime_receipt_sha256"]
        != saved_runtime_receipt["receipt_sha256"]
    ):
        raise PermissionError("checkpoint initialization/runtime authority drifted")
    if (
        completed_receipt["epoch_index"] != epoch_receipt["epoch_index"]
        or completed_receipt["next_epoch_index"] != manifest["next_epoch_index"]
        or saved_runtime_receipt["next_epoch_index"] != manifest["next_epoch_index"]
        or saved_runtime_receipt["completed_epoch_count"]
        != manifest["epoch_completed_one_based"]
    ):
        raise ValueError("checkpoint resume cursor drifted")
    expected_numeric = {
        "schema_version": "provider_epoch_numeric_environment_v1",
        "python_version": ".".join(str(value) for value in os.sys.version_info[:3]),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": current_runtime["device"],
        "training_autocast": (
            "cuda_bfloat16"
            if current_runtime["device"].startswith("cuda")
            and current_runtime["provider_family"]
            == SEIZURETRANSFORMER_PROVIDER_FAMILY
            else (
                "cuda_float32"
                if current_runtime["device"].startswith("cuda")
                else "cpu_float32_conformance_only"
            )
        ),
        "CUDA_TF32_allowed": False,
        "cudnn_benchmark": False,
        "deterministic_algorithms_required_during_epoch": True,
        "dataloader_workers": 0,
        "automatic_microbatch_backoff_allowed": False,
    }
    if numeric_environment != expected_numeric:
        raise PermissionError("checkpoint numeric environment changed")

    loaded_model = _validate_model_payload_against_runtime(
        payloads["model.safetensors"], runtime=runtime, manifest=manifest
    )
    optimizer_state = _validate_optimizer_state_payload(
        _deserialize_state_tree(payloads["optimizer.statezip"]), runtime
    )
    rng_state = _validate_rng_state_replay(
        _deserialize_state_tree(payloads["rng.statezip"]),
        device=_device_from_string(current_runtime["device"]),
    )
    if _state_digest(optimizer_state) != manifest["optimizer_state_sha256"]:
        raise ValueError("checkpoint optimizer state digest drifted")
    if _state_digest(rng_state) != manifest["rng_state_sha256"]:
        raise ValueError("checkpoint RNG state digest drifted")

    runtime.model.load_state_dict(loaded_model, strict=True)
    runtime.optimizer.load_state_dict(optimizer_state)
    if (
        _state_digest(runtime.model.state_dict()) != manifest["model_state_sha256"]
        or _state_digest(runtime.optimizer.state_dict())
        != manifest["optimizer_state_sha256"]
    ):
        raise ValueError("checkpoint model/optimizer did not restore tensor-exactly")
    restored_runtime = ProviderModelRuntimeAuthorityV1(
        model=runtime.model,
        optimizer=runtime.optimizer,
        _rng_state=deepcopy(rng_state),
        _receipt_json=_canonical_json_bytes(saved_runtime_receipt).decode("utf-8"),
        _validation_seal=_RUNTIME_AUTHORITY_SEAL,
    )
    _require_runtime_authority(restored_runtime)
    admission_receipt = _content_address(
        {
            "schema_version": "provider_epoch_checkpoint_actual_byte_admission_v1",
            "executor_id": EXECUTOR_ID,
            "executor_source_sha256": detector_provider_epoch_executor_source_sha256_v1(),
            "provider_family": manifest["provider_family"],
            "variant_id": manifest["variant_id"],
            "outer_fold": manifest["outer_fold"],
            "stage": manifest["stage"],
            "epoch_completed_one_based": manifest["epoch_completed_one_based"],
            "next_epoch_index": manifest["next_epoch_index"],
            "checkpoint_manifest_receipt_sha256": manifest["receipt_sha256"],
            "checkpoint_manifest_actual_bytes_sha256": _bytes_sha256(manifest_raw),
            "checkpoint_file_actual_byte_roster_sha256": _canonical_sha256(
                file_rows
            ),
            "model_optimizer_rng_actual_bytes_replayed": True,
            "phase_variant_roster_sampler_and_disk_bytes_replayed": True,
            "process_sealed_completed_epoch_replayed": True,
            "resume_scope": (
                "same_process_process_sealed_completed_epoch_exact_reload_only"
            ),
            "cross_process_formal_resume_admitted": False,
            "pickle_or_executable_deserialization_used": False,
            "architecture_promotable": manifest["architecture_promotable"],
            "numeric_execution_promotable": manifest[
                "numeric_execution_promotable"
            ],
            "test_fixture_checkpoint": manifest["test_fixture_checkpoint"],
            "real_EEG_checkpoint_or_performance_claim_materialized": False,
            "trusted_local_filesystem_required": True,
            "receipt_sha256": _CONTENT_PENDING,
        }
    )
    return AdmittedProviderEpochCheckpointV1(
        _root=str(root),
        disk_bundle=epoch.disk_bundle,
        runtime_authority=restored_runtime,
        _manifest_json=_canonical_json_bytes(manifest).decode("utf-8"),
        _receipt_json=_canonical_json_bytes(admission_receipt).decode("utf-8"),
        _validation_seal=_CHECKPOINT_AUTHORITY_SEAL,
    )


def _require_checkpoint_authority(
    value: object,
) -> AdmittedProviderEpochCheckpointV1:
    if (
        not isinstance(value, AdmittedProviderEpochCheckpointV1)
        or value._validation_seal is not _CHECKPOINT_AUTHORITY_SEAL
    ):
        raise TypeError("resume requires an opaque actual-byte-admitted checkpoint")
    _require_runtime_authority(value.runtime_authority)
    _require_disk_authority(value.disk_bundle)
    manifest = _validate_content_address(
        value.manifest, context="admitted provider checkpoint manifest"
    )
    receipt = _validate_content_address(
        value.receipt, context="provider checkpoint admission"
    )
    if (
        manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or receipt["schema_version"]
        != "provider_epoch_checkpoint_actual_byte_admission_v1"
        or receipt["executor_source_sha256"]
        != detector_provider_epoch_executor_source_sha256_v1()
        or receipt["checkpoint_manifest_receipt_sha256"]
        != manifest["receipt_sha256"]
        or receipt["model_optimizer_rng_actual_bytes_replayed"] is not True
        or receipt["phase_variant_roster_sampler_and_disk_bytes_replayed"]
        is not True
        or receipt["process_sealed_completed_epoch_replayed"] is not True
        or receipt["resume_scope"]
        != "same_process_process_sealed_completed_epoch_exact_reload_only"
        or receipt["cross_process_formal_resume_admitted"] is not False
        or receipt["pickle_or_executable_deserialization_used"] is not False
    ):
        raise ValueError("opaque provider checkpoint admission drifted")
    return value


def authorize_next_epoch_from_checkpoint_v1(
    checkpoint_authority: AdmittedProviderEpochCheckpointV1,
    *,
    class_weight_authority: object | None = None,
) -> AuthorizedProviderEpochExecutionV1:
    """Regenerate exactly N+1 from an admitted completed-epoch checkpoint."""

    checkpoint = _require_checkpoint_authority(checkpoint_authority)
    next_epoch = int(checkpoint.manifest["next_epoch_index"])
    return authorize_provider_epoch_execution_v1(
        checkpoint.disk_bundle,
        epoch_index=next_epoch,
        class_weight_authority=class_weight_authority,
    )


__all__ = [
    "AdmittedProviderEpochCheckpointV1",
    "AdmittedProviderTrainingDiskBundleV1",
    "AuthorizedProviderEpochExecutionV1",
    "CHECKPOINT_SCHEMA_VERSION",
    "CompletedProviderEpochV1",
    "DISK_BUNDLE_SCHEMA_VERSION",
    "EPOCH_EXECUTION_SCHEMA_VERSION",
    "EVENTNET_PROVIDER_FAMILY",
    "EXECUTOR_ID",
    "EventNetAuthorizedDiskRecordV1",
    "PROVIDER_FAMILIES",
    "ProviderModelRuntimeAuthorityV1",
    "SCHEMA_VERSION",
    "SEIZURETRANSFORMER_PROVIDER_FAMILY",
    "SeizureTransformerAuthorizedDiskRecordV1",
    "admit_provider_epoch_checkpoint_v1",
    "admit_provider_training_disk_bundle_v1",
    "authorize_provider_epoch_execution_v1",
    "authorize_next_epoch_from_checkpoint_v1",
    "build_registered_provider_runtime_v1",
    "execute_authorized_provider_epoch_v1",
    "materialize_provider_training_disk_bundle_v1",
    "seizuretransformer_authorized_differentiable_patient_macro_bce_v1",
    "write_completed_provider_epoch_checkpoint_v1",
]
