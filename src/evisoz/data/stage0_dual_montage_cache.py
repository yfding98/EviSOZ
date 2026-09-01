"""Synthetic, source-bound Stage 0 dual-montage cache.

This module intentionally has no file/private-data reader.  Its sole input is
one 200 Hz common-reference tensor in the frozen ``STANDARD_19 + A1 + A2``
order.  CAR19 and TCP22 are consequently reproducible from that tensor and
the observed mask alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from src.evisoz.baseline.v29_cache import canonical_tensor_bytes
from src.evisoz.baseline.v29_public_cache_materializer import (
    _capture_disk_snapshot,
    _create_staging_directory_at,
    _decode_canonical_tensor,
    _directory_open_flags,
    _fsync_directory_fd,
    _open_absolute_directory,
    _rename_directory_noreplace_at,
    _write_exclusive_bytes_at,
)
from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    build_raw_artifact_ref,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_bytes,
    validate_artifact_ref,
)
from src.evisoz.data.channel_registry import build_default_channel_registry
from src.evisoz.data.event_identity import build_event_identity, validate_event_identity
from src.evisoz.data.tcp22_views import (
    build_montage_derivation_receipt,
    validate_montage_derivation_receipt,
)
from src.evisoz.data.opaque_reference_authority import (
    OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION,
)
from src.clinical_eeg_long_recording.montage_reference_observability import (
    build_montage_reference_observability_receipt,
)
from src.soz.geometry import STANDARD_19, normalize_electrode_name


STAGE0_DUAL_MONTAGE_CACHE_SCHEMA_VERSION = "evisoz_stage0_dual_montage_cache_v1"
MATERIALIZATION_RECEIPT_SCHEMA_VERSION = (
    "evisoz_dual_montage_cache_materialization_receipt_v1"
)
MATERIALIZER_VERSION = "evisoz_stage0_dual_montage_disk_materializer_v1"
TENSOR_SCHEMA_VERSION = "evisoz_canonical_tensor_v1"

_PLACEHOLDER = "0" * 64
_PARENT_LABELS = tuple((*STANDARD_19, "A1", "A2"))
_TENSOR_FILES = {
    "v29_reference": "tensors/v29_reference.tensor",
    "tcp22_context": "tensors/tcp22_context.tensor",
    "tcp22_onset": "tensors/tcp22_onset.tensor",
}
_RECEIPT_FILES = {
    "montage_receipt": "sidecars/montage_receipt.json",
    "event_identity": "sidecars/event_identity.json",
    "parent_signal": "sidecars/parent_signal.tensor",
}
_MATERIALIZATION_FILE = "audit/materialization_receipt.json"
_EXPECTED_DIRS = {"tensors", "sidecars", "audit"}
_REQUIRED_FILES = set(_RECEIPT_FILES.values())
_OPTIONAL_FILES = set(_TENSOR_FILES.values())


def _canonical_input(value: object, *, context: str, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{context} must be a torch.Tensor")
    if value.dtype is not torch.float32:
        raise TypeError(f"{context} must have dtype torch.float32")
    if value.device.type != "cpu":
        raise ValueError(f"{context} must be a CPU tensor")
    if value.layout != torch.strided or value.requires_grad:
        raise ValueError(f"{context} must be a detached dense tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{context} must have shape {list(shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{context} must be contiguous")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{context} must contain finite values")
    return value.detach().clone().contiguous()


def _mask(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype is not torch.bool or value.device.type != "cpu":
            raise TypeError("observed_mask must be a CPU torch.bool tensor")
        if tuple(value.shape) != (21,):
            raise ValueError("observed_mask must have shape [21]")
        result = value.detach().clone().contiguous()
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("observed_mask must be an ordered boolean sequence")
        if len(value) != 21 or any(type(item) is not bool for item in value):
            raise ValueError("observed_mask must contain 21 booleans")
        result = torch.tensor(list(value), dtype=torch.bool)
    return result


def _tensor_ref(value: torch.Tensor) -> dict[str, Any]:
    raw = canonical_tensor_bytes(value)
    return build_raw_artifact_ref(
        raw,
        artifact_kind="tensor_cache",
        media_type="application/x-evisoz-canonical-tensor",
        payload_schema_version=TENSOR_SCHEMA_VERSION,
    )


def _parent_ref(value: torch.Tensor) -> dict[str, Any]:
    return build_raw_artifact_ref(
        canonical_tensor_bytes(value),
        artifact_kind="canonical_signal",
        media_type="application/x-evisoz-canonical-tensor",
        payload_schema_version=TENSOR_SCHEMA_VERSION,
    )


def _synthetic_identity(parent_ref: Mapping[str, object]) -> dict[str, Any]:
    anchor = build_raw_artifact_ref(
        b"evisoz-stage0-synthetic-anchor-v1",
        artifact_kind="source_event_annotation",
        media_type="application/octet-stream",
    )
    patient_hash = hashlib.sha256(b"evisoz-stage0-synthetic-patient-v1").hexdigest()
    return build_event_identity(
        dataset_id="synthetic",
        sample_id="SYNTHETIC-STAGE0",
        event_id="SYNTHETIC-STAGE0-EVENT",
        linkage_group_id="SYNTHETIC-STAGE0-LINKAGE",
        source_patient_sha256=patient_hash,
        parent_signal_ref=parent_ref,
        anchor_source_ref=anchor,
        anchor_quality="exact",
    )


def _observability(parent_ref: Mapping[str, object], observed: torch.Tensor) -> dict[str, Any]:
    labels = [
        f"EEG {name}-REF"
        for name, is_observed in zip(_PARENT_LABELS, observed.tolist())
        if is_observed
    ]
    return build_montage_reference_observability_receipt(
        signal_labels=labels,
        source_signal_sha256=str(parent_ref["content_hash"]["sha256"]),
    )


@dataclass(frozen=True)
class Stage0DualMontageCarrier:
    """Immutable-by-convention in-memory source and its three views."""

    parent_signal: torch.Tensor
    observed_mask: torch.Tensor
    v29_reference: torch.Tensor | None
    v29_observed_mask: torch.Tensor
    tcp22_context: torch.Tensor
    tcp22_onset: torch.Tensor
    tcp22_observed_mask: torch.Tensor
    parent_signal_ref: Mapping[str, Any]
    event_identity: Mapping[str, Any]
    montage_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class OpenedStage0DualMontageCache:
    """Disk-opened handle.  Every checkout decodes a fresh tensor clone."""

    materialization_receipt: Mapping[str, Any]
    montage_receipt: Mapping[str, Any]
    _snapshots: Mapping[str, bytes]

    def _checkout(self, name: str) -> torch.Tensor:
        raw = self._snapshots.get(name)
        if raw is None:
            raise KeyError(f"view is not materialized: {name}")
        return _decode_canonical_tensor(raw, context=name).clone().contiguous()

    def checkout_v29_reference(self) -> torch.Tensor | None:
        if "v29_reference" not in self._snapshots:
            return None
        return self._checkout("v29_reference")

    def checkout_tcp22_context(self) -> torch.Tensor:
        return self._checkout("tcp22_context")

    def checkout_tcp22_onset(self) -> torch.Tensor:
        return self._checkout("tcp22_onset")

    @property
    def receipt(self) -> Mapping[str, Any]:
        return self.materialization_receipt


@dataclass(frozen=True)
class Stage0DualMontageCacheMaterialization:
    path: Path
    opened: OpenedStage0DualMontageCache
    materialization_receipt: Mapping[str, Any]


def build_common_reference_event_carrier(
    parent_signal: torch.Tensor | None = None,
    observed_mask: torch.Tensor | Sequence[object] | None = None,
    *,
    authoritative_v29: torch.Tensor | None = None,
    authoritative_v29_car19: torch.Tensor | None = None,
    event_identity: Mapping[str, object] | None = None,
    parent_signal_ref: Mapping[str, object] | None = None,
    montage_reference_observability_receipt: Mapping[str, object] | None = None,
) -> Stage0DualMontageCarrier:
    """Build CAR19 and signed direct-parent TCP22 from synthetic input.

    Missing endpoints are represented by zero rows and false edge masks.  No
    interpolation or spatial substitution is performed.
    """

    if parent_signal is None:
        raise TypeError("parent_signal is required")
    if observed_mask is None:
        raise TypeError("observed_mask is required")
    parent = _canonical_input(parent_signal, context="parent_signal", shape=(21, 12000))
    mask = _mask(observed_mask)
    if torch.any(parent[~mask] != 0):
        raise ValueError("unobserved parent rows must be exactly zero")

    generated_parent_ref = _parent_ref(parent)
    if parent_signal_ref is None:
        parent_ref = generated_parent_ref
    else:
        parent_ref = validate_artifact_ref(parent_signal_ref)
        if parent_ref != generated_parent_ref:
            raise ValueError("parent_signal_ref does not bind canonical parent tensor")

    if event_identity is None:
        identity = _synthetic_identity(parent_ref)
    else:
        identity = validate_event_identity(event_identity)
        if identity["parent_signal_ref"] != parent_ref:
            raise ValueError("event identity parent does not match parent signal")

    if authoritative_v29 is not None and authoritative_v29_car19 is not None:
        raise TypeError("provide only one authoritative_v29 spelling")
    if authoritative_v29_car19 is not None:
        authoritative_v29 = authoritative_v29_car19
    if montage_reference_observability_receipt is None:
        observability = _observability(parent_ref, mask)
    else:
        observability = montage_reference_observability_receipt
    opaque_reference_route = (
        isinstance(observability, Mapping)
        and observability.get("schema_version")
        == OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION
    )

    standard_mask = mask[:19].clone()
    car = None
    if authoritative_v29 is not None:
        if not bool(torch.all(standard_mask).item()):
            raise ValueError("authoritative_v29 requires all 19 Standard19 rows observed")
        car = _canonical_input(authoritative_v29, context="authoritative_v29", shape=(19, 12000))
        reference = parent[:19].mean(dim=0)
        expected_car = (parent[:19] - reference).contiguous()
        if not torch.allclose(car, expected_car, rtol=0.0, atol=1e-12):
            raise ValueError("authoritative_v29 is not the Standard19 CAR19 replay")

    registry = build_default_channel_registry()
    tcp = torch.zeros((22, 12000), dtype=torch.float32)
    tcp_onset = torch.zeros((22, 2000), dtype=torch.float32)
    edge_mask = torch.zeros(22, dtype=torch.bool)
    names = {name: index for index, name in enumerate(_PARENT_LABELS)}
    edge_states: list[str] = []
    orientation: list[bool] = []
    for index, edge in enumerate(registry["tcp22_derivations"]):
        positive = str(edge["positive_electrode"]["normalized"])
        negative = str(edge["negative_electrode"]["normalized"])
        have = bool(mask[names[positive]].item() and mask[names[negative]].item())
        if have:
            tcp[index] = parent[names[positive]] - parent[names[negative]]
            tcp_onset[index] = (
                parent[names[positive], 2000:4000]
                - parent[names[negative], 2000:4000]
            )
            edge_mask[index] = True
            edge_states.append(
                "exact_derived_from_protocol_authorized_opaque_common_reference"
                if opaque_reference_route
                else "exact_derived_from_common_reference"
            )
            orientation.append(True)
        else:
            edge_states.append("unobserved")
            orientation.append(False)
    # The existing receipt deliberately reserves v29_reference for a formal
    # fully observed Standard19 field.  Partial CAR remains available only as
    # an in-memory diagnostic, never as a falsely complete formal view.
    receipt_v29 = car if bool(torch.all(standard_mask).item()) else None
    receipt_tcp = tcp if bool(torch.any(edge_mask).item()) else None
    montage = build_montage_derivation_receipt(
        parent_signal_ref=parent_ref,
        event_identity=identity,
        v29_tensor_ref=_tensor_ref(receipt_v29) if receipt_v29 is not None else None,
        tcp22_context_tensor_ref=_tensor_ref(receipt_tcp) if receipt_tcp is not None else None,
        tcp22_onset_tensor_ref=_tensor_ref(tcp_onset)
        if receipt_tcp is not None
        else None,
        standard19_observed_mask=standard_mask.tolist(),
        tcp22_edge_states=edge_states,
        tcp22_orientation_ok=orientation,
        montage_reference_observability_receipt=(
            observability if bool(torch.any(edge_mask).item()) else None
        ),
        channel_registry=registry,
    )
    return Stage0DualMontageCarrier(
        parent_signal=parent,
        observed_mask=mask,
        v29_reference=car,
        v29_observed_mask=standard_mask,
        tcp22_context=tcp,
        tcp22_onset=tcp_onset,
        tcp22_observed_mask=edge_mask,
        parent_signal_ref=_deep_freeze(parent_ref),
        event_identity=_deep_freeze(identity),
        montage_receipt=_deep_freeze(montage),
    )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _seal_receipt(body: dict[str, Any]) -> dict[str, Any]:
    body["receipt_sha256"] = _PLACEHOLDER
    body["receipt_sha256"] = canonical_json_sha256(body)
    return body


def _build_materialization_receipt(raw_files: Mapping[str, bytes], carrier: Stage0DualMontageCarrier) -> dict[str, Any]:
    files = {
        name: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
        for name, raw in sorted(raw_files.items())
    }
    body = {
        "schema_version": MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "status": "complete",
        "cache_id": "EVISOZ-STAGE0-" + canonical_json_sha256(_plain(carrier.montage_receipt))[:24],
        "parent_signal_ref": _plain(carrier.parent_signal_ref),
        "event_identity_ref": build_json_artifact_ref(
            _plain(carrier.event_identity),
            artifact_kind="event_identity",
            payload_schema_version="evisoz_event_identity_v1",
        ),
        "montage_receipt_ref": build_json_artifact_ref(
            _plain(carrier.montage_receipt),
            artifact_kind="montage_derivation_receipt",
            payload_schema_version="evisoz_montage_derivation_receipt_v1",
        ),
        "files": files,
        "receipt_sha256": _PLACEHOLDER,
    }
    return _seal_receipt(body)


def validate_stage0_dual_montage_cache_materialization_receipt(
    value: object,
    *,
    root: str | Path | None = None,
    trusted_event_identity: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate receipt seals and the deterministic file inventory."""

    if type(value) is not dict:
        raise TypeError("materialization receipt must be an object")
    required = {
        "schema_version", "materializer_version", "status", "cache_id",
        "parent_signal_ref", "event_identity_ref", "montage_receipt_ref",
        "files", "receipt_sha256",
    }
    if set(value) != required:
        raise ValueError("materialization receipt fields drifted")
    data = json.loads(canonical_json_bytes(value).decode("utf-8"))
    if data["schema_version"] != MATERIALIZATION_RECEIPT_SCHEMA_VERSION:
        raise ValueError("materialization receipt schema_version drifted")
    if data["materializer_version"] != MATERIALIZER_VERSION or data["status"] != "complete":
        raise ValueError("materialization receipt status/version drifted")
    if not isinstance(data["cache_id"], str) or not data["cache_id"].startswith("EVISOZ-STAGE0-"):
        raise ValueError("materialization cache_id is invalid")
    validate_artifact_ref(data["parent_signal_ref"])
    if data["parent_signal_ref"]["artifact_kind"] != "canonical_signal":
        raise ValueError("materialization parent reference has the wrong kind")
    event_ref = validate_artifact_ref(data["event_identity_ref"])
    if event_ref["artifact_kind"] != "event_identity":
        raise ValueError("materialization event identity reference has the wrong kind")
    montage_ref = validate_artifact_ref(data["montage_receipt_ref"])
    if montage_ref["artifact_kind"] != "montage_derivation_receipt":
        raise ValueError("materialization montage reference has the wrong kind")
    files = data["files"]
    if type(files) is not dict or not _REQUIRED_FILES.issubset(files) or not set(files).issubset(_REQUIRED_FILES | _OPTIONAL_FILES):
        raise ValueError("materialization file inventory drifted")
    for name, descriptor in files.items():
        if type(descriptor) is not dict or set(descriptor) != {"sha256", "size_bytes"}:
            raise ValueError(f"file descriptor drifted: {name}")
        if not isinstance(descriptor["sha256"], str) or len(descriptor["sha256"]) != 64:
            raise ValueError(f"file digest invalid: {name}")
        if type(descriptor["size_bytes"]) is not int or descriptor["size_bytes"] < 0:
            raise ValueError(f"file size invalid: {name}")
    if data["receipt_sha256"] != canonical_json_sha256({**data, "receipt_sha256": _PLACEHOLDER}):
        raise ValueError("materialization receipt hash drifted")
    if trusted_event_identity is not None:
        identity = validate_event_identity(_plain(trusted_event_identity))
        expected = build_json_artifact_ref(
            identity,
            artifact_kind="event_identity",
            payload_schema_version="evisoz_event_identity_v1",
        )
        if expected != data["event_identity_ref"]:
            raise ValueError("materialization receipt does not bind trusted event identity")
    if root is not None:
        replayed = open_stage0_dual_montage_cache_from_disk(
            root, trusted_event_identity=trusted_event_identity
        )
        if dict(replayed.materialization_receipt) != data:
            raise ValueError("materialization receipt does not replay from disk")
    return data


def _raw_files_from_carrier(carrier: Stage0DualMontageCarrier) -> dict[str, bytes]:
    if not isinstance(carrier, Stage0DualMontageCarrier):
        raise TypeError("carrier must be a Stage0DualMontageCarrier")
    # Re-run the receipt validator and validate all tensor invariants before
    # anything reaches disk.
    montage = validate_montage_derivation_receipt(
        _plain(carrier.montage_receipt),
        trusted_event_identity=_plain(carrier.event_identity),
    )
    del montage
    raw: dict[str, bytes] = {
        _RECEIPT_FILES["parent_signal"]: canonical_tensor_bytes(carrier.parent_signal),
        _RECEIPT_FILES["event_identity"]: canonical_json_bytes(_plain(carrier.event_identity)),
        _RECEIPT_FILES["montage_receipt"]: canonical_json_bytes(_plain(carrier.montage_receipt)),
    }
    # A formal v29 is materialized only if the montage receipt binds it.
    if carrier.montage_receipt["views"]["v29_reference"]["artifact_ref"] is not None:
        if carrier.v29_reference is None:
            raise ValueError("montage receipt references absent v29 tensor")
        raw[_TENSOR_FILES["v29_reference"]] = canonical_tensor_bytes(carrier.v29_reference)
    if carrier.montage_receipt["views"]["tcp22_context"]["artifact_ref"] is not None:
        raw[_TENSOR_FILES["tcp22_context"]] = canonical_tensor_bytes(carrier.tcp22_context)
        raw[_TENSOR_FILES["tcp22_onset"]] = canonical_tensor_bytes(carrier.tcp22_onset)
    return raw


def materialize_stage0_dual_montage_cache_to_disk(
    destination: str | Path,
    carrier: Stage0DualMontageCarrier | None = None,
    *,
    _write_hook: Any | None = None,
) -> Stage0DualMontageCacheMaterialization:
    """Precommit-replay and atomically publish one deterministic cache tree."""

    if carrier is None:
        raise TypeError("carrier is required")
    target = Path(destination).absolute()
    parent_fd = _open_absolute_directory(target.parent, context="Stage0 cache parent")
    staging_name: str | None = None
    committed = False
    try:
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Stage0 cache destination already exists: {target}")
        raw = _raw_files_from_carrier(carrier)
        # Keep formal file set stable: absent views are represented by absent
        # tensor files and their receipt artifact_ref is null.
        receipt = _build_materialization_receipt(raw, carrier)
        staging_name = _create_staging_directory_at(parent_fd, target.name)
        staging_fd = os.open(staging_name, _directory_open_flags(), dir_fd=parent_fd)
        directory_fds: dict[str, int] = {"": staging_fd}
        try:
            for name in sorted(_EXPECTED_DIRS):
                os.mkdir(name, mode=0o700, dir_fd=staging_fd)
                directory_fds[name] = os.open(name, _directory_open_flags(), dir_fd=staging_fd)
            for relative, payload in sorted(raw.items()):
                directory, leaf = relative.rsplit("/", 1)
                _write_exclusive_bytes_at(directory_fds[directory], leaf, payload)
            if _write_hook is not None:
                _write_hook(target.parent / staging_name)
            _write_exclusive_bytes_at(
                directory_fds["audit"],
                "materialization_receipt.json",
                canonical_json_bytes(receipt),
            )
            for descriptor in directory_fds.values():
                _fsync_directory_fd(descriptor)
        finally:
            for descriptor in directory_fds.values():
                os.close(descriptor)
        opened = open_stage0_dual_montage_cache_from_disk(
            target.parent / staging_name,
            trusted_event_identity=_plain(carrier.event_identity),
        )
        if dict(opened.materialization_receipt) != receipt:
            raise ValueError("staged materialization receipt changed during replay")
        result = Stage0DualMontageCacheMaterialization(
            path=target,
            opened=opened,
            materialization_receipt=_deep_freeze(receipt),
        )
        _fsync_directory_fd(parent_fd)
        _rename_directory_noreplace_at(parent_fd, staging_name, target.name)
        committed = True
        return result
    finally:
        if not committed and staging_name is not None:
            try:
                shutil.rmtree(staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        try:
            os.close(parent_fd)
        except OSError:
            if not committed:
                raise


def open_stage0_dual_montage_cache_from_disk(
    destination: str | Path,
    *,
    trusted_event_identity: Mapping[str, object] | None = None,
) -> OpenedStage0DualMontageCache:
    """Capture and fully replay one published Stage0 cache directory."""

    snapshot = _capture_disk_snapshot(destination)
    if snapshot.directories != frozenset(_EXPECTED_DIRS):
        raise ValueError("Stage0 cache directory set drifted")
    receipt_raw = snapshot.raw_files[_MATERIALIZATION_FILE]
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("materialization receipt is not JSON") from exc
    if receipt_raw != canonical_json_bytes(receipt):
        raise ValueError("materialization receipt is not canonical JSON")
    receipt = validate_stage0_dual_montage_cache_materialization_receipt(
        receipt, trusted_event_identity=trusted_event_identity
    )
    expected_files = set(receipt["files"]) | {_MATERIALIZATION_FILE}
    if set(snapshot.raw_files) != expected_files:
        raise ValueError("Stage0 cache file set drifted")
    for name, descriptor in receipt["files"].items():
        raw = snapshot.raw_files[name]
        if descriptor["sha256"] != sha256_bytes(raw) or descriptor["size_bytes"] != len(raw):
            raise ValueError(f"file bytes do not replay: {name}")
    parent = _decode_canonical_tensor(snapshot.raw_files[_RECEIPT_FILES["parent_signal"]], context="parent_signal")
    _canonical_input(parent, context="parent_signal", shape=(21, 12000))
    expected_parent_ref = _parent_ref(parent)
    if expected_parent_ref != receipt["parent_signal_ref"]:
        raise ValueError("parent tensor reference does not replay")
    identity_raw = snapshot.raw_files[_RECEIPT_FILES["event_identity"]]
    identity = json.loads(identity_raw.decode("utf-8"))
    if identity_raw != canonical_json_bytes(identity):
        raise ValueError("event identity sidecar is not canonical JSON")
    identity = validate_event_identity(identity)
    if trusted_event_identity is not None and identity != validate_event_identity(_plain(trusted_event_identity)):
        raise ValueError("disk event identity differs from trusted event identity")
    montage_raw = snapshot.raw_files[_RECEIPT_FILES["montage_receipt"]]
    montage = json.loads(montage_raw.decode("utf-8"))
    if montage_raw != canonical_json_bytes(montage):
        raise ValueError("montage receipt sidecar is not canonical JSON")
    montage = validate_montage_derivation_receipt(montage, trusted_event_identity=identity)
    if montage["parent_signal_ref"] != receipt["parent_signal_ref"]:
        raise ValueError("receipt parent binding drifted")
    if build_json_artifact_ref(identity, artifact_kind="event_identity", payload_schema_version="evisoz_event_identity_v1") != receipt["event_identity_ref"]:
        raise ValueError("event identity reference does not replay")
    if build_json_artifact_ref(montage, artifact_kind="montage_derivation_receipt", payload_schema_version="evisoz_montage_derivation_receipt_v1") != receipt["montage_receipt_ref"]:
        raise ValueError("montage reference does not replay")
    expected_cache_id = "EVISOZ-STAGE0-" + canonical_json_sha256(_plain(montage))[:24]
    if receipt["cache_id"] != expected_cache_id:
        raise ValueError("materialization cache_id does not replay")
    tensors: dict[str, bytes] = {}
    decoded: dict[str, torch.Tensor] = {}
    for view, relative in _TENSOR_FILES.items():
        if relative in snapshot.raw_files:
            tensor = _decode_canonical_tensor(snapshot.raw_files[relative], context=view)
            expected = montage["views"][view]["shape"]
            if list(tensor.shape) != expected or tensor.dtype is not torch.float32:
                raise ValueError(f"{view} tensor geometry drifted")
            if build_raw_artifact_ref(snapshot.raw_files[relative], artifact_kind="tensor_cache", media_type="application/x-evisoz-canonical-tensor", payload_schema_version=TENSOR_SCHEMA_VERSION) != montage["views"][view]["artifact_ref"]:
                raise ValueError(f"{view} tensor reference does not replay")
            tensors[view] = snapshot.raw_files[relative]
            decoded[view] = tensor
        elif montage["views"][view]["artifact_ref"] is not None:
            raise ValueError(f"missing materialized {view}")

    # Replay every derived value from the immutable parent.  The observability
    # receipt is intentionally the only source of the auxiliary A1/A2 mask;
    # it contains labels, while its source hash is bound to the full parent.
    observed = torch.tensor(
        list(montage["views"]["v29_reference"]["unit_observed_mask"])
        if montage["views"]["v29_reference"]["unit_observed_mask"]
        else [False] * 19,
        dtype=torch.bool,
    )
    observed = torch.cat((observed, torch.zeros(2, dtype=torch.bool)))
    binding = montage["reference_observability"]
    if binding is not None:
        reference_receipt = binding["receipt"]
        if (
            reference_receipt.get("schema_version")
            == OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION
        ):
            for electrode in reference_receipt["observed_parent_electrodes"]:
                if electrode in _PARENT_LABELS:
                    observed[_PARENT_LABELS.index(electrode)] = True
        else:
            for row in reference_receipt["signal_label_observations"]:
                try:
                    electrode = normalize_electrode_name(row["raw_label"])
                except (TypeError, ValueError):
                    continue
                if electrode in _PARENT_LABELS:
                    observed[_PARENT_LABELS.index(electrode)] = True
    if torch.any(parent[~observed] != 0):
        raise ValueError("unobserved parent rows are not zero")
    registry = build_default_channel_registry()
    expected_tcp = torch.zeros((22, 12000), dtype=torch.float32)
    expected_onset = torch.zeros((22, 2000), dtype=torch.float32)
    expected_edge_mask = torch.zeros(22, dtype=torch.bool)
    for index, edge in enumerate(registry["tcp22_derivations"]):
        positive = str(edge["positive_electrode"]["normalized"])
        negative = str(edge["negative_electrode"]["normalized"])
        pi, ni = _PARENT_LABELS.index(positive), _PARENT_LABELS.index(negative)
        if bool(observed[pi].item() and observed[ni].item()):
            expected_tcp[index] = parent[pi] - parent[ni]
            expected_onset[index] = (
                parent[pi, 2000:4000] - parent[ni, 2000:4000]
            )
            expected_edge_mask[index] = True
        support = montage["edge_support"][index]
        if support["support_state"] == "unobserved" and expected_edge_mask[index]:
            raise ValueError("montage edge support masks an observable endpoint pair")
        if (support["support_state"] != "unobserved") != bool(expected_edge_mask[index].item()):
            raise ValueError("montage edge support does not replay from parent mask")
    if "tcp22_context" in decoded:
        if not torch.equal(decoded["tcp22_context"], expected_tcp):
            raise ValueError("TCP22 context does not replay from direct parent")
        if not torch.equal(decoded["tcp22_onset"], expected_onset):
            raise ValueError("TCP22 onset does not replay from the direct parent crop")
    elif bool(torch.any(expected_edge_mask).item()):
        raise ValueError("TCP22 support exists but tensors are absent")
    if "v29_reference" in decoded:
        if not bool(torch.all(observed[:19]).item()):
            raise ValueError("v29 tensor exists without complete Standard19 support")
        expected_car = parent[:19] - parent[:19].mean(dim=0)
        if not torch.allclose(decoded["v29_reference"], expected_car, rtol=0.0, atol=1e-12):
            raise ValueError("v29 tensor does not replay Standard19-only CAR19")
    return OpenedStage0DualMontageCache(
        materialization_receipt=_deep_freeze(receipt),
        montage_receipt=_deep_freeze(montage),
        _snapshots=MappingProxyType(tensors),
    )


__all__ = [
    "STAGE0_DUAL_MONTAGE_CACHE_SCHEMA_VERSION",
    "MATERIALIZATION_RECEIPT_SCHEMA_VERSION",
    "Stage0DualMontageCarrier",
    "OpenedStage0DualMontageCache",
    "Stage0DualMontageCacheMaterialization",
    "build_common_reference_event_carrier",
    "materialize_stage0_dual_montage_cache_to_disk",
    "open_stage0_dual_montage_cache_from_disk",
    "validate_stage0_dual_montage_cache_materialization_receipt",
]
