"""Closed source-train-only I/V evidence for fold-local recovery experiments.

The exporter accepts only the strictly reloaded, frozen v1.1 development
capability.  That parent artifact is verified once, after which only its
``source_train`` split is copied into a new bundle.  The new bundle has no
other-split roster, field, or tensor and can be consumed without opening the
shared v1.1 safetensors file.

SOZ target values are deliberately absent.  A separate join helper can open
the already-published *train child* of the split-physical target scope and
returns one complete patient bag plus its fixed patient-fold carrier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from . import development_reasoner as _v1
from . import development_reasoner_v1_1 as _v11
from .data.deepsoz import normalize_patient_id
from .development_target_scope_v1_1 import (
    LoadedDevelopmentTargetScopeV11,
    load_development_target_scope_v1_1,
)


SOURCE_TRAIN_IV_CAPABILITY_SCHEMA = "soz_source_train_only_iv_capability_v1"
SOURCE_TRAIN_IV_RECEIPT_SCHEMA = (
    "soz_source_train_only_iv_capability_receipt_v1"
)
SOURCE_TRAIN_IV_EVENT_SCHEMA = "soz_source_train_only_iv_event_roster_v1"
SOURCE_TRAIN_IV_PURPOSE = (
    "fold_local_development_source_train_target_free_iv_evidence"
)
SOURCE_TRAIN_IV_MANIFEST_FILENAME = "manifest.json"
SOURCE_TRAIN_IV_EVENTS_FILENAME = "events.json"
SOURCE_TRAIN_IV_TENSORS_FILENAME = "evidence.safetensors"

EXPECTED_SOURCE_TRAIN_PATIENT_COUNT = 65
EXPECTED_SOURCE_TRAIN_EVENT_COUNT = 582
EXPECTED_OUTER_FOLDS = tuple(range(5))

_FILES = frozenset(
    {
        SOURCE_TRAIN_IV_MANIFEST_FILENAME,
        SOURCE_TRAIN_IV_EVENTS_FILENAME,
        SOURCE_TRAIN_IV_TENSORS_FILENAME,
    }
)
_TENSOR_NAMES = (
    "evolution",
    "ictal",
    "evolution_mask",
    "ictal_mask",
    "phase_mask",
    "reliability",
    "event_abstain",
)
_EVENT_FIELDS = frozenset({"event_id", "patient_id", "oof_fold"})
_EVENT_DOCUMENT_FIELDS = frozenset({"schema_version", "model_split", "events"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "serialization",
        "model_split",
        "source_train_only",
        "development_only",
        "target_values_loaded",
        "source_eval_used",
        "private_used",
        "formal_reasoner_authorized",
        "formal_promotion",
        "event_count",
        "patient_count",
        "receipt",
        "receipt_sha256",
        "tensor_specs",
        "files",
    }
)
_LINEAGE_FIELDS = frozenset(
    {
        "parent_v1_1_manifest_sha256",
        "parent_v1_1_authorization_receipt_sha256",
        "base_v1_manifest_sha256",
        "base_v1_authorization_receipt_sha256",
        "eligibility_amendment_artifact_sha256",
        "eligibility_amendment_receipt_sha256",
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "verified_target_header_artifact_sha256",
        "verified_target_header_receipt_sha256",
        "verified_target_header_policy_sha256",
    }
)
_FORBIDDEN_SPLIT_TOKEN = b"source_dev"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_TENSOR_BYTES = 512 * 1024 * 1024


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
        raise ValueError("Source-train I/V artifact is not canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(raw: bytes, *, field_name: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field_name} contains duplicate field {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"{field_name} contains non-finite constant {value}")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must be strict UTF-8 JSON") from exc


def _absolute_no_symlink(path: str | Path, *, field_name: str) -> Path:
    result = Path(os.path.abspath(path))
    for component in (result, *result.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field_name} cannot traverse symlinks")
    return result


def _safe_new_directory(path: str | Path) -> Path:
    target = _absolute_no_symlink(path, field_name="Source-train I/V output")
    if target.name in {"", ".", ".."}:
        raise ValueError("Source-train I/V output requires a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    return target


def _stable_file(path: Path, *, field_name: str, maximum_bytes: int) -> bytes:
    source = _absolute_no_symlink(path, field_name=field_name)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field_name} must be a regular non-symlinked file")
    before = source.stat()
    if not 1 <= before.st_size <= maximum_bytes:
        raise ValueError(f"{field_name} has an invalid size")
    raw = source.read_bytes()
    after = source.stat()
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
        raise RuntimeError(f"{field_name} changed while it was read")
    return raw


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _evidence_tensors(
    evidence: _v1.DevelopmentIVEvidenceBatch,
) -> dict[str, torch.Tensor]:
    return {
        name: getattr(evidence, name).detach().cpu().contiguous().clone()
        for name in _TENSOR_NAMES
    }


def _tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "tensor_sha256": _v1._tensor_sha256(name, tensor),
        }
        for name, tensor in sorted(tensors.items())
    }


def _event_payload(split: _v1._DevelopmentSplitEvidence) -> dict[str, object]:
    return {
        "schema_version": SOURCE_TRAIN_IV_EVENT_SCHEMA,
        "model_split": "source_train",
        "events": [
            {
                "event_id": event_id,
                "patient_id": patient_id,
                "oof_fold": fold,
            }
            for event_id, patient_id, fold in zip(
                split.event_ids,
                split.patient_ids_by_event,
                split.oof_folds,
            )
        ],
    }


def _fold_assignment_sha256(
    patient_ids_by_event: Sequence[str], oof_folds: Sequence[int | None]
) -> str:
    assignments: dict[str, int] = {}
    for patient_id, fold in zip(patient_ids_by_event, oof_folds):
        if fold is None:
            raise ValueError("Every source-train event requires an OOF fold")
        previous = assignments.setdefault(patient_id, int(fold))
        if previous != int(fold):
            raise ValueError("A source-train patient crosses OOF folds")
    return _canonical_sha256(tuple(sorted(assignments.items())))


@dataclass(frozen=True)
class SourceTrainIVCapabilityReceipt:
    lineage: Mapping[str, str]
    source_train_evidence_receipt_sha256: str
    source_train_event_roster_sha256: str
    source_train_event_set_sha256: str
    source_train_patient_roster_sha256: str
    event_order_sha256: str
    patient_fold_assignment_sha256: str
    evidence_sha256: str
    event_count: int = EXPECTED_SOURCE_TRAIN_EVENT_COUNT
    patient_count: int = EXPECTED_SOURCE_TRAIN_PATIENT_COUNT
    target_values_loaded: bool = False
    source_eval_used: bool = False
    private_used: bool = False
    development_only: bool = True
    formal_reasoner_authorized: bool = False
    formal_promotion: bool = False
    schema_version: str = SOURCE_TRAIN_IV_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, Mapping) or set(self.lineage) != set(
            _LINEAGE_FIELDS
        ):
            raise ValueError("Source-train I/V lineage violates its closed schema")
        lineage = {
            name: _require_sha256(value, field_name=f"lineage.{name}")
            for name, value in self.lineage.items()
        }
        frozen = {
            "parent_v1_1_manifest_sha256": (
                _v11.FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256
            ),
            "parent_v1_1_authorization_receipt_sha256": (
                _v11.FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256
            ),
            "base_v1_manifest_sha256": _v11.FROZEN_BASE_V1_MANIFEST_SHA256,
            "base_v1_authorization_receipt_sha256": (
                _v11.FROZEN_BASE_V1_AUTHORIZATION_RECEIPT_SHA256
            ),
            "eligibility_amendment_artifact_sha256": (
                _v11.FROZEN_AMENDMENT_ARTIFACT_SHA256
            ),
            "eligibility_amendment_receipt_sha256": (
                _v11.FROZEN_AMENDMENT_RECEIPT_SHA256
            ),
            "signal_preflight_artifact_sha256": (
                _v11.FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256
            ),
            "signal_preflight_receipt_sha256": (
                _v11.FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256
            ),
            "oof_protocol_artifact_sha256": (
                _v11.FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256
            ),
            "oof_protocol_receipt_sha256": (
                _v11.FROZEN_OOF_PROTOCOL_RECEIPT_SHA256
            ),
            "verified_target_header_artifact_sha256": (
                _v11.FROZEN_TARGET_V2_ARTIFACT_SHA256
            ),
            "verified_target_header_receipt_sha256": (
                _v11.FROZEN_TARGET_V2_RECEIPT_SHA256
            ),
            "verified_target_header_policy_sha256": (
                _v11.FROZEN_TARGET_V2_POLICY_SHA256
            ),
        }
        changed = tuple(
            name for name, expected in frozen.items() if lineage[name] != expected
        )
        if changed:
            raise ValueError(f"Source-train I/V lineage changed: {changed}")
        object.__setattr__(self, "lineage", lineage)
        for name in (
            "source_train_evidence_receipt_sha256",
            "source_train_event_roster_sha256",
            "source_train_event_set_sha256",
            "source_train_patient_roster_sha256",
            "event_order_sha256",
            "patient_fold_assignment_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        if self.source_train_event_set_sha256 != (
            _v11.FROZEN_SIGNAL_SOURCE_TRAIN_EVENT_SET_SHA256
        ):
            raise ValueError("Source-train event set changed")
        if self.event_count != EXPECTED_SOURCE_TRAIN_EVENT_COUNT or (
            self.patient_count != EXPECTED_SOURCE_TRAIN_PATIENT_COUNT
        ):
            raise ValueError("Source-train I/V count changed")
        fixed = {
            "target_values_loaded": False,
            "source_eval_used": False,
            "private_used": False,
            "development_only": True,
            "formal_reasoner_authorized": False,
            "formal_promotion": False,
            "schema_version": SOURCE_TRAIN_IV_RECEIPT_SCHEMA,
        }
        if any(getattr(self, name) != value for name, value in fixed.items()):
            raise ValueError("Source-train I/V scientific boundary changed")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class PublishedSourceTrainIVCapability:
    path: Path
    manifest_sha256: str
    receipt: SourceTrainIVCapabilityReceipt
    split: _v1._DevelopmentSplitEvidence = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_sha256",
            _require_sha256(self.manifest_sha256, field_name="manifest_sha256"),
        )
        self.assert_unchanged()

    @property
    def evidence(self) -> _v1.DevelopmentIVEvidenceBatch:
        return self.split.evidence

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return self.split.patient_ids

    @property
    def event_ids(self) -> tuple[str, ...]:
        return self.split.event_ids

    def assert_unchanged(self) -> None:
        if self.split.model_split != "source_train":
            raise ValueError("Source-train I/V split changed")
        if self.split.evidence.batch_size != self.receipt.event_count or len(
            self.split.patient_ids
        ) != self.receipt.patient_count:
            raise ValueError("Source-train I/V in-memory count changed")
        checks = {
            "evidence receipt": self.split.receipt_sha256
            == self.receipt.source_train_evidence_receipt_sha256,
            "event order": _canonical_sha256(self.split.event_ids)
            == self.receipt.event_order_sha256
            == self.receipt.source_train_event_roster_sha256,
            "event set": _canonical_sha256(tuple(sorted(self.split.event_ids)))
            == self.receipt.source_train_event_set_sha256,
            "patient roster": _canonical_sha256(self.split.patient_ids)
            == self.receipt.source_train_patient_roster_sha256,
            "fold assignment": _fold_assignment_sha256(
                self.split.patient_ids_by_event, self.split.oof_folds
            )
            == self.receipt.patient_fold_assignment_sha256,
            "evidence": _v1._evidence_batch_sha256(self.split.evidence)
            == self.receipt.evidence_sha256,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"Source-train I/V changed in memory: {failed}")


def _lineage_from_parent(
    parent: _v11.PublishedDevelopmentIVEvidenceCapabilityV11,
) -> dict[str, str]:
    receipt = parent.capability.receipt
    return {
        "parent_v1_1_manifest_sha256": parent.manifest_sha256,
        "parent_v1_1_authorization_receipt_sha256": (
            parent.authorization_receipt_sha256
        ),
        "base_v1_manifest_sha256": receipt.base_v1_manifest_sha256,
        "base_v1_authorization_receipt_sha256": (
            receipt.base_v1_authorization_receipt_sha256
        ),
        "eligibility_amendment_artifact_sha256": receipt.amendment_artifact_sha256,
        "eligibility_amendment_receipt_sha256": receipt.amendment_receipt_sha256,
        "signal_preflight_artifact_sha256": receipt.signal_preflight_artifact_sha256,
        "signal_preflight_receipt_sha256": receipt.signal_preflight_receipt_sha256,
        "oof_protocol_artifact_sha256": receipt.oof_protocol_artifact_sha256,
        "oof_protocol_receipt_sha256": receipt.oof_protocol_receipt_sha256,
        "verified_target_header_artifact_sha256": (
            receipt.verified_target_v2_artifact_sha256
        ),
        "verified_target_header_receipt_sha256": (
            receipt.verified_target_v2_receipt_sha256
        ),
        "verified_target_header_policy_sha256": (
            receipt.verified_target_v2_policy_sha256
        ),
    }


def _receipt_from_parent(
    parent: _v11.PublishedDevelopmentIVEvidenceCapabilityV11,
) -> SourceTrainIVCapabilityReceipt:
    parent.capability.assert_unchanged()
    if parent.manifest_sha256 != _v11.FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256 or (
        parent.authorization_receipt_sha256
        != _v11.FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256
    ):
        raise ValueError("Source-train export requires the frozen v1.1 parent")
    split = parent.capability.base.capability.source_train
    authorization = parent.capability.receipt
    receipt = SourceTrainIVCapabilityReceipt(
        lineage=_lineage_from_parent(parent),
        source_train_evidence_receipt_sha256=(
            authorization.source_train_evidence_receipt_sha256
        ),
        source_train_event_roster_sha256=(
            authorization.source_train_event_roster_sha256
        ),
        source_train_event_set_sha256=authorization.source_train_event_set_sha256,
        source_train_patient_roster_sha256=(
            authorization.source_train_patient_roster_sha256
        ),
        event_order_sha256=_canonical_sha256(split.event_ids),
        patient_fold_assignment_sha256=_fold_assignment_sha256(
            split.patient_ids_by_event, split.oof_folds
        ),
        evidence_sha256=_v1._evidence_batch_sha256(split.evidence),
    )
    candidate = PublishedSourceTrainIVCapability(
        path=Path("."),
        manifest_sha256="0" * 64,
        receipt=receipt,
        split=split,
    )
    candidate.assert_unchanged()
    return receipt


def _manifest_payload(
    receipt: SourceTrainIVCapabilityReceipt,
    *,
    tensor_specs: Mapping[str, object],
    files_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SOURCE_TRAIN_IV_CAPABILITY_SCHEMA,
        "purpose": SOURCE_TRAIN_IV_PURPOSE,
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "model_split": "source_train",
        "source_train_only": True,
        "development_only": True,
        "target_values_loaded": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "event_count": receipt.event_count,
        "patient_count": receipt.patient_count,
        "receipt": asdict(receipt),
        "receipt_sha256": receipt.receipt_sha256,
        "tensor_specs": dict(tensor_specs),
        "files": dict(files_payload),
    }


def publish_source_train_iv_capability_from_v1_1(
    parent: _v11.PublishedDevelopmentIVEvidenceCapabilityV11,
    output_directory: str | Path,
) -> PublishedSourceTrainIVCapability:
    """Verify the frozen v1.1 parent, then atomically publish only train data."""

    if type(parent) is not _v11.PublishedDevelopmentIVEvidenceCapabilityV11:
        raise TypeError("Source-train export requires the strict v1.1 loader")
    receipt = _receipt_from_parent(parent)
    split = parent.capability.base.capability.source_train
    tensors = _evidence_tensors(split.evidence)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    target = _safe_new_directory(output_directory)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / SOURCE_TRAIN_IV_TENSORS_FILENAME
        save_file(tensors, str(tensor_path))
        event_raw = _canonical_json_bytes(_event_payload(split))
        if _FORBIDDEN_SPLIT_TOKEN in event_raw:
            raise RuntimeError("Source-train event export contains a forbidden split token")
        event_path = staging / SOURCE_TRAIN_IV_EVENTS_FILENAME
        event_path.write_bytes(event_raw)
        tensor_size = tensor_path.stat().st_size
        if not 1 <= tensor_size <= _MAX_TENSOR_BYTES:
            raise ValueError("Source-train I/V tensor file has an invalid size")
        if not 1 <= len(event_raw) <= _MAX_JSON_BYTES:
            raise ValueError("Source-train I/V event file has an invalid size")
        files_payload = {
            SOURCE_TRAIN_IV_EVENTS_FILENAME: {
                "sha256": hashlib.sha256(event_raw).hexdigest(),
                "size_bytes": len(event_raw),
            },
            SOURCE_TRAIN_IV_TENSORS_FILENAME: {
                "sha256": _file_sha256(tensor_path),
                "size_bytes": tensor_size,
            },
        }
        manifest = _manifest_payload(
            receipt,
            tensor_specs=_tensor_specs(tensors),
            files_payload=files_payload,
        )
        manifest_raw = _canonical_json_bytes(manifest)
        if _FORBIDDEN_SPLIT_TOKEN in manifest_raw:
            raise RuntimeError("Source-train manifest contains a forbidden split token")
        manifest_path = staging / SOURCE_TRAIN_IV_MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_raw)
        for path in (tensor_path, event_path, manifest_path):
            _fsync_file(path)
        _fsync_directory(staging)
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)
        return load_source_train_iv_capability(
            target,
            expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _receipt_from_payload(value: object) -> SourceTrainIVCapabilityReceipt:
    if not isinstance(value, Mapping):
        raise ValueError("Source-train I/V receipt must be an object")
    expected = {item.name for item in fields(SourceTrainIVCapabilityReceipt)}
    if set(value) != expected:
        raise ValueError("Source-train I/V receipt violates its closed schema")
    payload = dict(value)
    try:
        return SourceTrainIVCapabilityReceipt(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("Source-train I/V receipt is invalid") from exc


def load_source_train_iv_capability(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> PublishedSourceTrainIVCapability:
    """Load the closed bundle without opening the shared v1.1 capability."""

    try:
        from safetensors.torch import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    source = _absolute_no_symlink(directory, field_name="Source-train I/V bundle")
    if source.is_symlink() or not source.is_dir() or {
        entry.name for entry in source.iterdir()
    } != set(_FILES):
        raise ValueError("Source-train I/V bundle violates its closed file schema")
    manifest_raw = _stable_file(
        source / SOURCE_TRAIN_IV_MANIFEST_FILENAME,
        field_name="Source-train I/V manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if _FORBIDDEN_SPLIT_TOKEN in manifest_raw:
        raise ValueError("Source-train I/V manifest contains a forbidden split token")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field_name="expected_manifest_sha256"
    ):
        raise ValueError("Source-train I/V manifest SHA mismatch")
    manifest = _strict_json(manifest_raw, field_name="Source-train I/V manifest")
    if not isinstance(manifest, Mapping) or _canonical_json_bytes(manifest) != (
        manifest_raw
    ):
        raise ValueError("Source-train I/V manifest is not canonical JSON")
    if set(manifest) != set(_MANIFEST_FIELDS):
        raise ValueError("Source-train I/V manifest violates its closed schema")
    fixed = {
        "schema_version": SOURCE_TRAIN_IV_CAPABILITY_SCHEMA,
        "purpose": SOURCE_TRAIN_IV_PURPOSE,
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "model_split": "source_train",
        "source_train_only": True,
        "development_only": True,
        "target_values_loaded": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "event_count": EXPECTED_SOURCE_TRAIN_EVENT_COUNT,
        "patient_count": EXPECTED_SOURCE_TRAIN_PATIENT_COUNT,
    }
    changed = tuple(
        name for name, expected in fixed.items() if manifest.get(name) != expected
    )
    if changed:
        raise ValueError(f"Source-train I/V manifest boundary changed: {changed}")
    receipt = _receipt_from_payload(manifest["receipt"])
    if manifest["receipt_sha256"] != receipt.receipt_sha256:
        raise ValueError("Source-train I/V receipt SHA mismatch")
    files_payload = manifest["files"]
    if not isinstance(files_payload, Mapping) or set(files_payload) != {
        SOURCE_TRAIN_IV_EVENTS_FILENAME,
        SOURCE_TRAIN_IV_TENSORS_FILENAME,
    }:
        raise ValueError("Source-train I/V file receipt schema changed")
    raw_files: dict[str, bytes] = {}
    for name, maximum in (
        (SOURCE_TRAIN_IV_EVENTS_FILENAME, _MAX_JSON_BYTES),
        (SOURCE_TRAIN_IV_TENSORS_FILENAME, _MAX_TENSOR_BYTES),
    ):
        record = files_payload[name]
        if not isinstance(record, Mapping) or set(record) != {"sha256", "size_bytes"}:
            raise ValueError("Source-train I/V file receipt changed")
        raw = _stable_file(
            source / name,
            field_name=f"Source-train I/V {name}",
            maximum_bytes=maximum,
        )
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != (
            record["sha256"]
        ):
            raise ValueError(f"Source-train I/V file changed: {name}")
        raw_files[name] = raw
    event_raw = raw_files[SOURCE_TRAIN_IV_EVENTS_FILENAME]
    if _FORBIDDEN_SPLIT_TOKEN in event_raw:
        raise ValueError("Source-train I/V events contain a forbidden split token")
    event_payload = _strict_json(event_raw, field_name="Source-train I/V events")
    if not isinstance(event_payload, Mapping) or _canonical_json_bytes(
        event_payload
    ) != event_raw:
        raise ValueError("Source-train I/V events are not canonical JSON")
    if set(event_payload) != set(_EVENT_DOCUMENT_FIELDS) or (
        event_payload.get("schema_version") != SOURCE_TRAIN_IV_EVENT_SCHEMA
        or event_payload.get("model_split") != "source_train"
    ):
        raise ValueError("Source-train I/V event document schema changed")
    rows = event_payload["events"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_TRAIN_EVENT_COUNT:
        raise ValueError("Source-train I/V event count changed")
    if any(not isinstance(row, Mapping) or set(row) != set(_EVENT_FIELDS) for row in rows):
        raise ValueError("Source-train I/V event row schema changed")
    tensors = load(raw_files[SOURCE_TRAIN_IV_TENSORS_FILENAME])
    if set(tensors) != set(_TENSOR_NAMES) or any(
        _FORBIDDEN_SPLIT_TOKEN.decode("ascii") in name for name in tensors
    ):
        raise ValueError("Source-train I/V tensor keys changed")
    evidence = _v1.DevelopmentIVEvidenceBatch(
        **{name: tensors[name].detach().cpu().contiguous() for name in _TENSOR_NAMES}
    )
    split = _v1._DevelopmentSplitEvidence(
        model_split="source_train",
        event_ids=tuple(str(row["event_id"]) for row in rows),
        patient_ids_by_event=tuple(str(row["patient_id"]) for row in rows),
        oof_folds=tuple(row["oof_fold"] for row in rows),
        evidence=evidence,
    )
    if _tensor_specs(tensors) != manifest["tensor_specs"]:
        raise ValueError("Source-train I/V tensor specifications changed")
    result = PublishedSourceTrainIVCapability(
        path=source,
        manifest_sha256=manifest_sha,
        receipt=receipt,
        split=split,
    )
    expected_manifest = _manifest_payload(
        receipt,
        tensor_specs=_tensor_specs(tensors),
        files_payload=files_payload,
    )
    if _canonical_json_bytes(expected_manifest) != manifest_raw:
        raise ValueError("Source-train I/V manifest did not replay")
    return result


@dataclass(frozen=True)
class SourceTrainIVTargetJoin:
    batch: _v1.DevelopmentReasonerPatientBatch = field(repr=False)
    patient_folds: tuple[int, ...]
    evidence_manifest_sha256: str
    evidence_receipt_sha256: str
    target_receipt_file_sha256: str

    def __post_init__(self) -> None:
        folds = tuple(int(value) for value in self.patient_folds)
        object.__setattr__(self, "patient_folds", folds)
        if len(folds) != EXPECTED_SOURCE_TRAIN_PATIENT_COUNT or set(folds) != set(
            EXPECTED_OUTER_FOLDS
        ):
            raise ValueError("Source-train patient fold carrier changed")
        if len(self.batch.patient_ids) != EXPECTED_SOURCE_TRAIN_PATIENT_COUNT or (
            self.batch.evidence.batch_size != EXPECTED_SOURCE_TRAIN_EVENT_COUNT
        ):
            raise ValueError("Source-train target join count changed")
        for name in (
            "evidence_manifest_sha256",
            "evidence_receipt_sha256",
            "target_receipt_file_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )


def join_source_train_iv_target_scope(
    capability: PublishedSourceTrainIVCapability,
    target: LoadedDevelopmentTargetScopeV11,
) -> SourceTrainIVTargetJoin:
    """Join only the independently loaded train target child to train evidence."""

    if type(capability) is not PublishedSourceTrainIVCapability:
        raise TypeError("Target join requires the strict source-train I/V loader")
    if type(target) is not LoadedDevelopmentTargetScopeV11:
        raise TypeError("Target join requires the strict split target loader")
    capability.assert_unchanged()
    target.assert_unchanged()
    if target.model_split != "source_train":
        raise ValueError("Source-train I/V cannot open another target split")
    split = capability.split
    if split.patient_ids != target.receipt.patient_ids:
        raise ValueError("Source-train evidence and target patient rosters differ")
    target_lineage = {
        "artifact": target.receipt.original_target_artifact_sha256,
        "receipt": target.receipt.original_verified_target_receipt_sha256,
        "policy": target.receipt.original_target_policy_sha256,
    }
    evidence_lineage = {
        "artifact": capability.receipt.lineage[
            "verified_target_header_artifact_sha256"
        ],
        "receipt": capability.receipt.lineage[
            "verified_target_header_receipt_sha256"
        ],
        "policy": capability.receipt.lineage[
            "verified_target_header_policy_sha256"
        ],
    }
    if target_lineage != evidence_lineage:
        raise ValueError("Source-train evidence and target lineage differ")
    patient_to_index = {
        patient_id: index for index, patient_id in enumerate(split.patient_ids)
    }
    event_patient_index = torch.tensor(
        [patient_to_index[value] for value in split.patient_ids_by_event],
        dtype=torch.long,
    )
    folds_by_patient: dict[str, set[int | None]] = {}
    for patient_id, fold in zip(split.patient_ids_by_event, split.oof_folds):
        folds_by_patient.setdefault(patient_id, set()).add(fold)
    if any(len(values) != 1 or None in values for values in folds_by_patient.values()):
        raise ValueError("Source-train fold carrier is not patient-disjoint")
    patient_folds = tuple(
        int(next(iter(folds_by_patient[patient_id]))) for patient_id in split.patient_ids
    )
    target_batch = target.target_batch("source_train", split.patient_ids)
    expected_counts = torch.bincount(
        event_patient_index, minlength=len(split.patient_ids)
    )
    batch = _v1.DevelopmentReasonerPatientBatch(
        _verification_marker=_v1._PATIENT_BATCH_MARKER,
        evidence=split.evidence,
        event_patient_index=event_patient_index,
        patient_ids=split.patient_ids,
        event_ids=split.event_ids,
        expected_event_counts=expected_counts,
        targets=target_batch.values,
        target_mask=target_batch.mask,
    )
    return SourceTrainIVTargetJoin(
        batch=batch,
        patient_folds=patient_folds,
        evidence_manifest_sha256=capability.manifest_sha256,
        evidence_receipt_sha256=capability.receipt.receipt_sha256,
        target_receipt_file_sha256=target.receipt_file_sha256,
    )


def load_and_join_source_train_iv_target_scope(
    capability_directory: str | Path,
    target_directory: str | Path,
    *,
    expected_capability_manifest_sha256: str,
    expected_target_receipt_file_sha256: str,
) -> SourceTrainIVTargetJoin:
    """Strictly load only two train-only child bundles and join them."""

    capability = load_source_train_iv_capability(
        capability_directory,
        expected_manifest_sha256=expected_capability_manifest_sha256,
    )
    target = load_development_target_scope_v1_1(
        target_directory,
        expected_model_split="source_train",
        expected_receipt_file_sha256=expected_target_receipt_file_sha256,
    )
    return join_source_train_iv_target_scope(capability, target)


__all__ = [
    "EXPECTED_SOURCE_TRAIN_EVENT_COUNT",
    "EXPECTED_SOURCE_TRAIN_PATIENT_COUNT",
    "PublishedSourceTrainIVCapability",
    "SOURCE_TRAIN_IV_CAPABILITY_SCHEMA",
    "SOURCE_TRAIN_IV_EVENTS_FILENAME",
    "SOURCE_TRAIN_IV_MANIFEST_FILENAME",
    "SOURCE_TRAIN_IV_TENSORS_FILENAME",
    "SourceTrainIVCapabilityReceipt",
    "SourceTrainIVTargetJoin",
    "join_source_train_iv_target_scope",
    "load_and_join_source_train_iv_target_scope",
    "load_source_train_iv_capability",
    "publish_source_train_iv_capability_from_v1_1",
]
