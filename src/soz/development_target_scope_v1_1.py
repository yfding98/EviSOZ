"""Split-physical development-only DeepSOZ target scopes.

The exporter is the sole process allowed to read the verified full target-v2
registry and the target-free signal-eligibility amendment.  It atomically
publishes two independent closed child bundles:

``train/``
    The amendment-signed 65-patient source-train target scope.
``dev/``
    The amendment-signed 16-patient source-dev diagnostic target scope.

Each consumer receives one child directory.  Its receipt, patient roster and
safetensors file contain no identity, filename or target from the other split.
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

from .data.deepsoz import SOZTargetBatch, normalize_patient_id
from .data.deepsoz_target_v2 import (
    TARGET_V2_POLICY_SHA256,
    VerifiedDeepSOZTargetV2Artifact,
    _build_target_frame,
    _target_csv_bytes,
)
from .development_reasoner_v1_1 import (
    AMENDMENT_FILENAME,
    FROZEN_AMENDMENT_ARTIFACT_SHA256,
    FROZEN_AMENDMENT_RECEIPT_SHA256,
    _receipt_from_payload as _amendment_receipt_from_payload,
)
from .geometry import CHANNEL_INDEX, STANDARD_19


DEVELOPMENT_TARGET_SCOPE_SCHEMA_V1_1 = (
    "soz_development_deepsoz_split_target_scope_v1_1"
)
DEVELOPMENT_TARGET_SCOPE_PURPOSE_V1_1 = (
    "development_reasoner_signal_eligible_split_target_only"
)
DEVELOPMENT_TARGET_SCOPE_RECEIPT_FILENAME = "receipt.json"
DEVELOPMENT_TARGET_SCOPE_TENSORS_FILENAME = "targets.safetensors"
DEVELOPMENT_TARGET_SCOPE_TRAIN_DIRECTORY = "train"
DEVELOPMENT_TARGET_SCOPE_DEV_DIRECTORY = "dev"

FROZEN_ORIGINAL_TARGET_ARTIFACT_SHA256 = (
    "5c01591c20328fb60817099cac669032bd743e36f47df77ac390842e9a2c67ed"
)
FROZEN_ORIGINAL_VERIFIED_TARGET_RECEIPT_SHA256 = (
    "80f2b71cfdf23d604849b2d1a52cc36f0b01c593906e3cef74e79d425cc442d3"
)
FROZEN_ORIGINAL_TARGET_POLICY_SHA256 = (
    "bc953272edf638150a7800b01be01261d7b96dfc6db5def5b98cfd6b93dea237"
)
if FROZEN_ORIGINAL_TARGET_POLICY_SHA256 != TARGET_V2_POLICY_SHA256:
    raise RuntimeError("Development target policy trust anchor drifted")

EXPECTED_FULL_ELIGIBLE_PATIENT_COUNT = 106
EXPECTED_TARGET_HEADER_COUNTS = {"source_train": 69, "source_dev": 16}
EXPECTED_SCOPE_PATIENT_COUNTS = {"source_train": 65, "source_dev": 16}
EXPECTED_OMITTED_SOURCE_EVAL_PATIENT_COUNT = 21

_ALLOWED_SPLITS = tuple(EXPECTED_SCOPE_PATIENT_COUNTS)
_SPLIT_DIRECTORY = {
    "source_train": DEVELOPMENT_TARGET_SCOPE_TRAIN_DIRECTORY,
    "source_dev": DEVELOPMENT_TARGET_SCOPE_DEV_DIRECTORY,
}
_CHILD_FILES = frozenset(
    {
        DEVELOPMENT_TARGET_SCOPE_RECEIPT_FILENAME,
        DEVELOPMENT_TARGET_SCOPE_TENSORS_FILENAME,
    }
)
_ROOT_ENTRIES = frozenset(_SPLIT_DIRECTORY.values())
_TENSOR_NAMES = ("values", "mask")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_MAX_TENSOR_BYTES = 8 * 1024 * 1024
_MAX_AMENDMENT_BYTES = 64 * 1024 * 1024
_AMENDMENT_VIEW_MARKER = object()


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
        raise ValueError("Development target scope is not canonical JSON data") from exc


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


def _reject_symlink_components(path: Path, *, field_name: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field_name} cannot contain symlink components")
    return absolute


def _safe_new_directory(value: str | Path, *, field_name: str) -> Path:
    target = _reject_symlink_components(Path(value), field_name=field_name)
    if target.name in {"", ".", ".."}:
        raise ValueError(f"{field_name} must be a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(f"{field_name} already exists: {target}")
    parent = _reject_symlink_components(target.parent, field_name=f"{field_name} parent")
    if not parent.is_dir():
        raise FileNotFoundError(f"{field_name} parent does not exist")
    return target


def _read_stable_regular_file(
    path: Path,
    *,
    field_name: str,
    maximum_bytes: int,
) -> bytes:
    source = _reject_symlink_components(path, field_name=field_name)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field_name} must be a regular non-symlinked file")
    before = source.stat()
    if not 1 <= before.st_size <= maximum_bytes:
        raise ValueError(f"{field_name} has an invalid size")
    payload = source.read_bytes()
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
    return payload


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _strict_json(payload: bytes, *, field_name: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must be strict UTF-8 JSON") from exc


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalized_roster(
    values: Sequence[object], *, field_name: str
) -> tuple[str, ...]:
    roster = tuple(normalize_patient_id(value) for value in values)
    if not roster or roster != tuple(sorted(roster)) or len(set(roster)) != len(roster):
        raise ValueError(f"{field_name} must be non-empty, sorted, and unique")
    return roster


def _roster_sha256(model_split: str, patient_ids: Sequence[str]) -> str:
    return _canonical_sha256(
        {"model_split": model_split, "patient_ids": tuple(patient_ids)}
    )


def _tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = _canonical_json_bytes(
        {
            "name": name,
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSignalEligibilityViewV11:
    _verification_marker: object = field(repr=False, compare=False)
    artifact_sha256: str
    receipt_sha256: str
    verified_target_v2_artifact_sha256: str
    verified_target_v2_receipt_sha256: str
    verified_target_v2_policy_sha256: str
    target_header_source_train_patient_ids: tuple[str, ...]
    target_header_source_dev_patient_ids: tuple[str, ...]
    signal_evidence_source_train_patient_ids: tuple[str, ...]
    signal_evidence_source_dev_patient_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self._verification_marker is not _AMENDMENT_VIEW_MARKER:
            raise TypeError("Signal-eligibility view requires the strict amendment loader")
        for name in (
            "artifact_sha256",
            "receipt_sha256",
            "verified_target_v2_artifact_sha256",
            "verified_target_v2_receipt_sha256",
            "verified_target_v2_policy_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        if (
            self.artifact_sha256 != FROZEN_AMENDMENT_ARTIFACT_SHA256
            or self.receipt_sha256 != FROZEN_AMENDMENT_RECEIPT_SHA256
            or self.verified_target_v2_artifact_sha256
            != FROZEN_ORIGINAL_TARGET_ARTIFACT_SHA256
            or self.verified_target_v2_receipt_sha256
            != FROZEN_ORIGINAL_VERIFIED_TARGET_RECEIPT_SHA256
            or self.verified_target_v2_policy_sha256
            != FROZEN_ORIGINAL_TARGET_POLICY_SHA256
        ):
            raise ValueError("Signal-eligibility export lineage changed")
        rosters = {}
        for name in (
            "target_header_source_train_patient_ids",
            "target_header_source_dev_patient_ids",
            "signal_evidence_source_train_patient_ids",
            "signal_evidence_source_dev_patient_ids",
        ):
            roster = _normalized_roster(getattr(self, name), field_name=name)
            object.__setattr__(self, name, roster)
            rosters[name] = roster
        if len(rosters["target_header_source_train_patient_ids"]) != 69:
            raise ValueError("Amendment target-header train count changed")
        if len(rosters["signal_evidence_source_train_patient_ids"]) != 65:
            raise ValueError("Amendment signal-evidence train count changed")
        if len(rosters["target_header_source_dev_patient_ids"]) != 16:
            raise ValueError("Amendment target-header dev count changed")
        if rosters["target_header_source_dev_patient_ids"] != rosters[
            "signal_evidence_source_dev_patient_ids"
        ]:
            raise ValueError("Amendment source-dev roster changed")
        if not set(rosters["signal_evidence_source_train_patient_ids"]) < set(
            rosters["target_header_source_train_patient_ids"]
        ):
            raise ValueError("Amendment source-train roster is not a strict subset")


def load_frozen_signal_eligibility_for_target_export_v1_1(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> FrozenSignalEligibilityViewV11:
    """Load the pinned target-free amendment for the exporter process only."""

    bundle = _reject_symlink_components(
        Path(bundle_directory), field_name="Signal-eligibility amendment bundle"
    )
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Signal-eligibility amendment must be a regular directory")
    entries = tuple(bundle.iterdir())
    if {entry.name for entry in entries} != {AMENDMENT_FILENAME} or len(entries) != 1:
        raise ValueError("Signal-eligibility amendment violates its closed schema")
    raw = _read_stable_regular_file(
        bundle / AMENDMENT_FILENAME,
        field_name="Signal-eligibility amendment",
        maximum_bytes=_MAX_AMENDMENT_BYTES,
    )
    artifact_sha = hashlib.sha256(raw).hexdigest()
    expected_artifact = _require_sha256(
        expected_artifact_sha256,
        field_name="expected_artifact_sha256",
    )
    expected_receipt = _require_sha256(
        expected_receipt_sha256,
        field_name="expected_receipt_sha256",
    )
    if artifact_sha != expected_artifact or artifact_sha != FROZEN_AMENDMENT_ARTIFACT_SHA256:
        raise ValueError("Signal-eligibility amendment artifact SHA mismatch")
    payload = _strict_json(raw, field_name="Signal-eligibility amendment")
    if not isinstance(payload, Mapping) or _canonical_json_bytes(payload) != raw:
        raise ValueError("Signal-eligibility amendment is not canonical JSON")
    required = {
        "schema_version",
        "purpose",
        "serialization",
        "policy",
        "policy_sha256",
        "receipt",
        "receipt_sha256",
        "target_values_loaded",
        "target_vectors_loaded",
        "source_eval_used",
        "private_used",
        "formal_reasoner_authorized",
        "formal_promotion",
    }
    if set(payload) != required:
        raise ValueError("Signal-eligibility amendment top-level schema changed")
    if any(
        payload[name] is not False
        for name in (
            "target_values_loaded",
            "target_vectors_loaded",
            "source_eval_used",
            "private_used",
            "formal_reasoner_authorized",
            "formal_promotion",
        )
    ):
        raise ValueError("Signal-eligibility amendment scientific boundary changed")
    receipt = _amendment_receipt_from_payload(payload["receipt"])
    if (
        payload["receipt_sha256"] != expected_receipt
        or receipt.receipt_sha256 != expected_receipt
        or expected_receipt != FROZEN_AMENDMENT_RECEIPT_SHA256
    ):
        raise ValueError("Signal-eligibility amendment receipt SHA mismatch")
    return FrozenSignalEligibilityViewV11(
        _verification_marker=_AMENDMENT_VIEW_MARKER,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt.receipt_sha256,
        verified_target_v2_artifact_sha256=receipt.verified_target_v2_artifact_sha256,
        verified_target_v2_receipt_sha256=receipt.verified_target_v2_receipt_sha256,
        verified_target_v2_policy_sha256=receipt.verified_target_v2_policy_sha256,
        target_header_source_train_patient_ids=(
            receipt.target_header_source_train_patient_ids
        ),
        target_header_source_dev_patient_ids=(
            receipt.target_header_source_dev_patient_ids
        ),
        signal_evidence_source_train_patient_ids=(
            receipt.signal_evidence_source_train_patient_ids
        ),
        signal_evidence_source_dev_patient_ids=(
            receipt.signal_evidence_source_dev_patient_ids
        ),
    )


@dataclass(frozen=True)
class DevelopmentTargetTensorReceiptV11:
    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        if self.name not in _TENSOR_NAMES:
            raise ValueError("Development target tensor name changed")
        expected_dtype = "bool" if self.name == "mask" else "float32"
        if self.dtype != expected_dtype:
            raise ValueError("Development target tensor dtype changed")
        shape = tuple(self.shape)
        if len(shape) != 2 or shape[1] != len(STANDARD_19):
            raise ValueError("Development target tensor shape changed")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, field_name=f"{self.name}.sha256"),
        )


@dataclass(frozen=True)
class DevelopmentTargetScopeReceiptV11:
    original_target_artifact_sha256: str
    original_verified_target_receipt_sha256: str
    original_target_policy_sha256: str
    eligibility_amendment_artifact_sha256: str
    eligibility_amendment_receipt_sha256: str
    model_split: str
    patient_ids: tuple[str, ...]
    patient_roster_sha256: str
    patient_count: int
    tensor_file_sha256: str
    tensor_receipts: tuple[DevelopmentTargetTensorReceiptV11, ...]
    standard_19: tuple[str, ...] = STANDARD_19
    purpose: str = DEVELOPMENT_TARGET_SCOPE_PURPOSE_V1_1
    serialization: str = "canonical_json_plus_safetensors_no_pickle"
    target_semantics: str = "deepsoz_patient_reference_dataset_complement_v2"
    selection_basis: str = "signed_signal_evidence_eligibility_amendment_v1_1"
    exporter_read_full_verified_target: bool = True
    consumer_reads_full_target_or_split: bool = False
    other_split_patient_ids_included: bool = False
    other_split_target_payload_included: bool = False
    source_eval_patient_ids_included: bool = False
    source_eval_target_payload_included: bool = False
    private_payload_included: bool = False
    development_only: bool = True
    formal_promotion: bool = False
    schema_version: str = DEVELOPMENT_TARGET_SCOPE_SCHEMA_V1_1

    def __post_init__(self) -> None:
        for name in (
            "original_target_artifact_sha256",
            "original_verified_target_receipt_sha256",
            "original_target_policy_sha256",
            "eligibility_amendment_artifact_sha256",
            "eligibility_amendment_receipt_sha256",
            "patient_roster_sha256",
            "tensor_file_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        if (
            self.original_target_artifact_sha256
            != FROZEN_ORIGINAL_TARGET_ARTIFACT_SHA256
            or self.original_verified_target_receipt_sha256
            != FROZEN_ORIGINAL_VERIFIED_TARGET_RECEIPT_SHA256
            or self.original_target_policy_sha256
            != FROZEN_ORIGINAL_TARGET_POLICY_SHA256
            or self.eligibility_amendment_artifact_sha256
            != FROZEN_AMENDMENT_ARTIFACT_SHA256
            or self.eligibility_amendment_receipt_sha256
            != FROZEN_AMENDMENT_RECEIPT_SHA256
        ):
            raise ValueError("Development target scope lineage trust anchor changed")
        if self.model_split not in _ALLOWED_SPLITS:
            raise ValueError("Development target scope split changed")
        roster = _normalized_roster(self.patient_ids, field_name="patient_ids")
        object.__setattr__(self, "patient_ids", roster)
        expected_count = EXPECTED_SCOPE_PATIENT_COUNTS[self.model_split]
        if self.patient_count != expected_count or len(roster) != expected_count:
            raise ValueError("Development target scope patient count changed")
        if self.patient_roster_sha256 != _roster_sha256(self.model_split, roster):
            raise ValueError("Development target scope roster SHA mismatch")
        receipts = tuple(self.tensor_receipts)
        if any(type(item) is not DevelopmentTargetTensorReceiptV11 for item in receipts):
            raise TypeError("Development target tensor receipt type changed")
        if tuple(item.name for item in receipts) != _TENSOR_NAMES:
            raise ValueError("Development target tensor receipts changed")
        expected_shape = (expected_count, len(STANDARD_19))
        if any(item.shape != expected_shape for item in receipts):
            raise ValueError("Development target tensor receipt shape changed")
        object.__setattr__(self, "tensor_receipts", receipts)
        fixed = {
            "standard_19": STANDARD_19,
            "purpose": DEVELOPMENT_TARGET_SCOPE_PURPOSE_V1_1,
            "serialization": "canonical_json_plus_safetensors_no_pickle",
            "target_semantics": "deepsoz_patient_reference_dataset_complement_v2",
            "selection_basis": "signed_signal_evidence_eligibility_amendment_v1_1",
            "exporter_read_full_verified_target": True,
            "consumer_reads_full_target_or_split": False,
            "other_split_patient_ids_included": False,
            "other_split_target_payload_included": False,
            "source_eval_patient_ids_included": False,
            "source_eval_target_payload_included": False,
            "private_payload_included": False,
            "development_only": True,
            "formal_promotion": False,
            "schema_version": DEVELOPMENT_TARGET_SCOPE_SCHEMA_V1_1,
        }
        if any(getattr(self, name) != value for name, value in fixed.items()):
            raise ValueError("Development target scope scientific boundary changed")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def _tensor_receipts(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[DevelopmentTargetTensorReceiptV11, ...]:
    return tuple(
        DevelopmentTargetTensorReceiptV11(
            name=name,
            dtype=str(tensors[name].dtype).removeprefix("torch."),
            shape=tuple(tensors[name].shape),
            sha256=_tensor_sha256(name, tensors[name]),
        )
        for name in _TENSOR_NAMES
    )


def _validate_target_tensors(
    tensors: Mapping[str, torch.Tensor],
    receipt: DevelopmentTargetScopeReceiptV11,
) -> tuple[torch.Tensor, torch.Tensor]:
    if set(tensors) != set(_TENSOR_NAMES):
        raise ValueError("Development target safetensors schema changed")
    values = tensors["values"].detach().cpu().contiguous()
    mask = tensors["mask"].detach().cpu().contiguous()
    expected_shape = (receipt.patient_count, len(STANDARD_19))
    if (
        values.dtype != torch.float32
        or mask.dtype != torch.bool
        or tuple(values.shape) != expected_shape
        or tuple(mask.shape) != expected_shape
    ):
        raise ValueError("Development target tensor shape or dtype changed")
    if not torch.isfinite(values).all():
        raise ValueError("Development target values contain non-finite entries")
    if torch.any(values[~mask] != 0):
        raise ValueError("Masked development target values must be zero")
    if torch.any((values[mask] != 0) & (values[mask] != 1)):
        raise ValueError("Observed development targets must be binary")
    if torch.any(mask[:, CHANNEL_INDEX["PZ"]]):
        raise ValueError("PZ must remain masked in the development target scope")
    if not torch.all(torch.any((values == 1) & mask, dim=1)):
        raise ValueError("Every scoped patient must retain an observed positive")
    copied = {"values": values.clone(), "mask": mask.clone()}
    if _tensor_receipts(copied) != receipt.tensor_receipts:
        raise ValueError("Development target tensor SHA or metadata changed")
    return copied["values"], copied["mask"]


@dataclass(frozen=True)
class LoadedDevelopmentTargetScopeV11:
    path: Path
    receipt: DevelopmentTargetScopeReceiptV11
    receipt_file_sha256: str
    _values: torch.Tensor = field(repr=False, compare=False)
    _mask: torch.Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_file_sha256",
            _require_sha256(
                self.receipt_file_sha256,
                field_name="receipt_file_sha256",
            ),
        )
        if self.receipt_file_sha256 != self.receipt.receipt_sha256:
            raise ValueError("Scoped target receipt file does not match its payload")
        values, mask = _validate_target_tensors(
            {"values": self._values, "mask": self._mask}, self.receipt
        )
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_mask", mask)

    @property
    def model_split(self) -> str:
        return self.receipt.model_split

    def assert_unchanged(self) -> None:
        _validate_target_tensors(
            {"values": self._values, "mask": self._mask}, self.receipt
        )

    def target_batch(
        self,
        model_split: str,
        patient_ids: Sequence[object] | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> SOZTargetBatch:
        self.assert_unchanged()
        if model_split != self.model_split:
            raise ValueError(
                f"Loaded target scope is {self.model_split}-only; {model_split} is inaccessible"
            )
        roster = self.receipt.patient_ids
        requested = (
            roster
            if patient_ids is None
            else tuple(normalize_patient_id(value) for value in patient_ids)
        )
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("A scoped target batch must be non-empty and unique")
        index = {patient_id: position for position, patient_id in enumerate(roster)}
        unknown = tuple(patient_id for patient_id in requested if patient_id not in index)
        if unknown:
            raise KeyError(
                f"Patients are absent from the signed {model_split} scope: {unknown}"
            )
        order = torch.tensor([index[patient_id] for patient_id in requested], dtype=torch.long)
        return SOZTargetBatch(
            patient_ids=requested,
            values=self._values.index_select(0, order).to(device=device),
            mask=self._mask.index_select(0, order).to(device=device),
        )


@dataclass(frozen=True)
class PublishedDevelopmentTargetScopesV11:
    path: Path
    source_train: LoadedDevelopmentTargetScopeV11
    source_dev: LoadedDevelopmentTargetScopeV11

    def __post_init__(self) -> None:
        if self.source_train.model_split != "source_train":
            raise ValueError("Published train target scope changed")
        if self.source_dev.model_split != "source_dev":
            raise ValueError("Published dev target scope changed")
        if self.source_train.path.parent != self.path or self.source_dev.path.parent != self.path:
            raise ValueError("Published split target paths changed")


def _receipt_from_payload(payload: object) -> DevelopmentTargetScopeReceiptV11:
    if not isinstance(payload, Mapping):
        raise ValueError("Development target receipt must be a JSON object")
    expected_fields = {item.name for item in fields(DevelopmentTargetScopeReceiptV11)}
    if set(payload) != expected_fields:
        raise ValueError(
            "Development target receipt violates its closed schema; "
            f"missing={sorted(expected_fields-set(payload))}, "
            f"unknown={sorted(set(payload)-expected_fields)}"
        )
    tensor_payload = payload.get("tensor_receipts")
    tensor_fields = {item.name for item in fields(DevelopmentTargetTensorReceiptV11)}
    if not isinstance(tensor_payload, list):
        raise ValueError("Development target tensor receipts must be a list")
    parsed_tensors = []
    for row in tensor_payload:
        if not isinstance(row, Mapping) or set(row) != tensor_fields:
            raise ValueError("Development target tensor receipt schema changed")
        parsed_tensors.append(DevelopmentTargetTensorReceiptV11(**dict(row)))
    values = dict(payload)
    values["tensor_receipts"] = tuple(parsed_tensors)
    values["patient_ids"] = tuple(values["patient_ids"])
    values["standard_19"] = tuple(values["standard_19"])
    try:
        return DevelopmentTargetScopeReceiptV11(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Development target receipt is invalid") from exc


def load_development_target_scope_v1_1(
    bundle_directory: str | Path,
    *,
    expected_model_split: str,
    expected_receipt_file_sha256: str,
) -> LoadedDevelopmentTargetScopeV11:
    """Load one independent split child; no full or other-split input exists."""

    try:
        from safetensors.torch import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for scoped target loading") from exc
    if expected_model_split not in _ALLOWED_SPLITS:
        raise ValueError("Scoped target loader requires source_train or source_dev")
    bundle = _reject_symlink_components(
        Path(bundle_directory), field_name="Development target scope bundle"
    )
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Development target scope must be a regular directory")
    entries = tuple(bundle.iterdir())
    names = {entry.name for entry in entries}
    if names != _CHILD_FILES or len(entries) != len(_CHILD_FILES):
        raise ValueError("Development target scope violates its closed file schema")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("Development target scope files must be regular and non-symlinked")
    receipt_raw = _read_stable_regular_file(
        bundle / DEVELOPMENT_TARGET_SCOPE_RECEIPT_FILENAME,
        field_name="Development target receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if receipt_sha != _require_sha256(
        expected_receipt_file_sha256,
        field_name="expected_receipt_file_sha256",
    ):
        raise ValueError("Development target receipt SHA mismatch")
    payload = _strict_json(receipt_raw, field_name="Development target receipt")
    if _canonical_json_bytes(payload) != receipt_raw:
        raise ValueError("Development target receipt is not canonical JSON")
    receipt = _receipt_from_payload(payload)
    if receipt.model_split != expected_model_split:
        raise ValueError("Development target receipt is for a different split")
    tensor_raw = _read_stable_regular_file(
        bundle / DEVELOPMENT_TARGET_SCOPE_TENSORS_FILENAME,
        field_name="Development target safetensors",
        maximum_bytes=_MAX_TENSOR_BYTES,
    )
    if hashlib.sha256(tensor_raw).hexdigest() != receipt.tensor_file_sha256:
        raise ValueError("Development target safetensors file SHA mismatch")
    try:
        tensors = load(tensor_raw)
    except Exception as exc:
        raise ValueError("Development target safetensors payload is invalid") from exc
    values, mask = _validate_target_tensors(tensors, receipt)
    return LoadedDevelopmentTargetScopeV11(
        path=bundle,
        receipt=receipt,
        receipt_file_sha256=receipt_sha,
        _values=values,
        _mask=mask,
    )


def _assert_full_verified_target_unchanged(
    verified_target: VerifiedDeepSOZTargetV2Artifact,
) -> None:
    if type(verified_target) is not VerifiedDeepSOZTargetV2Artifact:
        raise TypeError("Exporter requires the strict verified target-v2 object")
    verified_target.__post_init__()
    frame = _build_target_frame(
        verified_target.registry,
        source_sha256=verified_target.receipt.source_input_sha256,
        split_sha256=verified_target.receipt.split_input_sha256,
    )
    digest = hashlib.sha256(_target_csv_bytes(frame)).hexdigest()
    if digest != verified_target.receipt.target_artifact_sha256:
        raise ValueError("In-memory full target-v2 changed after strict verification")


def materialize_development_target_scopes_v1_1(
    verified_target: VerifiedDeepSOZTargetV2Artifact,
    amendment: FrozenSignalEligibilityViewV11,
    output_root: str | Path,
    *,
    expected_original_target_artifact_sha256: str,
    expected_original_verified_receipt_sha256: str,
) -> PublishedDevelopmentTargetScopesV11:
    """Atomically publish independent 65-train and 16-dev target bundles."""

    _assert_full_verified_target_unchanged(verified_target)
    if type(amendment) is not FrozenSignalEligibilityViewV11:
        raise TypeError("Exporter requires the frozen signal-eligibility view")
    amendment.__post_init__()
    expected_target = _require_sha256(
        expected_original_target_artifact_sha256,
        field_name="expected_original_target_artifact_sha256",
    )
    expected_receipt = _require_sha256(
        expected_original_verified_receipt_sha256,
        field_name="expected_original_verified_receipt_sha256",
    )
    if expected_target != FROZEN_ORIGINAL_TARGET_ARTIFACT_SHA256:
        raise ValueError("Expected full target artifact is not frozen")
    if expected_receipt != FROZEN_ORIGINAL_VERIFIED_TARGET_RECEIPT_SHA256:
        raise ValueError("Expected full target receipt is not frozen")
    source_receipt = verified_target.receipt
    if (
        source_receipt.target_artifact_sha256 != expected_target
        or source_receipt.receipt_sha256 != expected_receipt
        or source_receipt.policy_sha256 != FROZEN_ORIGINAL_TARGET_POLICY_SHA256
        or source_receipt.eligible_patient_count
        != EXPECTED_FULL_ELIGIBLE_PATIENT_COUNT
    ):
        raise ValueError("Full verified target identity changed")
    full_rosters = dict(source_receipt.eligible_split_patient_ids)
    if set(full_rosters) != {"source_train", "source_dev", "source_eval"}:
        raise ValueError("Full verified target split schema changed")
    if (
        len(full_rosters["source_train"]) != 69
        or len(full_rosters["source_dev"]) != 16
        or len(full_rosters["source_eval"])
        != EXPECTED_OMITTED_SOURCE_EVAL_PATIENT_COUNT
    ):
        raise ValueError("Full verified target split counts changed")
    if amendment.target_header_source_train_patient_ids != full_rosters["source_train"]:
        raise ValueError("Amendment target-header train roster differs from target-v2")
    if amendment.target_header_source_dev_patient_ids != full_rosters["source_dev"]:
        raise ValueError("Amendment target-header dev roster differs from target-v2")
    selected_rosters = {
        "source_train": amendment.signal_evidence_source_train_patient_ids,
        "source_dev": amendment.signal_evidence_source_dev_patient_ids,
    }

    tensors_by_split = {}
    for model_split, patient_ids in selected_rosters.items():
        references = tuple(verified_target.registry.get(value) for value in patient_ids)
        if any(
            reference.model_split != model_split
            or not reference.eligible_for_localization
            for reference in references
        ):
            raise ValueError("Amendment selected a target outside its verified split")
        tensors_by_split[model_split] = {
            "values": torch.stack([row.values for row in references])
            .detach()
            .cpu()
            .float()
            .contiguous(),
            "mask": torch.stack([row.mask for row in references])
            .detach()
            .cpu()
            .bool()
            .contiguous(),
        }
    _assert_full_verified_target_unchanged(verified_target)

    target = _safe_new_directory(output_root, field_name="Development target root")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for scoped target export") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    receipt_shas = {}
    try:
        for model_split in _ALLOWED_SPLITS:
            child = temporary / _SPLIT_DIRECTORY[model_split]
            child.mkdir()
            tensor_path = child / DEVELOPMENT_TARGET_SCOPE_TENSORS_FILENAME
            tensors = tensors_by_split[model_split]
            save_file(tensors, str(tensor_path))
            if not 1 <= tensor_path.stat().st_size <= _MAX_TENSOR_BYTES:
                raise ValueError("Development target safetensors has an invalid size")
            patient_ids = selected_rosters[model_split]
            receipt = DevelopmentTargetScopeReceiptV11(
                original_target_artifact_sha256=source_receipt.target_artifact_sha256,
                original_verified_target_receipt_sha256=source_receipt.receipt_sha256,
                original_target_policy_sha256=source_receipt.policy_sha256,
                eligibility_amendment_artifact_sha256=amendment.artifact_sha256,
                eligibility_amendment_receipt_sha256=amendment.receipt_sha256,
                model_split=model_split,
                patient_ids=patient_ids,
                patient_roster_sha256=_roster_sha256(model_split, patient_ids),
                patient_count=len(patient_ids),
                tensor_file_sha256=_file_sha256(tensor_path),
                tensor_receipts=_tensor_receipts(tensors),
            )
            receipt_raw = _canonical_json_bytes(asdict(receipt))
            if not 1 <= len(receipt_raw) <= _MAX_RECEIPT_BYTES:
                raise ValueError("Development target receipt has an invalid size")
            receipt_path = child / DEVELOPMENT_TARGET_SCOPE_RECEIPT_FILENAME
            receipt_path.write_bytes(receipt_raw)
            receipt_shas[model_split] = receipt.receipt_sha256
            _fsync_file(tensor_path)
            _fsync_file(receipt_path)
            _fsync_directory(child)
        _fsync_directory(temporary)
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        train = load_development_target_scope_v1_1(
            target / DEVELOPMENT_TARGET_SCOPE_TRAIN_DIRECTORY,
            expected_model_split="source_train",
            expected_receipt_file_sha256=receipt_shas["source_train"],
        )
        dev = load_development_target_scope_v1_1(
            target / DEVELOPMENT_TARGET_SCOPE_DEV_DIRECTORY,
            expected_model_split="source_dev",
            expected_receipt_file_sha256=receipt_shas["source_dev"],
        )
        return PublishedDevelopmentTargetScopesV11(
            path=target,
            source_train=train,
            source_dev=dev,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "DEVELOPMENT_TARGET_SCOPE_DEV_DIRECTORY",
    "DEVELOPMENT_TARGET_SCOPE_PURPOSE_V1_1",
    "DEVELOPMENT_TARGET_SCOPE_RECEIPT_FILENAME",
    "DEVELOPMENT_TARGET_SCOPE_SCHEMA_V1_1",
    "DEVELOPMENT_TARGET_SCOPE_TENSORS_FILENAME",
    "DEVELOPMENT_TARGET_SCOPE_TRAIN_DIRECTORY",
    "DevelopmentTargetScopeReceiptV11",
    "DevelopmentTargetTensorReceiptV11",
    "EXPECTED_FULL_ELIGIBLE_PATIENT_COUNT",
    "EXPECTED_OMITTED_SOURCE_EVAL_PATIENT_COUNT",
    "EXPECTED_SCOPE_PATIENT_COUNTS",
    "EXPECTED_TARGET_HEADER_COUNTS",
    "FROZEN_ORIGINAL_TARGET_ARTIFACT_SHA256",
    "FROZEN_ORIGINAL_TARGET_POLICY_SHA256",
    "FROZEN_ORIGINAL_VERIFIED_TARGET_RECEIPT_SHA256",
    "FrozenSignalEligibilityViewV11",
    "LoadedDevelopmentTargetScopeV11",
    "PublishedDevelopmentTargetScopesV11",
    "load_development_target_scope_v1_1",
    "load_frozen_signal_eligibility_for_target_export_v1_1",
    "materialize_development_target_scopes_v1_1",
]
