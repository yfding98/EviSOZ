"""Physically split fit and diagnostic stages for the LaBraM I+V candidate.

This module deliberately does not reuse the v1 combined train/dev trainer.
The fit stage accepts a source-train-only target scope, performs the frozen
twenty-epoch optimization schedule, and publishes only the final checkpoint.
The diagnostic stage is a separate call accepting a source-dev-only target
scope and a strictly loaded frozen checkpoint.  It performs one full-bag
forward with no optimizer and verifies that numerical explanations reconstruct
the patient logits and that the checkpoint state did not change.

The v1.1 evidence capability still contains target-free train and development
evidence in one upstream artifact.  Receipts state this accurately; only the
target values are physically split.  In particular, the fit stage never claims
that source-dev evidence bytes are absent, only that no source-dev target is
reachable and no source-dev forward is executed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from . import development_reasoner as _v1
from .development_reasoner_training import development_reasoner_state_sha256
from .development_reasoner_v1_1 import (
    FROZEN_AMENDMENT_ARTIFACT_SHA256,
    FROZEN_AMENDMENT_RECEIPT_SHA256,
    FROZEN_TARGET_V2_ARTIFACT_SHA256,
    FROZEN_TARGET_V2_POLICY_SHA256,
    FROZEN_TARGET_V2_RECEIPT_SHA256,
    FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256,
    FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
    PublishedDevelopmentIVEvidenceCapabilityV11,
)
from .development_target_scope_v1_1 import LoadedDevelopmentTargetScopeV11
from .formal_reasoner_pipeline import (
    FORMAL_REASONER_FIT_POLICY_SHA256,
    FormalReasonerFitConfig,
)
from .geometry import STANDARD_19


DEVELOPMENT_REASONER_SPLIT_DATASET_SCHEMA_V1_1 = (
    "soz_development_iv_split_dataset_v1_1"
)
DEVELOPMENT_REASONER_FIT_SCHEMA_V1_1 = "soz_development_iv_fit_v1_1"
DEVELOPMENT_REASONER_FIT_ARTIFACT_SCHEMA_V1_1 = (
    "soz_development_iv_frozen_checkpoint_v1_1"
)
DEVELOPMENT_REASONER_DIAGNOSTIC_SCHEMA_V1_1 = (
    "soz_development_iv_dev_diagnostic_v1_1"
)
DEVELOPMENT_REASONER_DIAGNOSTIC_ARTIFACT_SCHEMA_V1_1 = (
    "soz_development_iv_dev_diagnostic_artifact_v1_1"
)

FIT_MANIFEST_FILENAME = "manifest.json"
FIT_CHECKPOINT_FILENAME = "checkpoint.safetensors"
DIAGNOSTIC_MANIFEST_FILENAME = "diagnostic.json"
DIAGNOSTIC_TENSORS_FILENAME = "diagnostic.safetensors"

FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256 = (
    "90529bb91df657a27f52d82300ce13431c94d4a4b76f28691bea59eeddcde361"
)
FROZEN_SOURCE_TRAIN_TARGET_TENSOR_FILE_SHA256 = (
    "1a548c6d8e70863cf40488b5ccfbd2bdb261826a71bc0afe8960dd05286571f2"
)
FROZEN_SOURCE_DEV_TARGET_SCOPE_RECEIPT_SHA256 = (
    "666b0d3218df41ccf711a93e83bbdd5c941eb910e0fafeb3c20430c12cd6e899"
)
FROZEN_SOURCE_DEV_TARGET_TENSOR_FILE_SHA256 = (
    "c7cca784aca196e92aec3728ef0897f522a13bcd09aa01f2bc8ca00be9577e95"
)

_FIT_FILES = frozenset({FIT_MANIFEST_FILENAME, FIT_CHECKPOINT_FILENAME})
_DIAGNOSTIC_FILES = frozenset(
    {DIAGNOSTIC_MANIFEST_FILENAME, DIAGNOSTIC_TENSORS_FILENAME}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_TENSOR_BYTES = 128 * 1024 * 1024
_DATASET_MARKER = object()
_FIT_RUN_MARKER = object()
_DIAGNOSTIC_RUN_MARKER = object()
_PUBLISHED_FIT_MARKER = object()
_EXPECTED_PARAMETER_COUNT = 159
_CHANNEL_ORDER_SHA256 = _v1._canonical_sha256(
    {"standard_19_channel_order": STANDARD_19}
)


_EXPECTED_COMPONENT_NAMES = tuple(
    sorted(
        ("channel_prior",)
        + tuple(
            f"{family}/{phase}"
            for family in (
                "evolution/raw",
                "quality_attenuation/evolution",
                "ictal_involvement/raw",
                "quality_attenuation/ictal_involvement",
            )
            for phase in _v1.PHASE_COMPONENT_NAMES
        )
    )
)
_EXPECTED_COMPONENT_MAP = tuple(
    (name, f"component_{index:02d}")
    for index, name in enumerate(_EXPECTED_COMPONENT_NAMES)
)


_FIT_POLICY = {
    "schema_version": DEVELOPMENT_REASONER_FIT_SCHEMA_V1_1,
    "optimizer_policy_sha256": FORMAL_REASONER_FIT_POLICY_SHA256,
    "fit_split": "source_train",
    "fit_patient_count": 65,
    "fit_event_count": 582,
    "checkpoint_selection": "final_epoch_20_only",
    "source_train_postfit_full_bag_forward_count": 1,
    "source_dev_forward_count": 0,
    "source_dev_target_values_reachable": False,
    "source_dev_evidence_loaded_with_target_free_capability": True,
    "source_dev_evidence_used_for_fit_or_statistics": False,
    "threshold_selection": "forbidden",
    "calibration": "forbidden",
    "source_eval_allowed": False,
    "private_allowed": False,
    "formal_promotion": False,
}
DEVELOPMENT_REASONER_FIT_POLICY_SHA256_V1_1 = _v1._canonical_sha256(_FIT_POLICY)


_DIAGNOSTIC_POLICY = {
    "schema_version": DEVELOPMENT_REASONER_DIAGNOSTIC_SCHEMA_V1_1,
    "diagnostic_split": "source_dev",
    "diagnostic_patient_count": 16,
    "diagnostic_event_count": 221,
    "checkpoint_source": "immutable_final_epoch_20_fit_artifact",
    "optimizer_instantiated": False,
    "all_parameters_require_grad_false": True,
    "full_bag_forward_count": 1,
    "checkpoint_state_must_remain_identical": True,
    "numeric_explanation_must_reconstruct_logits": True,
    "threshold_selection": "forbidden",
    "calibration": "forbidden",
    "source_eval_allowed": False,
    "private_allowed": False,
    "formal_promotion": False,
}
DEVELOPMENT_REASONER_DIAGNOSTIC_POLICY_SHA256_V1_1 = _v1._canonical_sha256(
    _DIAGNOSTIC_POLICY
)


def _sha(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")
    return text


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


def _read_stable_file(path: Path, *, field_name: str, maximum_bytes: int) -> bytes:
    source = _reject_symlink_components(path, field_name=field_name)
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


def _closed_directory(path: str | Path, expected: frozenset[str], *, name: str) -> Path:
    bundle = _reject_symlink_components(Path(path), field_name=name)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError(f"{name} must be a regular directory")
    entries = tuple(bundle.iterdir())
    if len(entries) != len(expected) or {row.name for row in entries} != expected:
        raise ValueError(f"{name} violates its closed file schema")
    if any(row.is_symlink() or not row.is_file() for row in entries):
        raise ValueError(f"{name} files must be regular and non-symlinked")
    return bundle


def _patient_order(patient_ids: tuple[str, ...], *, seed: int, epoch: int) -> tuple[str, ...]:
    order = list(patient_ids)
    random.Random((seed << 20) ^ epoch).shuffle(order)
    return tuple(order)


def _tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {
        name: {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": _v1._tensor_sha256(name, value),
        }
        for name, value in sorted(tensors.items())
    }


def _patient_roster_sha256(
    model_split: str, patient_ids: Sequence[str]
) -> str:
    return _v1._canonical_sha256(
        {"model_split": model_split, "patient_ids": tuple(patient_ids)}
    )


def _diagnostic_tensor_set_sha256(
    tensors: Mapping[str, torch.Tensor],
    component_map: Sequence[tuple[str, str]],
) -> str:
    return _v1._canonical_sha256(
        {
            "component_map": [list(row) for row in component_map],
            "tensors": {
                name: _v1._tensor_sha256(name, value)
                for name, value in sorted(tensors.items())
            },
        }
    )


def _numeric_explanation_receipt_sha256(
    tensors: Mapping[str, torch.Tensor],
    component_map: Sequence[tuple[str, str]],
) -> str:
    return _v1._canonical_sha256(
        {
            "explanation_mode": _v1.DEVELOPMENT_IV_EXPLANATION_MODE,
            "llm_used_for_prediction": False,
            "patient_logits_sha256": _v1._tensor_sha256(
                "patient_logits", tensors["patient_logits"]
            ),
            "components": {
                name: _v1._tensor_sha256(key, tensors[key])
                for name, key in component_map
            },
        }
    )


@dataclass(frozen=True)
class DevelopmentReasonerSplitDatasetReceiptV11:
    model_split: str
    capability_manifest_sha256: str
    evidence_authorization_sha256: str
    evidence_split_receipt_sha256: str
    target_scope_receipt_sha256: str
    target_tensor_file_sha256: str
    original_target_artifact_sha256: str
    original_target_receipt_sha256: str
    original_target_policy_sha256: str
    eligibility_amendment_artifact_sha256: str
    eligibility_amendment_receipt_sha256: str
    patient_roster_sha256: str
    patient_count: int
    event_count: int
    base_dataset_sha256: str
    other_split_evidence_loaded_with_target_free_capability: bool = True
    other_split_evidence_used_for_fit_or_statistics: bool = False
    other_split_target_values_reachable: bool = False
    source_eval_used: bool = False
    private_used: bool = False
    formal_promotion: bool = False
    schema_version: str = DEVELOPMENT_REASONER_SPLIT_DATASET_SCHEMA_V1_1

    def __post_init__(self) -> None:
        for name in (
            "capability_manifest_sha256",
            "evidence_authorization_sha256",
            "evidence_split_receipt_sha256",
            "target_scope_receipt_sha256",
            "target_tensor_file_sha256",
            "original_target_artifact_sha256",
            "original_target_receipt_sha256",
            "original_target_policy_sha256",
            "eligibility_amendment_artifact_sha256",
            "eligibility_amendment_receipt_sha256",
            "patient_roster_sha256",
            "base_dataset_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), field_name=name))
        if self.model_split not in {"source_train", "source_dev"}:
            raise ValueError("Split dataset rejects source_eval/private")
        expected = {
            "source_train": (65, 582),
            "source_dev": (16, 221),
        }[self.model_split]
        if (self.patient_count, self.event_count) != expected:
            raise ValueError("Split dataset patient/event count changed")
        frozen = {
            "capability_manifest_sha256": FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
            "evidence_authorization_sha256": FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256,
            "original_target_artifact_sha256": FROZEN_TARGET_V2_ARTIFACT_SHA256,
            "original_target_receipt_sha256": FROZEN_TARGET_V2_RECEIPT_SHA256,
            "original_target_policy_sha256": FROZEN_TARGET_V2_POLICY_SHA256,
            "eligibility_amendment_artifact_sha256": FROZEN_AMENDMENT_ARTIFACT_SHA256,
            "eligibility_amendment_receipt_sha256": FROZEN_AMENDMENT_RECEIPT_SHA256,
        }
        if any(getattr(self, name) != value for name, value in frozen.items()):
            raise ValueError("Split dataset frozen lineage changed")
        split_target = {
            "source_train": (
                FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
                FROZEN_SOURCE_TRAIN_TARGET_TENSOR_FILE_SHA256,
            ),
            "source_dev": (
                FROZEN_SOURCE_DEV_TARGET_SCOPE_RECEIPT_SHA256,
                FROZEN_SOURCE_DEV_TARGET_TENSOR_FILE_SHA256,
            ),
        }[self.model_split]
        if (
            self.target_scope_receipt_sha256,
            self.target_tensor_file_sha256,
        ) != split_target:
            raise ValueError("Split dataset target-scope trust anchor changed")
        if (
            not self.other_split_evidence_loaded_with_target_free_capability
            or self.other_split_evidence_used_for_fit_or_statistics
            or self.other_split_target_values_reachable
            or self.source_eval_used
            or self.private_used
            or self.formal_promotion
        ):
            raise ValueError("Split dataset isolation boundary changed")
        if self.schema_version != DEVELOPMENT_REASONER_SPLIT_DATASET_SCHEMA_V1_1:
            raise ValueError("Unsupported split dataset schema")

    @property
    def receipt_sha256(self) -> str:
        return _v1._canonical_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class VerifiedDevelopmentReasonerSplitDatasetV11:
    dataset: _v1.DevelopmentReasonerDataset = field(repr=False)
    target_scope: LoadedDevelopmentTargetScopeV11 = field(repr=False)
    receipt: DevelopmentReasonerSplitDatasetReceiptV11
    _receipt_sha256: str = field(repr=False)

    def __init__(
        self,
        *,
        _verification_marker: object,
        dataset: _v1.DevelopmentReasonerDataset,
        target_scope: LoadedDevelopmentTargetScopeV11,
        receipt: DevelopmentReasonerSplitDatasetReceiptV11,
    ) -> None:
        if _verification_marker is not _DATASET_MARKER:
            raise TypeError("v1.1 split dataset requires the closed target join")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "target_scope", target_scope)
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "_receipt_sha256", receipt.receipt_sha256)
        self.assert_unchanged()

    @property
    def model_split(self) -> str:
        return self.receipt.model_split

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return self.dataset.patient_ids

    def assert_unchanged(self) -> None:
        if type(self.dataset) is not _v1.DevelopmentReasonerDataset:
            raise TypeError("v1.1 split dataset base type changed")
        if type(self.target_scope) is not LoadedDevelopmentTargetScopeV11:
            raise TypeError("v1.1 split target scope type changed")
        self.dataset.assert_unchanged()
        self.target_scope.assert_unchanged()
        if self.receipt.receipt_sha256 != self._receipt_sha256:
            raise ValueError("v1.1 split dataset receipt changed in memory")
        if self.dataset.model_split != self.model_split or self.target_scope.model_split != self.model_split:
            raise ValueError("v1.1 split dataset identity changed")
        if self.dataset.receipt_sha256 != self.receipt.base_dataset_sha256:
            raise ValueError("v1.1 split dataset tensors changed")
        if self.target_scope.receipt.receipt_sha256 != self.receipt.target_scope_receipt_sha256:
            raise ValueError("v1.1 split target receipt changed")
        if self.target_scope.receipt_file_sha256 != self.receipt.target_scope_receipt_sha256:
            raise ValueError("v1.1 split target receipt file binding changed")
        if (
            self.target_scope.receipt.tensor_file_sha256
            != self.receipt.target_tensor_file_sha256
        ):
            raise ValueError("v1.1 split target tensor file binding changed")
        target_batch = self.target_scope.target_batch(self.model_split, self.patient_ids)
        full = self.dataset.full_batch()
        if _v1._tensor_sha256("targets", target_batch.values) != _v1._tensor_sha256(
            "targets", full.targets
        ) or _v1._tensor_sha256("target_mask", target_batch.mask) != _v1._tensor_sha256(
            "target_mask", full.target_mask
        ):
            raise ValueError("v1.1 joined target tensors changed")


def join_development_iv_split_targets_v1_1(
    capability: PublishedDevelopmentIVEvidenceCapabilityV11,
    target_scope: LoadedDevelopmentTargetScopeV11,
) -> VerifiedDevelopmentReasonerSplitDatasetV11:
    """Join exactly one split-scoped target payload to the signed I+V evidence."""

    if type(capability) is not PublishedDevelopmentIVEvidenceCapabilityV11:
        raise TypeError("v1.1 split join requires the strict published capability")
    if type(target_scope) is not LoadedDevelopmentTargetScopeV11:
        raise TypeError("v1.1 split join requires one strict scoped target")
    capability.capability.assert_unchanged()
    target_scope.assert_unchanged()
    authorization = capability.capability.receipt
    target_receipt = target_scope.receipt
    bindings = {
        "capability manifest": capability.manifest_sha256
        == FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
        "authorization receipt": capability.authorization_receipt_sha256
        == authorization.receipt_sha256,
        "target artifact": target_receipt.original_target_artifact_sha256
        == authorization.verified_target_v2_artifact_sha256,
        "target receipt": target_receipt.original_verified_target_receipt_sha256
        == authorization.verified_target_v2_receipt_sha256,
        "target policy": target_receipt.original_target_policy_sha256
        == authorization.verified_target_v2_policy_sha256,
        "amendment artifact": target_receipt.eligibility_amendment_artifact_sha256
        == authorization.amendment_artifact_sha256,
        "amendment receipt": target_receipt.eligibility_amendment_receipt_sha256
        == authorization.amendment_receipt_sha256,
    }
    failed = tuple(name for name, passed in bindings.items() if not passed)
    if failed:
        raise ValueError(f"v1.1 evidence/target lineage disagrees: {failed}")
    model_split = target_scope.model_split
    evidence = (
        capability.capability.base.capability.source_train
        if model_split == "source_train"
        else capability.capability.base.capability.source_dev
    )
    if evidence.patient_ids != target_receipt.patient_ids:
        raise ValueError("Scoped target roster differs from signed evidence roster")
    target_batch = target_scope.target_batch(model_split, evidence.patient_ids)
    patient_index = {value: index for index, value in enumerate(evidence.patient_ids)}
    event_patient_index = torch.tensor(
        [patient_index[value] for value in evidence.patient_ids_by_event], dtype=torch.long
    )
    counts = torch.bincount(event_patient_index, minlength=len(evidence.patient_ids))
    full_batch = _v1.DevelopmentReasonerPatientBatch(
        _verification_marker=_v1._PATIENT_BATCH_MARKER,
        evidence=evidence.evidence,
        event_patient_index=event_patient_index,
        patient_ids=evidence.patient_ids,
        event_ids=evidence.event_ids,
        expected_event_counts=counts,
        targets=target_batch.values.to(torch.float32),
        target_mask=target_batch.mask,
    )
    base_dataset = _v1.DevelopmentReasonerDataset(
        _verification_marker=_v1._DATASET_MARKER,
        model_split=model_split,
        full_batch=full_batch,
        evidence_authorization_sha256=authorization.receipt_sha256,
        verified_target_v2_receipt_sha256=authorization.verified_target_v2_receipt_sha256,
    )
    split_receipt = (
        authorization.source_train_evidence_receipt_sha256
        if model_split == "source_train"
        else authorization.source_dev_evidence_receipt_sha256
    )
    receipt = DevelopmentReasonerSplitDatasetReceiptV11(
        model_split=model_split,
        capability_manifest_sha256=capability.manifest_sha256,
        evidence_authorization_sha256=authorization.receipt_sha256,
        evidence_split_receipt_sha256=split_receipt,
        target_scope_receipt_sha256=target_receipt.receipt_sha256,
        target_tensor_file_sha256=target_receipt.tensor_file_sha256,
        original_target_artifact_sha256=target_receipt.original_target_artifact_sha256,
        original_target_receipt_sha256=target_receipt.original_verified_target_receipt_sha256,
        original_target_policy_sha256=target_receipt.original_target_policy_sha256,
        eligibility_amendment_artifact_sha256=target_receipt.eligibility_amendment_artifact_sha256,
        eligibility_amendment_receipt_sha256=target_receipt.eligibility_amendment_receipt_sha256,
        patient_roster_sha256=target_receipt.patient_roster_sha256,
        patient_count=len(evidence.patient_ids),
        event_count=evidence.evidence.batch_size,
        base_dataset_sha256=base_dataset.receipt_sha256,
    )
    return VerifiedDevelopmentReasonerSplitDatasetV11(
        _verification_marker=_DATASET_MARKER,
        dataset=base_dataset,
        target_scope=target_scope,
        receipt=receipt,
    )


@dataclass(frozen=True)
class DevelopmentReasonerEpochReceiptV11:
    epoch_index: int
    patient_order_sha256: str
    mean_total_loss: float
    mean_bce_loss: float
    mean_ranking_loss: float
    patient_count: int
    event_count: int

    def __post_init__(self) -> None:
        if isinstance(self.epoch_index, bool) or not isinstance(self.epoch_index, int) or self.epoch_index < 0:
            raise ValueError("epoch_index must be a non-negative integer")
        object.__setattr__(
            self,
            "patient_order_sha256",
            _sha(self.patient_order_sha256, field_name="patient_order_sha256"),
        )
        for name in ("mean_total_loss", "mean_bce_loss", "mean_ranking_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.patient_count != 65 or self.event_count != 582:
            raise ValueError("Every fit epoch must visit the exact 65/582 roster")


@dataclass(frozen=True)
class DevelopmentReasonerDiagnosticSummaryV11:
    model_split: str
    total_loss: float
    bce_loss: float
    ranking_loss: float
    patient_count: int
    event_count: int
    patient_abstain_recommended_count: int
    patient_logits_sha256: str
    target_mask_sha256: str
    numeric_explanation_receipt_sha256: str
    diagnostic_tensor_set_sha256: str

    def __post_init__(self) -> None:
        if self.model_split not in {"source_train", "source_dev"}:
            raise ValueError("Diagnostic summary rejects source_eval/private")
        expected = {"source_train": (65, 582), "source_dev": (16, 221)}[
            self.model_split
        ]
        if (self.patient_count, self.event_count) != expected:
            raise ValueError("Diagnostic summary patient/event count changed")
        for name in ("total_loss", "bce_loss", "ranking_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.patient_abstain_recommended_count <= self.patient_count:
            raise ValueError("Diagnostic abstention count is invalid")
        for name in (
            "patient_logits_sha256",
            "target_mask_sha256",
            "numeric_explanation_receipt_sha256",
            "diagnostic_tensor_set_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), field_name=name))


def _diagnostic_once(
    model: _v1.DevelopmentIVAdditiveReasoner,
    data: VerifiedDevelopmentReasonerSplitDatasetV11,
    *,
    device: torch.device,
) -> tuple[
    DevelopmentReasonerDiagnosticSummaryV11,
    dict[str, torch.Tensor],
    tuple[tuple[str, str], ...],
]:
    data.assert_unchanged()
    batch = data.dataset.full_batch().to(device)
    model.eval()
    with torch.no_grad():
        step = _v1.development_reasoner_step(model, batch)
        explanation = _v1.aggregate_numeric_explanations(
            step.reasoner, batch.event_patient_index
        )
    if not torch.allclose(
        explanation.patient_logits, step.patient_logits, atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Numerical explanation changed patient logits")
    tensors: dict[str, torch.Tensor] = {
        "patient_logits": step.patient_logits.detach().cpu().contiguous(),
        "patient_probabilities": step.patient_probabilities.detach().cpu().contiguous(),
        "patient_abstain_recommended": step.patient_abstain_recommended.detach()
        .cpu()
        .contiguous(),
        "event_counts": step.event_counts.detach().cpu().contiguous(),
    }
    component_map = []
    for index, (name, value) in enumerate(
        sorted(explanation.component_contributions.items())
    ):
        key = f"component_{index:02d}"
        tensors[key] = value.detach().cpu().contiguous()
        component_map.append((name, key))
    if tuple(component_map) != _EXPECTED_COMPONENT_MAP:
        raise RuntimeError("Numerical explanation component schema changed")
    reconstructed = sum(tensors[key] for _, key in component_map)
    if not torch.allclose(
        reconstructed, tensors["patient_logits"], atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Stored numerical components do not reconstruct logits")
    if explanation.explanation_mode != _v1.DEVELOPMENT_IV_EXPLANATION_MODE or (
        explanation.llm_used_for_prediction
    ):
        raise RuntimeError("Numerical explanation boundary changed")
    explanation_receipt = _numeric_explanation_receipt_sha256(
        tensors, component_map
    )
    summary = DevelopmentReasonerDiagnosticSummaryV11(
        model_split=data.model_split,
        total_loss=float(step.loss.total.detach().cpu()),
        bce_loss=float(step.loss.bce.detach().cpu()),
        ranking_loss=float(step.loss.ranking.detach().cpu()),
        patient_count=len(batch.patient_ids),
        event_count=int(step.event_counts.sum().item()),
        patient_abstain_recommended_count=int(
            step.patient_abstain_recommended.sum().item()
        ),
        patient_logits_sha256=_v1._tensor_sha256(
            "patient_logits", tensors["patient_logits"]
        ),
        target_mask_sha256=_v1._tensor_sha256(
            "target_mask", batch.target_mask.detach().cpu()
        ),
        numeric_explanation_receipt_sha256=explanation_receipt,
        diagnostic_tensor_set_sha256=_diagnostic_tensor_set_sha256(
            tensors, component_map
        ),
    )
    return summary, tensors, tuple(component_map)


@dataclass(frozen=True)
class DevelopmentReasonerFitReceiptV11:
    split_dataset_receipt_sha256: str
    capability_manifest_sha256: str
    evidence_authorization_sha256: str
    target_scope_receipt_sha256: str
    target_tensor_file_sha256: str
    config: FormalReasonerFitConfig
    config_sha256: str
    fit_policy_sha256: str
    initial_state_sha256: str
    final_state_sha256: str
    parameter_count: int
    epochs: tuple[DevelopmentReasonerEpochReceiptV11, ...]
    source_train_postfit_diagnostic: DevelopmentReasonerDiagnosticSummaryV11
    source_train_postfit_full_bag_forward_count: int = 1
    source_dev_forward_count: int = 0
    source_dev_target_values_reachable: bool = False
    source_dev_evidence_loaded_with_target_free_capability: bool = True
    source_dev_evidence_used_for_fit_or_statistics: bool = False
    checkpoint_selection: str = "final_epoch_20_only"
    checkpoint_parameters_frozen: bool = True
    threshold_selected: bool = False
    calibrator_fitted: bool = False
    source_eval_used: bool = False
    private_used: bool = False
    formal_reasoner_authorized: bool = False
    formal_promotion: bool = False
    schema_version: str = DEVELOPMENT_REASONER_FIT_SCHEMA_V1_1

    def __post_init__(self) -> None:
        for name in (
            "split_dataset_receipt_sha256",
            "capability_manifest_sha256",
            "evidence_authorization_sha256",
            "target_scope_receipt_sha256",
            "target_tensor_file_sha256",
            "config_sha256",
            "fit_policy_sha256",
            "initial_state_sha256",
            "final_state_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), field_name=name))
        if type(self.config) is not FormalReasonerFitConfig or (
            self.config.receipt_sha256 != FORMAL_REASONER_FIT_POLICY_SHA256
            or self.config_sha256 != FORMAL_REASONER_FIT_POLICY_SHA256
        ):
            raise ValueError("v1.1 fit optimizer schedule changed")
        if self.fit_policy_sha256 != DEVELOPMENT_REASONER_FIT_POLICY_SHA256_V1_1:
            raise ValueError("v1.1 fit policy changed")
        frozen_lineage = {
            "capability_manifest_sha256": FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
            "evidence_authorization_sha256": FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256,
            "target_scope_receipt_sha256": FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
            "target_tensor_file_sha256": FROZEN_SOURCE_TRAIN_TARGET_TENSOR_FILE_SHA256,
        }
        if any(
            getattr(self, name) != expected
            for name, expected in frozen_lineage.items()
        ):
            raise ValueError("v1.1 fit frozen lineage changed")
        if len(self.epochs) != 20 or tuple(row.epoch_index for row in self.epochs) != tuple(range(20)):
            raise ValueError("v1.1 fit requires exactly twenty complete epochs")
        if self.parameter_count != _EXPECTED_PARAMETER_COUNT:
            raise ValueError("v1.1 reasoner parameter count changed")
        if self.source_train_postfit_diagnostic.model_split != "source_train":
            raise ValueError("v1.1 fit diagnostic changed split")
        fixed = {
            "source_train_postfit_full_bag_forward_count": 1,
            "source_dev_forward_count": 0,
            "source_dev_target_values_reachable": False,
            "source_dev_evidence_loaded_with_target_free_capability": True,
            "source_dev_evidence_used_for_fit_or_statistics": False,
            "checkpoint_selection": "final_epoch_20_only",
            "checkpoint_parameters_frozen": True,
            "threshold_selected": False,
            "calibrator_fitted": False,
            "source_eval_used": False,
            "private_used": False,
            "formal_reasoner_authorized": False,
            "formal_promotion": False,
            "schema_version": DEVELOPMENT_REASONER_FIT_SCHEMA_V1_1,
        }
        if any(getattr(self, name) != value for name, value in fixed.items()):
            raise ValueError("v1.1 fit scientific boundary changed")

    @property
    def receipt_sha256(self) -> str:
        return _v1._canonical_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class VerifiedDevelopmentReasonerFitRunV11:
    model: _v1.DevelopmentIVAdditiveReasoner = field(repr=False)
    receipt: DevelopmentReasonerFitReceiptV11
    _receipt_sha256: str = field(repr=False)

    def __init__(
        self,
        *,
        _verification_marker: object,
        model: _v1.DevelopmentIVAdditiveReasoner,
        receipt: DevelopmentReasonerFitReceiptV11,
    ) -> None:
        if _verification_marker is not _FIT_RUN_MARKER:
            raise TypeError("v1.1 fit run requires the closed fitter")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "_receipt_sha256", receipt.receipt_sha256)
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Published v1.1 checkpoint must remain frozen in eval mode")
        if self.receipt.receipt_sha256 != self._receipt_sha256:
            raise ValueError("v1.1 fit receipt changed in memory")
        if development_reasoner_state_sha256(self.model) != self.receipt.final_state_sha256:
            raise ValueError("v1.1 checkpoint state changed")


def fit_development_iv_reasoner_v1_1(
    data: VerifiedDevelopmentReasonerSplitDatasetV11,
    *,
    device: str | torch.device = "cpu",
) -> VerifiedDevelopmentReasonerFitRunV11:
    """Fit the exact train-only candidate and freeze the final epoch state."""

    if type(data) is not VerifiedDevelopmentReasonerSplitDatasetV11:
        raise TypeError("v1.1 fit requires the strict split dataset")
    if data.model_split != "source_train":
        raise ValueError("v1.1 fit accepts source_train only")
    data.assert_unchanged()
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("v1.1 fit device must be cpu or cuda")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = FormalReasonerFitConfig()
    fork_devices: list[int] = []
    if execution_device.type == "cuda":
        fork_devices = [
            execution_device.index
            if execution_device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(config.seed)
        model = _v1.DevelopmentIVAdditiveReasoner(hidden_dim=config.hidden_dim).to(
            execution_device
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_state = development_reasoner_state_sha256(model)
    epoch_rows = []
    for epoch in range(config.epochs):
        data.assert_unchanged()
        order = _patient_order(data.patient_ids, seed=config.seed, epoch=epoch)
        totals: list[float] = []
        bces: list[float] = []
        rankings: list[float] = []
        event_count = 0
        model.train()
        for raw_batch in data.dataset.iter_epoch(order):
            batch = raw_batch.to(execution_device)
            optimizer.zero_grad(set_to_none=True)
            step = _v1.development_reasoner_step(model, batch)
            step.loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            totals.append(float(step.loss.total.detach().cpu()))
            bces.append(float(step.loss.bce.detach().cpu()))
            rankings.append(float(step.loss.ranking.detach().cpu()))
            event_count += int(step.event_counts.sum().item())
        if len(totals) != 65 or event_count != 582:
            raise RuntimeError("v1.1 fit epoch did not visit the exact signed roster")
        epoch_rows.append(
            DevelopmentReasonerEpochReceiptV11(
                epoch_index=epoch,
                patient_order_sha256=_v1._canonical_sha256(order),
                mean_total_loss=sum(totals) / len(totals),
                mean_bce_loss=sum(bces) / len(bces),
                mean_ranking_loss=sum(rankings) / len(rankings),
                patient_count=len(totals),
                event_count=event_count,
            )
        )
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()
    model.requires_grad_(False)
    final_state = development_reasoner_state_sha256(model)
    train_diagnostic, _, _ = _diagnostic_once(model, data, device=execution_device)
    if development_reasoner_state_sha256(model) != final_state:
        raise RuntimeError("Source-train postfit diagnostic changed checkpoint state")
    data.assert_unchanged()
    receipt = DevelopmentReasonerFitReceiptV11(
        split_dataset_receipt_sha256=data.receipt.receipt_sha256,
        capability_manifest_sha256=data.receipt.capability_manifest_sha256,
        evidence_authorization_sha256=data.receipt.evidence_authorization_sha256,
        target_scope_receipt_sha256=data.receipt.target_scope_receipt_sha256,
        target_tensor_file_sha256=data.receipt.target_tensor_file_sha256,
        config=config,
        config_sha256=config.receipt_sha256,
        fit_policy_sha256=DEVELOPMENT_REASONER_FIT_POLICY_SHA256_V1_1,
        initial_state_sha256=initial_state,
        final_state_sha256=final_state,
        parameter_count=parameter_count,
        epochs=tuple(epoch_rows),
        source_train_postfit_diagnostic=train_diagnostic,
    )
    return VerifiedDevelopmentReasonerFitRunV11(
        _verification_marker=_FIT_RUN_MARKER, model=model, receipt=receipt
    )


@dataclass(frozen=True)
class PublishedDevelopmentReasonerFitArtifactV11:
    path: Path
    manifest_sha256: str
    checkpoint_file_sha256: str
    run: VerifiedDevelopmentReasonerFitRunV11 = field(repr=False)

    @property
    def fit_receipt_sha256(self) -> str:
        return self.run.receipt.receipt_sha256


def _fit_manifest(
    run: VerifiedDevelopmentReasonerFitRunV11,
    *,
    checkpoint_record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": DEVELOPMENT_REASONER_FIT_ARTIFACT_SCHEMA_V1_1,
        "purpose": "labram_iv_signal_eligible_train_only_frozen_checkpoint",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "fit_policy": dict(_FIT_POLICY),
        "fit_policy_sha256": DEVELOPMENT_REASONER_FIT_POLICY_SHA256_V1_1,
        "fit_receipt": asdict(run.receipt),
        "fit_receipt_sha256": run.receipt.receipt_sha256,
        "model_schema_version": "soz_development_iv_additive_reasoner_v1",
        "source_train_target_values_loaded": True,
        "source_dev_target_values_reachable": False,
        "source_dev_evidence_loaded_with_target_free_capability": True,
        "source_dev_evidence_used_for_fit_or_statistics": False,
        "source_dev_forward_count": 0,
        "checkpoint_selection": "final_epoch_20_only",
        "checkpoint_parameters_frozen": True,
        "threshold_selected": False,
        "calibrator_fitted": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "files": {FIT_CHECKPOINT_FILENAME: dict(checkpoint_record)},
    }


def publish_development_reasoner_fit_v1_1(
    run: VerifiedDevelopmentReasonerFitRunV11,
    output_directory: str | Path,
) -> PublishedDevelopmentReasonerFitArtifactV11:
    if type(run) is not VerifiedDevelopmentReasonerFitRunV11:
        raise TypeError("Only the closed v1.1 fitter may publish a checkpoint")
    run.assert_unchanged()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for fit publication") from exc
    target = _safe_new_directory(output_directory, field_name="v1.1 fit output")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        checkpoint = temporary / FIT_CHECKPOINT_FILENAME
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in run.model.state_dict().items()
        }
        save_file(state, str(checkpoint))
        record = {
            "sha256": _v1._file_sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "state_sha256": development_reasoner_state_sha256(state),
        }
        manifest = _fit_manifest(run, checkpoint_record=record)
        raw = _v1._canonical_json_bytes(manifest)
        manifest_path = temporary / FIT_MANIFEST_FILENAME
        manifest_path.write_bytes(raw)
        _fsync_file(checkpoint)
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_development_reasoner_fit_v1_1(
            target, expected_manifest_sha256=hashlib.sha256(raw).hexdigest()
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _epoch_from_payload(value: object) -> DevelopmentReasonerEpochReceiptV11:
    if not isinstance(value, Mapping) or set(value) != {
        row.name for row in fields(DevelopmentReasonerEpochReceiptV11)
    }:
        raise ValueError("v1.1 epoch receipt schema changed")
    return DevelopmentReasonerEpochReceiptV11(**dict(value))


def _summary_from_payload(value: object) -> DevelopmentReasonerDiagnosticSummaryV11:
    if not isinstance(value, Mapping) or set(value) != {
        row.name for row in fields(DevelopmentReasonerDiagnosticSummaryV11)
    }:
        raise ValueError("v1.1 diagnostic summary schema changed")
    return DevelopmentReasonerDiagnosticSummaryV11(**dict(value))


def _fit_receipt_from_payload(value: object) -> DevelopmentReasonerFitReceiptV11:
    if not isinstance(value, Mapping) or set(value) != {
        row.name for row in fields(DevelopmentReasonerFitReceiptV11)
    }:
        raise ValueError("v1.1 fit receipt schema changed")
    payload = dict(value)
    try:
        payload["config"] = FormalReasonerFitConfig(**payload["config"])
        payload["epochs"] = tuple(_epoch_from_payload(row) for row in payload["epochs"])
        payload["source_train_postfit_diagnostic"] = _summary_from_payload(
            payload["source_train_postfit_diagnostic"]
        )
        return DevelopmentReasonerFitReceiptV11(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v1.1 fit receipt is invalid") from exc


def load_development_reasoner_fit_v1_1(
    bundle_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> PublishedDevelopmentReasonerFitArtifactV11:
    try:
        from safetensors.torch import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for fit loading") from exc
    bundle = _closed_directory(
        bundle_directory, _FIT_FILES, name="v1.1 fit artifact"
    )
    manifest_raw = _read_stable_file(
        bundle / FIT_MANIFEST_FILENAME,
        field_name="v1.1 fit manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != _sha(expected_manifest_sha256, field_name="expected_manifest_sha256"):
        raise ValueError("v1.1 fit manifest SHA mismatch")
    manifest = _v1._strict_json(manifest_raw, field_name="v1.1 fit manifest")
    if not isinstance(manifest, Mapping) or _v1._canonical_json_bytes(manifest) != manifest_raw:
        raise ValueError("v1.1 fit manifest is not canonical JSON")
    fixed = {
        "schema_version": DEVELOPMENT_REASONER_FIT_ARTIFACT_SCHEMA_V1_1,
        "purpose": "labram_iv_signal_eligible_train_only_frozen_checkpoint",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "fit_policy": _FIT_POLICY,
        "fit_policy_sha256": DEVELOPMENT_REASONER_FIT_POLICY_SHA256_V1_1,
        "model_schema_version": "soz_development_iv_additive_reasoner_v1",
        "source_train_target_values_loaded": True,
        "source_dev_target_values_reachable": False,
        "source_dev_evidence_loaded_with_target_free_capability": True,
        "source_dev_evidence_used_for_fit_or_statistics": False,
        "source_dev_forward_count": 0,
        "checkpoint_selection": "final_epoch_20_only",
        "checkpoint_parameters_frozen": True,
        "threshold_selected": False,
        "calibrator_fitted": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
    }
    if set(manifest) != set(fixed) | {
        "fit_receipt",
        "fit_receipt_sha256",
        "files",
    }:
        raise ValueError("v1.1 fit manifest schema changed")
    if any(manifest.get(name) != value for name, value in fixed.items()):
        raise ValueError("v1.1 fit artifact scientific boundary changed")
    files_payload = manifest.get("files")
    if not isinstance(files_payload, Mapping) or set(files_payload) != {FIT_CHECKPOINT_FILENAME}:
        raise ValueError("v1.1 fit checkpoint record changed")
    record = files_payload[FIT_CHECKPOINT_FILENAME]
    if not isinstance(record, Mapping) or set(record) != {"sha256", "size_bytes", "state_sha256"}:
        raise ValueError("v1.1 fit checkpoint schema changed")
    if (
        _sha(record.get("sha256"), field_name="fit checkpoint sha256")
        != record.get("sha256")
        or _sha(record.get("state_sha256"), field_name="fit checkpoint state_sha256")
        != record.get("state_sha256")
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or not 1 <= record["size_bytes"] <= _MAX_TENSOR_BYTES
    ):
        raise ValueError("v1.1 fit checkpoint record is invalid")
    checkpoint_raw = _read_stable_file(
        bundle / FIT_CHECKPOINT_FILENAME,
        field_name="v1.1 fit checkpoint",
        maximum_bytes=_MAX_TENSOR_BYTES,
    )
    if len(checkpoint_raw) != record["size_bytes"] or hashlib.sha256(checkpoint_raw).hexdigest() != record["sha256"]:
        raise ValueError("v1.1 fit checkpoint bytes changed")
    try:
        state = load(checkpoint_raw)
    except Exception as exc:
        raise ValueError("v1.1 fit checkpoint is invalid safetensors") from exc
    if development_reasoner_state_sha256(state) != record["state_sha256"]:
        raise ValueError("v1.1 fit checkpoint tensor state changed")
    receipt = _fit_receipt_from_payload(manifest.get("fit_receipt"))
    if receipt.receipt_sha256 != manifest.get("fit_receipt_sha256"):
        raise ValueError("v1.1 fit receipt SHA mismatch")
    model = _v1.DevelopmentIVAdditiveReasoner(hidden_dim=receipt.config.hidden_dim)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("v1.1 fit checkpoint state schema changed") from exc
    model.eval()
    model.requires_grad_(False)
    run = VerifiedDevelopmentReasonerFitRunV11(
        _verification_marker=_FIT_RUN_MARKER, model=model, receipt=receipt
    )
    expected = _fit_manifest(run, checkpoint_record=record)
    if _v1._canonical_json_bytes(expected) != manifest_raw:
        raise ValueError("v1.1 fit manifest differs from reconstructed checkpoint")
    return PublishedDevelopmentReasonerFitArtifactV11(
        path=bundle,
        manifest_sha256=manifest_sha,
        checkpoint_file_sha256=str(record["sha256"]),
        run=run,
    )


@dataclass(frozen=True)
class DevelopmentReasonerDevDiagnosticReceiptV11:
    fit_manifest_sha256: str
    fit_receipt_sha256: str
    checkpoint_file_sha256: str
    split_dataset_receipt_sha256: str
    capability_manifest_sha256: str
    evidence_authorization_sha256: str
    target_scope_receipt_sha256: str
    target_tensor_file_sha256: str
    patient_roster_sha256: str
    channel_order_sha256: str
    diagnostic_policy_sha256: str
    checkpoint_state_before_sha256: str
    checkpoint_state_after_sha256: str
    summary: DevelopmentReasonerDiagnosticSummaryV11
    optimizer_instantiated: bool = False
    all_parameters_require_grad_false: bool = True
    full_bag_forward_count: int = 1
    checkpoint_state_unchanged: bool = True
    numeric_explanation_reconstructs_logits: bool = True
    threshold_selected: bool = False
    calibrator_fitted: bool = False
    source_eval_used: bool = False
    private_used: bool = False
    formal_reasoner_authorized: bool = False
    formal_promotion: bool = False
    schema_version: str = DEVELOPMENT_REASONER_DIAGNOSTIC_SCHEMA_V1_1

    def __post_init__(self) -> None:
        for name in (
            "fit_manifest_sha256",
            "fit_receipt_sha256",
            "checkpoint_file_sha256",
            "split_dataset_receipt_sha256",
            "capability_manifest_sha256",
            "evidence_authorization_sha256",
            "target_scope_receipt_sha256",
            "target_tensor_file_sha256",
            "patient_roster_sha256",
            "channel_order_sha256",
            "diagnostic_policy_sha256",
            "checkpoint_state_before_sha256",
            "checkpoint_state_after_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), field_name=name))
        if self.summary.model_split != "source_dev":
            raise ValueError("v1.1 post-freeze diagnostic requires source_dev")
        if self.diagnostic_policy_sha256 != DEVELOPMENT_REASONER_DIAGNOSTIC_POLICY_SHA256_V1_1:
            raise ValueError("v1.1 diagnostic policy changed")
        if self.channel_order_sha256 != _CHANNEL_ORDER_SHA256:
            raise ValueError("v1.1 diagnostic channel order changed")
        frozen_lineage = {
            "capability_manifest_sha256": FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
            "evidence_authorization_sha256": FROZEN_V1_1_AUTHORIZATION_RECEIPT_SHA256,
            "target_scope_receipt_sha256": FROZEN_SOURCE_DEV_TARGET_SCOPE_RECEIPT_SHA256,
            "target_tensor_file_sha256": FROZEN_SOURCE_DEV_TARGET_TENSOR_FILE_SHA256,
        }
        if any(
            getattr(self, name) != expected
            for name, expected in frozen_lineage.items()
        ):
            raise ValueError("v1.1 diagnostic frozen lineage changed")
        if self.checkpoint_state_before_sha256 != self.checkpoint_state_after_sha256:
            raise ValueError("v1.1 diagnostic changed checkpoint state")
        fixed = {
            "optimizer_instantiated": False,
            "all_parameters_require_grad_false": True,
            "full_bag_forward_count": 1,
            "checkpoint_state_unchanged": True,
            "numeric_explanation_reconstructs_logits": True,
            "threshold_selected": False,
            "calibrator_fitted": False,
            "source_eval_used": False,
            "private_used": False,
            "formal_reasoner_authorized": False,
            "formal_promotion": False,
            "schema_version": DEVELOPMENT_REASONER_DIAGNOSTIC_SCHEMA_V1_1,
        }
        if any(getattr(self, name) != value for name, value in fixed.items()):
            raise ValueError("v1.1 diagnostic scientific boundary changed")

    @property
    def receipt_sha256(self) -> str:
        return _v1._canonical_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class VerifiedDevelopmentReasonerDevDiagnosticV11:
    receipt: DevelopmentReasonerDevDiagnosticReceiptV11
    patient_ids: tuple[str, ...]
    tensors: Mapping[str, torch.Tensor] = field(repr=False)
    component_map: tuple[tuple[str, str], ...]
    _receipt_sha256: str = field(repr=False)
    _tensor_specs_sha256: str = field(repr=False)

    def __init__(
        self,
        *,
        _verification_marker: object,
        receipt: DevelopmentReasonerDevDiagnosticReceiptV11,
        patient_ids: Sequence[str],
        tensors: Mapping[str, torch.Tensor],
        component_map: Sequence[tuple[str, str]],
    ) -> None:
        if _verification_marker is not _DIAGNOSTIC_RUN_MARKER:
            raise TypeError("v1.1 diagnostic requires the closed diagnostic runner")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "patient_ids", tuple(patient_ids))
        object.__setattr__(
            self,
            "tensors",
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        )
        object.__setattr__(self, "component_map", tuple(component_map))
        object.__setattr__(self, "_receipt_sha256", receipt.receipt_sha256)
        object.__setattr__(
            self,
            "_tensor_specs_sha256",
            _v1._canonical_sha256(_tensor_specs(self.tensors)),
        )
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        if len(self.patient_ids) != 16 or len(set(self.patient_ids)) != 16:
            raise ValueError("v1.1 diagnostic patient roster changed")
        if self.component_map != _EXPECTED_COMPONENT_MAP:
            raise ValueError("v1.1 diagnostic component map changed")
        if _patient_roster_sha256(
            "source_dev", self.patient_ids
        ) != self.receipt.patient_roster_sha256:
            raise ValueError("v1.1 diagnostic patient roster identity changed")
        if self.receipt.receipt_sha256 != self._receipt_sha256:
            raise ValueError("v1.1 diagnostic receipt changed in memory")
        if _v1._canonical_sha256(_tensor_specs(self.tensors)) != self._tensor_specs_sha256:
            raise ValueError("v1.1 diagnostic tensors changed in memory")
        required = {
            "patient_logits",
            "patient_probabilities",
            "patient_abstain_recommended",
            "event_counts",
        } | {key for _, key in self.component_map}
        if set(self.tensors) != required:
            raise ValueError("v1.1 diagnostic tensor schema changed")
        logits = self.tensors["patient_logits"]
        probabilities = self.tensors["patient_probabilities"]
        abstain = self.tensors["patient_abstain_recommended"]
        event_counts = self.tensors["event_counts"]
        if tuple(logits.shape) != (16, 19) or probabilities.shape != logits.shape:
            raise ValueError("v1.1 diagnostic logit shape changed")
        if (
            not logits.is_floating_point()
            or probabilities.dtype != logits.dtype
            or not torch.isfinite(logits).all()
            or not torch.isfinite(probabilities).all()
            or abstain.dtype != torch.bool
            or tuple(abstain.shape) != (16,)
            or event_counts.dtype != torch.long
            or tuple(event_counts.shape) != (16,)
            or torch.any(event_counts < 1)
            or int(event_counts.sum().item()) != 221
            or int(abstain.sum().item())
            != self.receipt.summary.patient_abstain_recommended_count
        ):
            raise ValueError("v1.1 diagnostic output tensor contract changed")
        for _, key in self.component_map:
            component = self.tensors[key]
            if (
                component.shape != logits.shape
                or component.dtype != logits.dtype
                or not torch.isfinite(component).all()
            ):
                raise ValueError("v1.1 diagnostic component tensor contract changed")
        if not torch.allclose(probabilities, torch.sigmoid(logits), atol=1e-7, rtol=1e-7):
            raise ValueError("v1.1 diagnostic probabilities do not match logits")
        reconstructed = sum(self.tensors[key] for _, key in self.component_map)
        if not torch.allclose(reconstructed, logits, atol=1e-6, rtol=1e-6):
            raise ValueError("v1.1 diagnostic explanation no longer reconstructs logits")
        if _v1._tensor_sha256("patient_logits", logits) != self.receipt.summary.patient_logits_sha256:
            raise ValueError("v1.1 diagnostic logits changed")
        if _numeric_explanation_receipt_sha256(
            self.tensors, self.component_map
        ) != self.receipt.summary.numeric_explanation_receipt_sha256:
            raise ValueError("v1.1 diagnostic numerical explanation changed")
        if _diagnostic_tensor_set_sha256(
            self.tensors, self.component_map
        ) != self.receipt.summary.diagnostic_tensor_set_sha256:
            raise ValueError("v1.1 diagnostic tensor set changed")


def diagnose_development_iv_reasoner_v1_1(
    fit: PublishedDevelopmentReasonerFitArtifactV11,
    data: VerifiedDevelopmentReasonerSplitDatasetV11,
    *,
    device: str | torch.device = "cpu",
) -> VerifiedDevelopmentReasonerDevDiagnosticV11:
    """Run the single post-freeze source-dev full-bag diagnostic."""

    if type(fit) is not PublishedDevelopmentReasonerFitArtifactV11:
        raise TypeError("v1.1 diagnostic requires a strict frozen fit artifact")
    if type(data) is not VerifiedDevelopmentReasonerSplitDatasetV11 or data.model_split != "source_dev":
        raise ValueError("v1.1 diagnostic requires the strict source_dev dataset")
    fit.run.assert_unchanged()
    data.assert_unchanged()
    fit_receipt = fit.run.receipt
    if (
        data.receipt.capability_manifest_sha256 != fit_receipt.capability_manifest_sha256
        or data.receipt.evidence_authorization_sha256
        != fit_receipt.evidence_authorization_sha256
        or data.receipt.original_target_receipt_sha256
        != FROZEN_TARGET_V2_RECEIPT_SHA256
    ):
        raise ValueError("v1.1 fit and diagnostic dataset lineage disagree")
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("v1.1 diagnostic device must be cpu or cuda")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = fit.run.model.to(execution_device)
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("v1.1 diagnostic checkpoint is not frozen")
    state_before = development_reasoner_state_sha256(model)
    summary, tensors, component_map = _diagnostic_once(
        model, data, device=execution_device
    )
    state_after = development_reasoner_state_sha256(model)
    if state_before != state_after or state_before != fit_receipt.final_state_sha256:
        raise RuntimeError("v1.1 diagnostic changed the frozen checkpoint")
    data.assert_unchanged()
    receipt = DevelopmentReasonerDevDiagnosticReceiptV11(
        fit_manifest_sha256=fit.manifest_sha256,
        fit_receipt_sha256=fit.fit_receipt_sha256,
        checkpoint_file_sha256=fit.checkpoint_file_sha256,
        split_dataset_receipt_sha256=data.receipt.receipt_sha256,
        capability_manifest_sha256=data.receipt.capability_manifest_sha256,
        evidence_authorization_sha256=data.receipt.evidence_authorization_sha256,
        target_scope_receipt_sha256=data.receipt.target_scope_receipt_sha256,
        target_tensor_file_sha256=data.receipt.target_tensor_file_sha256,
        patient_roster_sha256=data.receipt.patient_roster_sha256,
        channel_order_sha256=_CHANNEL_ORDER_SHA256,
        diagnostic_policy_sha256=DEVELOPMENT_REASONER_DIAGNOSTIC_POLICY_SHA256_V1_1,
        checkpoint_state_before_sha256=state_before,
        checkpoint_state_after_sha256=state_after,
        summary=summary,
    )
    return VerifiedDevelopmentReasonerDevDiagnosticV11(
        _verification_marker=_DIAGNOSTIC_RUN_MARKER,
        receipt=receipt,
        patient_ids=data.patient_ids,
        tensors=tensors,
        component_map=component_map,
    )


@dataclass(frozen=True)
class PublishedDevelopmentReasonerDevDiagnosticV11:
    path: Path
    manifest_sha256: str
    tensor_file_sha256: str
    run: VerifiedDevelopmentReasonerDevDiagnosticV11 = field(repr=False)

    @property
    def diagnostic_receipt_sha256(self) -> str:
        return self.run.receipt.receipt_sha256


def _diagnostic_manifest(
    run: VerifiedDevelopmentReasonerDevDiagnosticV11,
    *,
    tensor_file_record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": DEVELOPMENT_REASONER_DIAGNOSTIC_ARTIFACT_SCHEMA_V1_1,
        "purpose": "append_only_post_freeze_source_dev_diagnostic",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "diagnostic_policy": dict(_DIAGNOSTIC_POLICY),
        "diagnostic_policy_sha256": DEVELOPMENT_REASONER_DIAGNOSTIC_POLICY_SHA256_V1_1,
        "diagnostic_receipt": asdict(run.receipt),
        "diagnostic_receipt_sha256": run.receipt.receipt_sha256,
        "patient_ids": list(run.patient_ids),
        "component_map": [list(row) for row in run.component_map],
        "tensor_specs": _tensor_specs(run.tensors),
        "optimizer_instantiated": False,
        "all_parameters_require_grad_false": True,
        "full_bag_forward_count": 1,
        "checkpoint_state_unchanged": True,
        "numeric_explanation_reconstructs_logits": True,
        "threshold_selected": False,
        "calibrator_fitted": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
        "files": {DIAGNOSTIC_TENSORS_FILENAME: dict(tensor_file_record)},
    }


def publish_development_reasoner_dev_diagnostic_v1_1(
    run: VerifiedDevelopmentReasonerDevDiagnosticV11,
    output_directory: str | Path,
) -> PublishedDevelopmentReasonerDevDiagnosticV11:
    if type(run) is not VerifiedDevelopmentReasonerDevDiagnosticV11:
        raise TypeError("Only the closed v1.1 diagnostic may be published")
    run.assert_unchanged()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for diagnostic publication") from exc
    target = _safe_new_directory(
        output_directory, field_name="v1.1 append-only diagnostic output"
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = temporary / DIAGNOSTIC_TENSORS_FILENAME
        save_file(dict(run.tensors), str(tensor_path))
        record = {
            "sha256": _v1._file_sha256(tensor_path),
            "size_bytes": tensor_path.stat().st_size,
        }
        manifest = _diagnostic_manifest(run, tensor_file_record=record)
        raw = _v1._canonical_json_bytes(manifest)
        manifest_path = temporary / DIAGNOSTIC_MANIFEST_FILENAME
        manifest_path.write_bytes(raw)
        _fsync_file(tensor_path)
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_development_reasoner_dev_diagnostic_v1_1(
            target, expected_manifest_sha256=hashlib.sha256(raw).hexdigest()
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _dev_receipt_from_payload(value: object) -> DevelopmentReasonerDevDiagnosticReceiptV11:
    if not isinstance(value, Mapping) or set(value) != {
        row.name for row in fields(DevelopmentReasonerDevDiagnosticReceiptV11)
    }:
        raise ValueError("v1.1 dev diagnostic receipt schema changed")
    payload = dict(value)
    payload["summary"] = _summary_from_payload(payload.get("summary"))
    try:
        return DevelopmentReasonerDevDiagnosticReceiptV11(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("v1.1 dev diagnostic receipt is invalid") from exc


def load_development_reasoner_dev_diagnostic_v1_1(
    bundle_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> PublishedDevelopmentReasonerDevDiagnosticV11:
    try:
        from safetensors.torch import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for diagnostic loading") from exc
    bundle = _closed_directory(
        bundle_directory, _DIAGNOSTIC_FILES, name="v1.1 diagnostic artifact"
    )
    raw = _read_stable_file(
        bundle / DIAGNOSTIC_MANIFEST_FILENAME,
        field_name="v1.1 diagnostic manifest",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != _sha(expected_manifest_sha256, field_name="expected_manifest_sha256"):
        raise ValueError("v1.1 diagnostic manifest SHA mismatch")
    manifest = _v1._strict_json(raw, field_name="v1.1 diagnostic manifest")
    if not isinstance(manifest, Mapping) or _v1._canonical_json_bytes(manifest) != raw:
        raise ValueError("v1.1 diagnostic manifest is not canonical JSON")
    fixed = {
        "schema_version": DEVELOPMENT_REASONER_DIAGNOSTIC_ARTIFACT_SCHEMA_V1_1,
        "purpose": "append_only_post_freeze_source_dev_diagnostic",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "diagnostic_policy": _DIAGNOSTIC_POLICY,
        "diagnostic_policy_sha256": DEVELOPMENT_REASONER_DIAGNOSTIC_POLICY_SHA256_V1_1,
        "optimizer_instantiated": False,
        "all_parameters_require_grad_false": True,
        "full_bag_forward_count": 1,
        "checkpoint_state_unchanged": True,
        "numeric_explanation_reconstructs_logits": True,
        "threshold_selected": False,
        "calibrator_fitted": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_reasoner_authorized": False,
        "formal_promotion": False,
    }
    if set(manifest) != set(fixed) | {
        "diagnostic_receipt",
        "diagnostic_receipt_sha256",
        "patient_ids",
        "component_map",
        "tensor_specs",
        "files",
    }:
        raise ValueError("v1.1 diagnostic manifest schema changed")
    if any(manifest.get(name) != value for name, value in fixed.items()):
        raise ValueError("v1.1 diagnostic artifact scientific boundary changed")
    files_payload = manifest.get("files")
    if not isinstance(files_payload, Mapping) or set(files_payload) != {DIAGNOSTIC_TENSORS_FILENAME}:
        raise ValueError("v1.1 diagnostic tensor record changed")
    record = files_payload[DIAGNOSTIC_TENSORS_FILENAME]
    if not isinstance(record, Mapping) or set(record) != {"sha256", "size_bytes"}:
        raise ValueError("v1.1 diagnostic tensor file schema changed")
    if (
        _sha(record.get("sha256"), field_name="diagnostic tensor sha256")
        != record.get("sha256")
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or not 1 <= record["size_bytes"] <= _MAX_TENSOR_BYTES
    ):
        raise ValueError("v1.1 diagnostic tensor file record is invalid")
    tensor_raw = _read_stable_file(
        bundle / DIAGNOSTIC_TENSORS_FILENAME,
        field_name="v1.1 diagnostic tensors",
        maximum_bytes=_MAX_TENSOR_BYTES,
    )
    if len(tensor_raw) != record["size_bytes"] or hashlib.sha256(tensor_raw).hexdigest() != record["sha256"]:
        raise ValueError("v1.1 diagnostic tensor bytes changed")
    try:
        tensors = load(tensor_raw)
    except Exception as exc:
        raise ValueError("v1.1 diagnostic tensor payload is invalid") from exc
    if _tensor_specs(tensors) != manifest.get("tensor_specs"):
        raise ValueError("v1.1 diagnostic tensor specs changed")
    patient_ids = manifest.get("patient_ids")
    component_rows = manifest.get("component_map")
    if not isinstance(patient_ids, list) or not isinstance(component_rows, list) or any(
        not isinstance(row, list) or len(row) != 2 for row in component_rows
    ):
        raise ValueError("v1.1 diagnostic identity/component schema changed")
    receipt = _dev_receipt_from_payload(manifest.get("diagnostic_receipt"))
    if receipt.receipt_sha256 != manifest.get("diagnostic_receipt_sha256"):
        raise ValueError("v1.1 diagnostic receipt SHA mismatch")
    run = VerifiedDevelopmentReasonerDevDiagnosticV11(
        _verification_marker=_DIAGNOSTIC_RUN_MARKER,
        receipt=receipt,
        patient_ids=tuple(str(value) for value in patient_ids),
        tensors=tensors,
        component_map=tuple((str(row[0]), str(row[1])) for row in component_rows),
    )
    expected = _diagnostic_manifest(run, tensor_file_record=record)
    if _v1._canonical_json_bytes(expected) != raw:
        raise ValueError("v1.1 diagnostic manifest differs from reconstructed receipt")
    return PublishedDevelopmentReasonerDevDiagnosticV11(
        path=bundle,
        manifest_sha256=manifest_sha,
        tensor_file_sha256=str(record["sha256"]),
        run=run,
    )


__all__ = [
    "DEVELOPMENT_REASONER_DIAGNOSTIC_POLICY_SHA256_V1_1",
    "DEVELOPMENT_REASONER_FIT_POLICY_SHA256_V1_1",
    "FROZEN_SOURCE_DEV_TARGET_SCOPE_RECEIPT_SHA256",
    "FROZEN_SOURCE_DEV_TARGET_TENSOR_FILE_SHA256",
    "FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256",
    "FROZEN_SOURCE_TRAIN_TARGET_TENSOR_FILE_SHA256",
    "DevelopmentReasonerDevDiagnosticReceiptV11",
    "DevelopmentReasonerDiagnosticSummaryV11",
    "DevelopmentReasonerEpochReceiptV11",
    "DevelopmentReasonerFitReceiptV11",
    "DevelopmentReasonerSplitDatasetReceiptV11",
    "PublishedDevelopmentReasonerDevDiagnosticV11",
    "PublishedDevelopmentReasonerFitArtifactV11",
    "VerifiedDevelopmentReasonerDevDiagnosticV11",
    "VerifiedDevelopmentReasonerFitRunV11",
    "VerifiedDevelopmentReasonerSplitDatasetV11",
    "diagnose_development_iv_reasoner_v1_1",
    "fit_development_iv_reasoner_v1_1",
    "join_development_iv_split_targets_v1_1",
    "load_development_reasoner_dev_diagnostic_v1_1",
    "load_development_reasoner_fit_v1_1",
    "publish_development_reasoner_dev_diagnostic_v1_1",
    "publish_development_reasoner_fit_v1_1",
]
