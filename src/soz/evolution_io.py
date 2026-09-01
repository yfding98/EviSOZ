"""Strict canonical artifact IO for computed temporal-evolution scalers.

The scaler statistics are deterministic but fold-specific.  This module binds
the six robust statistics to the exact preflighted TUSZ fit manifest and the
frozen ictal OOF plan that authorized it.  It stores canonical JSON only; raw
descriptors, tensors, pickle payloads, and filesystem source paths are absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

from .concept_oof import IctalConceptOOFPlan, IctalConceptOOFProtocol
from .data.edf import CausalEDFConfig, EDF_PREPROCESS_SCHEMA
from .data.overlap import (
    canonical_public_roster_sha256,
    normalize_public_patient_key,
)
from .data.tusz_training import (
    TUSZIctalTrainingManifest,
    load_tusz_ictal_training_manifest,
)
from .evolution import (
    COMPLETE19_DESCRIPTOR_MASK_SHA256,
    EvolutionDescriptorReceipt,
    PatientBalancedRobustScaler,
    PatientBalancedScalerReceipt,
    patient_roster_sha256,
)


COMPUTED_EVOLUTION_EVENT_RECEIPT_SCHEMA = (
    "soz_computed_evolution_event_replay_receipt_v2"
)
COMPUTED_EVOLUTION_COMPUTATION_RECEIPT_SCHEMA = (
    "soz_computed_evolution_runtime_source_receipt_v1"
)
COMPUTED_EVOLUTION_FIT_RESULT_SCHEMA = "soz_computed_evolution_fit_result_v1"
VERIFIED_EVOLUTION_FIT_RECEIPT_SCHEMA = (
    "soz_verified_evolution_independent_replay_receipt_v1"
)
COMPUTED_EVOLUTION_SCALER_RECEIPT_SCHEMA = (
    "soz_computed_evolution_scaler_artifact_receipt_v3"
)
COMPUTED_EVOLUTION_SCALER_ARTIFACT_SCHEMA = (
    "soz_computed_evolution_scaler_artifact_v3"
)
COMPUTED_EVOLUTION_SCALER_ARTIFACT_FILENAME = (
    "computed_evolution_scaler.json"
)
_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAX_MANIFEST_ENVELOPE_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VERIFIED_FIT_ISSUER_TOKEN = object()

_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "artifact_receipt_sha256",
        "artifact_receipt",
        "scaler_receipt",
        "computation_receipt",
        "verification_receipt",
    }
)
_ARTIFACT_RECEIPT_FIELDS = frozenset(
    {
        "oof_fold",
        "fit_public_patient_keys",
        "fit_patient_roster_sha256",
        "patient_count",
        "fit_event_receipts",
        "fit_event_roster_sha256",
        "patient_event_roster_sha256",
        "edf_roster_sha256",
        "signal_roster_sha256",
        "raw_descriptor_roster_sha256",
        "descriptor_mask_roster_sha256",
        "complete19_descriptor_mask_sha256",
        "event_count",
        "fit_manifest_source_sha256",
        "fit_manifest_bundle_sha256",
        "split_manifest_sha256",
        "oof_protocol_receipt_sha256",
        "oof_plan_receipt_sha256",
        "held_out_target_roster_sha256",
        "held_out_public_roster_sha256",
        "authorized_source_record_roster_sha256",
        "descriptor_schema_sha256",
        "preprocess_schema_sha256",
        "computation_receipt_sha256",
        "scaler_receipt_sha256",
        "schema_version",
    }
)
_COMPUTATION_RECEIPT_FIELDS = frozenset(
    {
        "evolution_source_sha256",
        "evolution_fit_source_sha256",
        "geometry_source_sha256",
        "torch_version",
        "numpy_version",
        "scipy_version",
        "platform_machine",
        "torch_num_threads",
        "torch_num_interop_threads",
        "compute_device_policy",
        "compute_dtype",
        "descriptor_output_dtype",
        "scaler_dtype",
        "mask_policy",
        "complete19_descriptor_mask_sha256",
        "schema_version",
    }
)
_VERIFICATION_RECEIPT_FIELDS = frozenset(
    {
        "candidate_fit_result_sha256",
        "independent_fit_result_sha256",
        "canonical_artifact_core_sha256",
        "event_receipt_roster_sha256",
        "scaler_receipt_sha256",
        "computation_receipt_sha256",
        "event_count",
        "patient_count",
        "raw_edf_replay_count",
        "verification_policy",
        "schema_version",
    }
)
_EVENT_RECEIPT_FIELDS = frozenset(
    {
        "event_id",
        "patient_id",
        "event_record_sha256",
        "edf_sha256",
        "signal_content_sha256",
        "signal_preflight_receipt_sha256",
        "signal_window_sha256",
        "raw_descriptor_sha256",
        "descriptor_mask_sha256",
        "schema_version",
    }
)
_SCALER_RECEIPT_FIELDS = frozenset(
    {
        "feature_names",
        "feature_schema_sha256",
        "patient_roster_sha256",
        "split_manifest_sha256",
        "fit_split_sha256",
        "fit_split",
        "patient_count",
        "patient_feature_medians_sha256",
        "center",
        "iqr",
        "scale",
        "clip",
        "statistic_scope",
        "patient_balance_policy",
        "zero_iqr_policy",
        "schema_version",
    }
)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Evolution scaler artifact is not canonical JSON data"
        ) from exc
    return (encoded + "\n").encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)[:-1]).hexdigest()


EVOLUTION_DESCRIPTOR_SCHEMA_SHA256 = _canonical_sha256(
    asdict(EvolutionDescriptorReceipt())
)


def evolution_preprocess_schema_sha256(config: CausalEDFConfig) -> str:
    """Hash the exact preprocessing schema and complete causal configuration."""

    if not isinstance(config, CausalEDFConfig):
        raise TypeError("config must be a CausalEDFConfig")
    return _canonical_sha256(
        {
            "edf_preprocess_schema": EDF_PREPROCESS_SCHEMA,
            "preprocess_config": asdict(config),
        }
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a JSON string")
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return value


def _normalize_oof_fold(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(5):
        raise ValueError(f"{field} must be an integer in [0,4] or null for final")
    return value


def _normalize_fit_roster(values: Sequence[object]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("fit_public_patient_keys must be a sequence")
    roster = tuple(sorted(normalize_public_patient_key(value) for value in values))
    if not roster or len(set(roster)) != len(roster):
        raise ValueError("Fit public-patient roster must be non-empty and unique")
    return roster


def _protocol_plan(
    protocol: IctalConceptOOFProtocol,
    oof_fold: int | None,
) -> IctalConceptOOFPlan:
    if not isinstance(protocol, IctalConceptOOFProtocol):
        raise TypeError("oof_protocol must be an IctalConceptOOFProtocol")
    fold = _normalize_oof_fold(oof_fold, field="oof_fold")
    return protocol.final_plan if fold is None else protocol.for_fold(fold)


@dataclass(frozen=True)
class ComputedEvolutionEventReceipt:
    """One target-free causal replay and its direct-descriptor identities."""

    event_id: str
    patient_id: str
    event_record_sha256: str
    edf_sha256: str
    signal_content_sha256: str
    signal_preflight_receipt_sha256: str
    signal_window_sha256: str
    raw_descriptor_sha256: str
    descriptor_mask_sha256: str
    schema_version: str = COMPUTED_EVOLUTION_EVENT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("Evolution replay event_id cannot be empty")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(
            self,
            "patient_id",
            normalize_public_patient_key(self.patient_id),
        )
        for field in (
            "event_record_sha256",
            "edf_sha256",
            "signal_content_sha256",
            "signal_preflight_receipt_sha256",
            "signal_window_sha256",
            "raw_descriptor_sha256",
            "descriptor_mask_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.schema_version != COMPUTED_EVOLUTION_EVENT_RECEIPT_SCHEMA:
            raise ValueError("Unsupported computed-evolution event receipt schema")
        if self.descriptor_mask_sha256 != COMPLETE19_DESCRIPTOR_MASK_SHA256:
            raise ValueError(
                "Evolution event receipt must use the frozen complete19 mask SHA"
            )


def _normalize_event_receipts(
    values: Sequence[ComputedEvolutionEventReceipt],
) -> tuple[ComputedEvolutionEventReceipt, ...]:
    if isinstance(values, (str, bytes)) or any(
        not isinstance(value, ComputedEvolutionEventReceipt) for value in values
    ):
        raise TypeError(
            "fit_event_receipts must contain ComputedEvolutionEventReceipt objects"
        )
    receipts = tuple(values)
    if not receipts:
        raise ValueError("Evolution scaler requires a non-empty event replay roster")
    canonical = tuple(
        sorted(receipts, key=lambda value: (value.patient_id, value.event_id))
    )
    if receipts != canonical:
        raise ValueError("Evolution event receipts must be canonically ordered")
    event_ids = tuple(value.event_id for value in receipts)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Evolution event receipts contain duplicate event IDs")
    return receipts


def _event_roster_hashes(
    receipts: Sequence[ComputedEvolutionEventReceipt],
) -> dict[str, str]:
    values = _normalize_event_receipts(receipts)
    patient_ids = tuple(sorted({value.patient_id for value in values}))
    return {
        "fit_event_roster_sha256": _canonical_sha256(
            tuple(
                (
                    value.event_id,
                    value.patient_id,
                    value.event_record_sha256,
                )
                for value in values
            )
        ),
        "patient_event_roster_sha256": _canonical_sha256(
            tuple(
                (
                    patient_id,
                    tuple(
                        value.event_id
                        for value in values
                        if value.patient_id == patient_id
                    ),
                )
                for patient_id in patient_ids
            )
        ),
        "edf_roster_sha256": _canonical_sha256(
            tuple((value.event_id, value.edf_sha256) for value in values)
        ),
        "signal_roster_sha256": _canonical_sha256(
            tuple(
                (
                    value.event_id,
                    value.signal_content_sha256,
                    value.signal_preflight_receipt_sha256,
                    value.signal_window_sha256,
                )
                for value in values
            )
        ),
        "raw_descriptor_roster_sha256": _canonical_sha256(
            tuple(
                (value.event_id, value.raw_descriptor_sha256)
                for value in values
            )
        ),
        "descriptor_mask_roster_sha256": _canonical_sha256(
            tuple(
                (value.event_id, value.descriptor_mask_sha256)
                for value in values
            )
        ),
    }


@dataclass(frozen=True)
class EvolutionComputationReceipt:
    """Exact source/runtime policy for deterministic CPU-float64 production."""

    evolution_source_sha256: str
    evolution_fit_source_sha256: str
    geometry_source_sha256: str
    torch_version: str
    numpy_version: str
    scipy_version: str
    platform_machine: str
    torch_num_threads: int
    torch_num_interop_threads: int
    compute_device_policy: str = "cpu_only_no_accelerator"
    compute_dtype: str = "torch.float64"
    descriptor_output_dtype: str = "torch.float64"
    scaler_dtype: str = "torch.float64"
    mask_policy: str = "constant_all_true_complete19_no_tile_qc"
    complete19_descriptor_mask_sha256: str = (
        COMPLETE19_DESCRIPTOR_MASK_SHA256
    )
    schema_version: str = COMPUTED_EVOLUTION_COMPUTATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "evolution_source_sha256",
            "evolution_fit_source_sha256",
            "geometry_source_sha256",
            "complete19_descriptor_mask_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        for field in (
            "torch_version",
            "numpy_version",
            "scipy_version",
            "platform_machine",
        ):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} cannot be empty")
            object.__setattr__(self, field, value)
        for field in ("torch_num_threads", "torch_num_interop_threads"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if (
            self.compute_device_policy != "cpu_only_no_accelerator"
            or self.compute_dtype != "torch.float64"
            or self.descriptor_output_dtype != "torch.float64"
            or self.scaler_dtype != "torch.float64"
        ):
            raise ValueError("Evolution computation receipt must freeze CPU float64")
        if self.mask_policy != "constant_all_true_complete19_no_tile_qc":
            raise ValueError("Evolution computation receipt mask policy drifted")
        if (
            self.complete19_descriptor_mask_sha256
            != COMPLETE19_DESCRIPTOR_MASK_SHA256
        ):
            raise ValueError("Evolution computation complete19 mask SHA drifted")
        if self.schema_version != COMPUTED_EVOLUTION_COMPUTATION_RECEIPT_SCHEMA:
            raise ValueError("Unsupported evolution computation receipt schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ComputedEvolutionFitResult:
    """Complete candidate output from one full raw-EDF replay pass."""

    scaler: PatientBalancedRobustScaler
    event_receipts: tuple[ComputedEvolutionEventReceipt, ...]
    computation_receipt: EvolutionComputationReceipt
    schema_version: str = COMPUTED_EVOLUTION_FIT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.scaler, PatientBalancedRobustScaler):
            raise TypeError("fit-result scaler must be PatientBalancedRobustScaler")
        receipts = _normalize_event_receipts(self.event_receipts)
        object.__setattr__(self, "event_receipts", receipts)
        if not isinstance(self.computation_receipt, EvolutionComputationReceipt):
            raise TypeError("fit result requires EvolutionComputationReceipt")
        patient_ids = tuple(sorted({value.patient_id for value in receipts}))
        if self.scaler.receipt.patient_count != len(patient_ids):
            raise ValueError("Fit-result scaler patient count disagrees with events")
        if self.scaler.receipt.patient_roster_sha256 != patient_roster_sha256(
            patient_ids
        ):
            raise ValueError("Fit-result scaler patient roster disagrees with events")
        for tensor in (self.scaler.center, self.scaler.iqr, self.scaler.scale):
            if tensor.dtype != torch.float64 or tensor.device.type != "cpu":
                raise TypeError("Fit-result scaler must remain CPU float64")
        if self.schema_version != COMPUTED_EVOLUTION_FIT_RESULT_SCHEMA:
            raise ValueError("Unsupported computed-evolution fit-result schema")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scaler_receipt": asdict(self.scaler.receipt),
            "event_receipts": tuple(asdict(value) for value in self.event_receipts),
            "computation_receipt": asdict(self.computation_receipt),
        }

    @property
    def fit_result_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)

    @property
    def event_receipt_roster_sha256(self) -> str:
        return _canonical_sha256(
            tuple(asdict(value) for value in self.event_receipts)
        )

    @property
    def scaler_receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self.scaler.receipt))


@dataclass(frozen=True)
class VerifiedEvolutionFitReceipt:
    """Proof that a distinct second raw-EDF pass reproduced the candidate."""

    candidate_fit_result_sha256: str
    independent_fit_result_sha256: str
    canonical_artifact_core_sha256: str
    event_receipt_roster_sha256: str
    scaler_receipt_sha256: str
    computation_receipt_sha256: str
    event_count: int
    patient_count: int
    raw_edf_replay_count: int = 2
    verification_policy: str = (
        "distinct_full_raw_edf_second_pass_exact_event_scaler_artifact_replay"
    )
    schema_version: str = VERIFIED_EVOLUTION_FIT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "candidate_fit_result_sha256",
            "independent_fit_result_sha256",
            "canonical_artifact_core_sha256",
            "event_receipt_roster_sha256",
            "scaler_receipt_sha256",
            "computation_receipt_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.candidate_fit_result_sha256 != self.independent_fit_result_sha256:
            raise ValueError("Independent evolution replay did not exactly reproduce fit")
        for field in ("event_count", "patient_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.raw_edf_replay_count != 2:
            raise ValueError("Formal evolution verification requires exactly two passes")
        if self.verification_policy != (
            "distinct_full_raw_edf_second_pass_exact_event_scaler_artifact_replay"
        ):
            raise ValueError("Evolution verification policy cannot be weakened")
        if self.schema_version != VERIFIED_EVOLUTION_FIT_RECEIPT_SCHEMA:
            raise ValueError("Unsupported verified-evolution receipt schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class VerifiedComputedEvolutionFitResult:
    """Capability object accepted by the formal scaler artifact builder."""

    fit_result: ComputedEvolutionFitResult
    verification_receipt: VerifiedEvolutionFitReceipt
    _issuer_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer_token is not _VERIFIED_FIT_ISSUER_TOKEN:
            raise PermissionError(
                "Verified evolution fit capability can only be issued by verifier"
            )
        _validate_verified_receipt_against_fit_result(
            self.fit_result, self.verification_receipt
        )
        # Do not retain the issuer secret on the capability: dataclasses.replace
        # must not be able to copy it onto a different candidate result.
        object.__setattr__(self, "_issuer_token", None)


def _validate_verified_receipt_against_fit_result(
    fit_result: ComputedEvolutionFitResult,
    verification_receipt: VerifiedEvolutionFitReceipt,
) -> None:
    """Authenticate a persisted replay receipt without issuing a capability."""

    if not isinstance(fit_result, ComputedEvolutionFitResult):
        raise TypeError("verified fit requires ComputedEvolutionFitResult")
    if not isinstance(verification_receipt, VerifiedEvolutionFitReceipt):
        raise TypeError("verified fit requires VerifiedEvolutionFitReceipt")
    receipt = verification_receipt
    if receipt.candidate_fit_result_sha256 != fit_result.fit_result_sha256:
        raise ValueError("Verified receipt does not authenticate the fit result")
    if (
        receipt.event_receipt_roster_sha256
        != fit_result.event_receipt_roster_sha256
        or receipt.scaler_receipt_sha256
        != fit_result.scaler_receipt_sha256
        or receipt.computation_receipt_sha256
        != fit_result.computation_receipt.receipt_sha256
    ):
        raise ValueError("Verified receipt disagrees with fit-result components")
    if receipt.event_count != len(fit_result.event_receipts):
        raise ValueError("Verified receipt event count disagrees with fit result")
    if receipt.patient_count != fit_result.scaler.receipt.patient_count:
        raise ValueError("Verified receipt patient count disagrees with fit result")


def _issue_verified_evolution_fit_result(
    candidate: ComputedEvolutionFitResult,
    independent: ComputedEvolutionFitResult,
    *,
    canonical_artifact_core_sha256: str,
    issuer_token: object,
) -> VerifiedComputedEvolutionFitResult:
    if issuer_token is not _VERIFIED_FIT_ISSUER_TOKEN:
        raise PermissionError("Verified evolution fit can only be issued by verifier")
    if candidate is independent:
        raise ValueError("Independent verifier cannot reuse the candidate result object")
    if candidate.canonical_payload != independent.canonical_payload:
        raise ValueError("Independent raw-EDF replay changed fit-result payload")
    for first, second, name in (
        (candidate.scaler.center, independent.scaler.center, "center"),
        (candidate.scaler.iqr, independent.scaler.iqr, "iqr"),
        (candidate.scaler.scale, independent.scaler.scale, "scale"),
    ):
        if not torch.equal(first, second):
            raise ValueError(f"Independent raw-EDF replay changed scaler {name}")
    receipt = VerifiedEvolutionFitReceipt(
        candidate_fit_result_sha256=candidate.fit_result_sha256,
        independent_fit_result_sha256=independent.fit_result_sha256,
        canonical_artifact_core_sha256=canonical_artifact_core_sha256,
        event_receipt_roster_sha256=candidate.event_receipt_roster_sha256,
        scaler_receipt_sha256=candidate.scaler_receipt_sha256,
        computation_receipt_sha256=candidate.computation_receipt.receipt_sha256,
        event_count=len(candidate.event_receipts),
        patient_count=candidate.scaler.receipt.patient_count,
    )
    return VerifiedComputedEvolutionFitResult(
        fit_result=candidate,
        verification_receipt=receipt,
        _issuer_token=issuer_token,
    )


@dataclass(frozen=True)
class ComputedEvolutionScalerArtifactReceipt:
    """Fold, fit-manifest, OOF-plan, and descriptor lineage."""

    oof_fold: int | None
    fit_public_patient_keys: tuple[str, ...]
    fit_patient_roster_sha256: str
    patient_count: int
    fit_event_receipts: tuple[ComputedEvolutionEventReceipt, ...]
    fit_event_roster_sha256: str
    patient_event_roster_sha256: str
    edf_roster_sha256: str
    signal_roster_sha256: str
    raw_descriptor_roster_sha256: str
    descriptor_mask_roster_sha256: str
    complete19_descriptor_mask_sha256: str
    event_count: int
    fit_manifest_source_sha256: str
    fit_manifest_bundle_sha256: str
    split_manifest_sha256: str
    oof_protocol_receipt_sha256: str
    oof_plan_receipt_sha256: str
    held_out_target_roster_sha256: str
    held_out_public_roster_sha256: str
    authorized_source_record_roster_sha256: str
    descriptor_schema_sha256: str
    preprocess_schema_sha256: str
    computation_receipt_sha256: str
    scaler_receipt_sha256: str
    schema_version: str = COMPUTED_EVOLUTION_SCALER_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "oof_fold",
            _normalize_oof_fold(self.oof_fold, field="oof_fold"),
        )
        roster = _normalize_fit_roster(self.fit_public_patient_keys)
        object.__setattr__(self, "fit_public_patient_keys", roster)
        event_receipts = _normalize_event_receipts(self.fit_event_receipts)
        object.__setattr__(self, "fit_event_receipts", event_receipts)
        for field in (
            "fit_patient_roster_sha256",
            "fit_event_roster_sha256",
            "patient_event_roster_sha256",
            "edf_roster_sha256",
            "signal_roster_sha256",
            "raw_descriptor_roster_sha256",
            "descriptor_mask_roster_sha256",
            "complete19_descriptor_mask_sha256",
            "fit_manifest_source_sha256",
            "fit_manifest_bundle_sha256",
            "split_manifest_sha256",
            "oof_protocol_receipt_sha256",
            "oof_plan_receipt_sha256",
            "held_out_target_roster_sha256",
            "held_out_public_roster_sha256",
            "authorized_source_record_roster_sha256",
            "descriptor_schema_sha256",
            "preprocess_schema_sha256",
            "computation_receipt_sha256",
            "scaler_receipt_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.fit_patient_roster_sha256 != patient_roster_sha256(roster):
            raise ValueError("Fit public-patient roster SHA does not match its roster")
        if (
            isinstance(self.patient_count, bool)
            or not isinstance(self.patient_count, int)
            or self.patient_count != len(roster)
        ):
            raise ValueError("patient_count must equal the exact fit roster")
        if (
            isinstance(self.event_count, bool)
            or not isinstance(self.event_count, int)
            or self.event_count != len(event_receipts)
        ):
            raise ValueError("event_count must equal the exact event replay roster")
        if {value.patient_id for value in event_receipts} != set(roster):
            raise ValueError(
                "Evolution event receipts do not exactly cover the fit-patient roster"
            )
        expected_event_hashes = _event_roster_hashes(event_receipts)
        for field, expected in expected_event_hashes.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} does not match the exact event replay roster")
        if self.descriptor_schema_sha256 != EVOLUTION_DESCRIPTOR_SCHEMA_SHA256:
            raise ValueError("Descriptor schema SHA is not the frozen computed schema")
        if (
            self.complete19_descriptor_mask_sha256
            != COMPLETE19_DESCRIPTOR_MASK_SHA256
        ):
            raise ValueError("Artifact complete19 descriptor mask SHA drifted")
        if self.schema_version != COMPUTED_EVOLUTION_SCALER_RECEIPT_SCHEMA:
            raise ValueError("Unsupported computed-evolution scaler receipt schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ComputedEvolutionScalerArtifact:
    """Portable, path-free scaler plus its verified selection receipt."""

    scaler: PatientBalancedRobustScaler
    receipt: ComputedEvolutionScalerArtifactReceipt
    computation_receipt: EvolutionComputationReceipt
    verification_receipt: VerifiedEvolutionFitReceipt
    schema_version: str = COMPUTED_EVOLUTION_SCALER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.scaler, PatientBalancedRobustScaler):
            raise TypeError("scaler must be a PatientBalancedRobustScaler")
        if not isinstance(self.receipt, ComputedEvolutionScalerArtifactReceipt):
            raise TypeError("receipt must be a ComputedEvolutionScalerArtifactReceipt")
        if not isinstance(self.computation_receipt, EvolutionComputationReceipt):
            raise TypeError("artifact requires EvolutionComputationReceipt")
        if not isinstance(self.verification_receipt, VerifiedEvolutionFitReceipt):
            raise TypeError("artifact requires VerifiedEvolutionFitReceipt")
        if self.schema_version != COMPUTED_EVOLUTION_SCALER_ARTIFACT_SCHEMA:
            raise ValueError("Unsupported computed-evolution scaler artifact schema")
        for name, tensor in (
            ("center", self.scaler.center),
            ("iqr", self.scaler.iqr),
            ("scale", self.scaler.scale),
        ):
            if tensor.dtype != torch.float64 or tensor.device.type != "cpu":
                raise TypeError(f"Artifact scaler {name} must be CPU float64")
        scaler_receipt_sha = _canonical_sha256(asdict(self.scaler.receipt))
        if self.receipt.scaler_receipt_sha256 != scaler_receipt_sha:
            raise ValueError("Scaler receipt SHA disagrees with artifact receipt")
        if (
            self.receipt.computation_receipt_sha256
            != self.computation_receipt.receipt_sha256
        ):
            raise ValueError("Computation receipt SHA disagrees with artifact receipt")
        if (
            self.scaler.receipt.patient_roster_sha256
            != self.receipt.fit_patient_roster_sha256
            or self.scaler.receipt.patient_count != self.receipt.patient_count
        ):
            raise ValueError("Scaler fit roster disagrees with artifact receipt")
        if (
            self.scaler.receipt.split_manifest_sha256
            != self.receipt.split_manifest_sha256
        ):
            raise ValueError("Scaler split manifest disagrees with artifact receipt")
        persisted_fit_result = ComputedEvolutionFitResult(
            scaler=self.scaler,
            event_receipts=self.receipt.fit_event_receipts,
            computation_receipt=self.computation_receipt,
        )
        _validate_verified_receipt_against_fit_result(
            persisted_fit_result, self.verification_receipt
        )
        if (
            self.verification_receipt.canonical_artifact_core_sha256
            != _canonical_sha256(self.canonical_core_payload)
        ):
            raise ValueError("Verification receipt does not bind canonical artifact core")

    @property
    def canonical_core_payload(self) -> dict[str, object]:
        return _artifact_core_payload(
            self.scaler,
            self.receipt,
            self.computation_receipt,
            schema_version=self.schema_version,
        )

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            **self.canonical_core_payload,
            "verification_receipt": asdict(self.verification_receipt),
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.canonical_payload)).hexdigest()


@dataclass(frozen=True)
class SavedComputedEvolutionScalerArtifact:
    path: Path
    artifact_sha256: str
    artifact_receipt_sha256: str
    scaler_receipt_sha256: str
    verification_receipt_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "artifact_sha256",
            "artifact_receipt_sha256",
            "scaler_receipt_sha256",
            "verification_receipt_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )


def _artifact_core_payload(
    scaler: PatientBalancedRobustScaler,
    receipt: ComputedEvolutionScalerArtifactReceipt,
    computation_receipt: EvolutionComputationReceipt,
    *,
    schema_version: str = COMPUTED_EVOLUTION_SCALER_ARTIFACT_SCHEMA,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "serialization": "canonical_json_utf8_newline_no_pickle",
        "artifact_receipt_sha256": receipt.receipt_sha256,
        "artifact_receipt": asdict(receipt),
        "scaler_receipt": asdict(scaler.receipt),
        "computation_receipt": asdict(computation_receipt),
    }


def _build_computed_evolution_scaler_artifact_core_from_manifest(
    fit_result: ComputedEvolutionFitResult,
    *,
    oof_fold: int | None,
    fit_public_patient_keys: Sequence[object],
    fit_manifest: TUSZIctalTrainingManifest,
    fit_manifest_bundle_sha256: str,
    oof_protocol: IctalConceptOOFProtocol,
) -> tuple[
    PatientBalancedRobustScaler,
    ComputedEvolutionScalerArtifactReceipt,
    EvolutionComputationReceipt,
    dict[str, object],
]:
    """Build the canonical artifact core before independent verification."""

    if not isinstance(fit_result, ComputedEvolutionFitResult):
        raise TypeError("formal artifact core requires ComputedEvolutionFitResult")
    scaler = fit_result.scaler
    event_receipts = fit_result.event_receipts
    computation_receipt = fit_result.computation_receipt
    if not isinstance(fit_manifest, TUSZIctalTrainingManifest):
        raise TypeError("fit_manifest must be a TUSZIctalTrainingManifest")
    bundle_sha = _require_sha256(
        fit_manifest_bundle_sha256, field="fit_manifest_bundle_sha256"
    )
    fold = _normalize_oof_fold(oof_fold, field="oof_fold")
    plan = _protocol_plan(oof_protocol, fold)
    roster = _normalize_fit_roster(fit_public_patient_keys)

    if not fit_manifest.preflight_performed:
        raise ValueError("Formal evolution scaler fit manifest must be preflighted")
    if roster != fit_manifest.patient_ids:
        raise ValueError(
            "Exact fit public-patient roster must equal the preflighted manifest roster"
        )
    if scaler.receipt.patient_roster_sha256 != patient_roster_sha256(roster):
        raise ValueError("Scaler receipt was fitted on a different public roster")
    if scaler.receipt.patient_count != len(roster):
        raise ValueError("Scaler patient_count disagrees with the exact fit roster")
    if (
        scaler.receipt.split_manifest_sha256
        != oof_protocol.receipt.split_manifest_sha256
    ):
        raise ValueError("Scaler and OOF protocol use different split manifests")
    if (
        fit_manifest.cohort_receipt.receipt_sha256
        != plan.training_cohort.receipt.receipt_sha256
    ):
        raise ValueError("Fit manifest does not bind the selected OOF training cohort")
    if (
        fit_manifest.authorized_source_record_sha256s
        != plan.receipt.authorized_record_sha256s
    ):
        raise ValueError(
            "Fit manifest authorized source roster disagrees with OOF plan"
        )
    authorized_roster_sha = canonical_public_roster_sha256(
        fit_manifest.authorized_source_record_sha256s
    )
    if authorized_roster_sha != plan.receipt.authorized_record_roster_sha256:
        raise ValueError("Authorized source-record roster SHA disagrees with OOF plan")
    leaked = tuple(sorted(set(roster) & set(plan.held_out_public_patient_keys)))
    if leaked:
        raise ValueError(
            f"Evolution scaler fit roster contains held-out patients: {leaked}"
        )

    expected_events = tuple(
        sorted(fit_manifest.events, key=lambda value: (value.patient_id, value.event_id))
    )
    if tuple(value.event_id for value in event_receipts) != tuple(
        value.event_id for value in expected_events
    ):
        raise ValueError(
            "Evolution replay receipts do not exactly match the fit-manifest event roster"
        )
    for event, replay in zip(expected_events, event_receipts):
        checks = {
            "patient_id": replay.patient_id == event.patient_id,
            "event_record_sha256": (
                replay.event_record_sha256 == event.event_record_sha256
            ),
            "edf_sha256": replay.edf_sha256 == event.edf_sha256,
            "signal_content_sha256": (
                replay.signal_content_sha256 == event.signal_content_sha256
            ),
            "signal_preflight_receipt_sha256": (
                replay.signal_preflight_receipt_sha256
                == event.signal_preflight_receipt_sha256
            ),
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Evolution replay {event.event_id} disagrees with manifest fields "
                f"{failed}"
            )
    event_hashes = _event_roster_hashes(event_receipts)

    receipt = ComputedEvolutionScalerArtifactReceipt(
        oof_fold=fold,
        fit_public_patient_keys=roster,
        fit_patient_roster_sha256=patient_roster_sha256(roster),
        patient_count=len(roster),
        fit_event_receipts=event_receipts,
        fit_event_roster_sha256=event_hashes["fit_event_roster_sha256"],
        patient_event_roster_sha256=event_hashes[
            "patient_event_roster_sha256"
        ],
        edf_roster_sha256=event_hashes["edf_roster_sha256"],
        signal_roster_sha256=event_hashes["signal_roster_sha256"],
        raw_descriptor_roster_sha256=event_hashes[
            "raw_descriptor_roster_sha256"
        ],
        descriptor_mask_roster_sha256=event_hashes[
            "descriptor_mask_roster_sha256"
        ],
        complete19_descriptor_mask_sha256=(
            COMPLETE19_DESCRIPTOR_MASK_SHA256
        ),
        event_count=len(event_receipts),
        fit_manifest_source_sha256=fit_manifest.manifest_sha256,
        fit_manifest_bundle_sha256=bundle_sha,
        split_manifest_sha256=oof_protocol.receipt.split_manifest_sha256,
        oof_protocol_receipt_sha256=oof_protocol.receipt.receipt_sha256,
        oof_plan_receipt_sha256=plan.receipt.receipt_sha256,
        held_out_target_roster_sha256=(
            plan.receipt.held_out_target_roster_sha256
        ),
        held_out_public_roster_sha256=(
            plan.receipt.held_out_public_roster_sha256
        ),
        authorized_source_record_roster_sha256=authorized_roster_sha,
        descriptor_schema_sha256=EVOLUTION_DESCRIPTOR_SCHEMA_SHA256,
        preprocess_schema_sha256=evolution_preprocess_schema_sha256(
            fit_manifest.preprocess_config
        ),
        computation_receipt_sha256=computation_receipt.receipt_sha256,
        scaler_receipt_sha256=_canonical_sha256(asdict(scaler.receipt)),
    )
    core = _artifact_core_payload(scaler, receipt, computation_receipt)
    return scaler, receipt, computation_receipt, core


def _reject_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    return absolute


def _read_stable_file(
    path: Path,
    *,
    field: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    source = _reject_symlink_components(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular file")
    before = source.stat()
    if before.st_size < 1 or before.st_size > max_bytes:
        raise ValueError(f"{field} has an invalid size")
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
        raise RuntimeError(f"{field} changed while it was read")
    return payload, hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_computed_evolution_scaler_artifact(
    artifact: ComputedEvolutionScalerArtifact,
    bundle_directory: str | Path,
) -> SavedComputedEvolutionScalerArtifact:
    """Atomically publish a new one-file canonical JSON bundle."""

    if not isinstance(artifact, ComputedEvolutionScalerArtifact):
        raise TypeError("artifact must be a ComputedEvolutionScalerArtifact")
    encoded = _canonical_json_bytes(artifact.canonical_payload)
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Evolution scaler artifact exceeds the closed size limit")
    # Exercise strict reconstruction before publishing any bytes.
    reconstructed = _artifact_from_payload(
        _parse_strict_canonical_json(encoded, field="evolution scaler artifact")
    )
    if _canonical_json_bytes(reconstructed.canonical_payload) != encoded:
        raise ValueError("Evolution scaler artifact is unstable under reconstruction")

    bundle = _reject_symlink_components(
        Path(bundle_directory), field="evolution scaler bundle"
    )
    if bundle.name in {"", ".", ".."}:
        raise ValueError("Evolution scaler bundle requires a concrete directory name")
    if os.path.lexists(bundle):
        raise FileExistsError("Evolution scaler bundle destination already exists")
    parent = _reject_symlink_components(bundle.parent, field="bundle parent")
    if not parent.is_dir():
        raise FileNotFoundError("Evolution scaler bundle parent does not exist")

    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.tmp-", dir=parent))
    artifact_file = temporary / COMPUTED_EVOLUTION_SCALER_ARTIFACT_FILENAME
    published = False
    try:
        with artifact_file.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        if os.path.lexists(bundle):
            raise FileExistsError("Evolution scaler bundle destination already exists")
        os.rename(temporary, bundle)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return SavedComputedEvolutionScalerArtifact(
        path=bundle,
        artifact_sha256=hashlib.sha256(encoded).hexdigest(),
        artifact_receipt_sha256=artifact.receipt.receipt_sha256,
        scaler_receipt_sha256=artifact.receipt.scaler_receipt_sha256,
        verification_receipt_sha256=(
            artifact.verification_receipt.receipt_sha256
        ),
    )


def _reject_duplicate_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _parse_strict_canonical_json(payload: bytes, *, field: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{field} must be a JSON object")
    if _canonical_json_bytes(decoded) != payload:
        raise ValueError(f"{field} bytes are not canonical JSON")
    return decoded


def _closed_object(
    value: object,
    *,
    fields: frozenset[str],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{field} violates the closed schema; missing={missing}, unknown={unknown}"
        )
    return value


def _json_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a JSON string")
    return value


def _json_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON string array")
    return tuple(value)


def _json_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _json_float_tuple(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON number array")
    return tuple(
        _json_float(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def _json_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a JSON integer")
    return value


def _scaler_receipt_from_payload(value: object) -> PatientBalancedScalerReceipt:
    payload = _closed_object(
        value, fields=_SCALER_RECEIPT_FIELDS, field="scaler_receipt"
    )
    return PatientBalancedScalerReceipt(
        feature_names=_json_string_tuple(
            payload["feature_names"], field="scaler_receipt.feature_names"
        ),
        feature_schema_sha256=_require_sha256(
            payload["feature_schema_sha256"],
            field="scaler_receipt.feature_schema_sha256",
        ),
        patient_roster_sha256=_require_sha256(
            payload["patient_roster_sha256"],
            field="scaler_receipt.patient_roster_sha256",
        ),
        split_manifest_sha256=_require_sha256(
            payload["split_manifest_sha256"],
            field="scaler_receipt.split_manifest_sha256",
        ),
        fit_split_sha256=_require_sha256(
            payload["fit_split_sha256"], field="scaler_receipt.fit_split_sha256"
        ),
        fit_split=_json_string(
            payload["fit_split"], field="scaler_receipt.fit_split"
        ),
        patient_count=_json_int(
            payload["patient_count"], field="scaler_receipt.patient_count"
        ),
        patient_feature_medians_sha256=_require_sha256(
            payload["patient_feature_medians_sha256"],
            field="scaler_receipt.patient_feature_medians_sha256",
        ),
        center=_json_float_tuple(
            payload["center"], field="scaler_receipt.center"
        ),
        iqr=_json_float_tuple(payload["iqr"], field="scaler_receipt.iqr"),
        scale=_json_float_tuple(payload["scale"], field="scaler_receipt.scale"),
        clip=_json_float(payload["clip"], field="scaler_receipt.clip"),
        statistic_scope=_json_string(
            payload["statistic_scope"], field="scaler_receipt.statistic_scope"
        ),
        patient_balance_policy=_json_string(
            payload["patient_balance_policy"],
            field="scaler_receipt.patient_balance_policy",
        ),
        zero_iqr_policy=_json_string(
            payload["zero_iqr_policy"], field="scaler_receipt.zero_iqr_policy"
        ),
        schema_version=_json_string(
            payload["schema_version"], field="scaler_receipt.schema_version"
        ),
    )


def _event_receipt_from_payload(
    value: object, *, index: int
) -> ComputedEvolutionEventReceipt:
    field = f"artifact_receipt.fit_event_receipts[{index}]"
    payload = _closed_object(value, fields=_EVENT_RECEIPT_FIELDS, field=field)
    return ComputedEvolutionEventReceipt(
        event_id=_json_string(payload["event_id"], field=f"{field}.event_id"),
        patient_id=_json_string(
            payload["patient_id"], field=f"{field}.patient_id"
        ),
        event_record_sha256=_require_sha256(
            payload["event_record_sha256"],
            field=f"{field}.event_record_sha256",
        ),
        edf_sha256=_require_sha256(
            payload["edf_sha256"], field=f"{field}.edf_sha256"
        ),
        signal_content_sha256=_require_sha256(
            payload["signal_content_sha256"],
            field=f"{field}.signal_content_sha256",
        ),
        signal_preflight_receipt_sha256=_require_sha256(
            payload["signal_preflight_receipt_sha256"],
            field=f"{field}.signal_preflight_receipt_sha256",
        ),
        signal_window_sha256=_require_sha256(
            payload["signal_window_sha256"],
            field=f"{field}.signal_window_sha256",
        ),
        raw_descriptor_sha256=_require_sha256(
            payload["raw_descriptor_sha256"],
            field=f"{field}.raw_descriptor_sha256",
        ),
        descriptor_mask_sha256=_require_sha256(
            payload["descriptor_mask_sha256"],
            field=f"{field}.descriptor_mask_sha256",
        ),
        schema_version=_json_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
    )


def _computation_receipt_from_payload(
    value: object,
) -> EvolutionComputationReceipt:
    payload = _closed_object(
        value,
        fields=_COMPUTATION_RECEIPT_FIELDS,
        field="computation_receipt",
    )
    return EvolutionComputationReceipt(
        evolution_source_sha256=_require_sha256(
            payload["evolution_source_sha256"],
            field="computation_receipt.evolution_source_sha256",
        ),
        evolution_fit_source_sha256=_require_sha256(
            payload["evolution_fit_source_sha256"],
            field="computation_receipt.evolution_fit_source_sha256",
        ),
        geometry_source_sha256=_require_sha256(
            payload["geometry_source_sha256"],
            field="computation_receipt.geometry_source_sha256",
        ),
        torch_version=_json_string(
            payload["torch_version"], field="computation_receipt.torch_version"
        ),
        numpy_version=_json_string(
            payload["numpy_version"], field="computation_receipt.numpy_version"
        ),
        scipy_version=_json_string(
            payload["scipy_version"], field="computation_receipt.scipy_version"
        ),
        platform_machine=_json_string(
            payload["platform_machine"],
            field="computation_receipt.platform_machine",
        ),
        torch_num_threads=_json_int(
            payload["torch_num_threads"],
            field="computation_receipt.torch_num_threads",
        ),
        torch_num_interop_threads=_json_int(
            payload["torch_num_interop_threads"],
            field="computation_receipt.torch_num_interop_threads",
        ),
        compute_device_policy=_json_string(
            payload["compute_device_policy"],
            field="computation_receipt.compute_device_policy",
        ),
        compute_dtype=_json_string(
            payload["compute_dtype"], field="computation_receipt.compute_dtype"
        ),
        descriptor_output_dtype=_json_string(
            payload["descriptor_output_dtype"],
            field="computation_receipt.descriptor_output_dtype",
        ),
        scaler_dtype=_json_string(
            payload["scaler_dtype"], field="computation_receipt.scaler_dtype"
        ),
        mask_policy=_json_string(
            payload["mask_policy"], field="computation_receipt.mask_policy"
        ),
        complete19_descriptor_mask_sha256=_require_sha256(
            payload["complete19_descriptor_mask_sha256"],
            field="computation_receipt.complete19_descriptor_mask_sha256",
        ),
        schema_version=_json_string(
            payload["schema_version"], field="computation_receipt.schema_version"
        ),
    )


def _verification_receipt_from_payload(
    value: object,
) -> VerifiedEvolutionFitReceipt:
    payload = _closed_object(
        value,
        fields=_VERIFICATION_RECEIPT_FIELDS,
        field="verification_receipt",
    )
    return VerifiedEvolutionFitReceipt(
        candidate_fit_result_sha256=_require_sha256(
            payload["candidate_fit_result_sha256"],
            field="verification_receipt.candidate_fit_result_sha256",
        ),
        independent_fit_result_sha256=_require_sha256(
            payload["independent_fit_result_sha256"],
            field="verification_receipt.independent_fit_result_sha256",
        ),
        canonical_artifact_core_sha256=_require_sha256(
            payload["canonical_artifact_core_sha256"],
            field="verification_receipt.canonical_artifact_core_sha256",
        ),
        event_receipt_roster_sha256=_require_sha256(
            payload["event_receipt_roster_sha256"],
            field="verification_receipt.event_receipt_roster_sha256",
        ),
        scaler_receipt_sha256=_require_sha256(
            payload["scaler_receipt_sha256"],
            field="verification_receipt.scaler_receipt_sha256",
        ),
        computation_receipt_sha256=_require_sha256(
            payload["computation_receipt_sha256"],
            field="verification_receipt.computation_receipt_sha256",
        ),
        event_count=_json_int(
            payload["event_count"], field="verification_receipt.event_count"
        ),
        patient_count=_json_int(
            payload["patient_count"], field="verification_receipt.patient_count"
        ),
        raw_edf_replay_count=_json_int(
            payload["raw_edf_replay_count"],
            field="verification_receipt.raw_edf_replay_count",
        ),
        verification_policy=_json_string(
            payload["verification_policy"],
            field="verification_receipt.verification_policy",
        ),
        schema_version=_json_string(
            payload["schema_version"], field="verification_receipt.schema_version"
        ),
    )


def _artifact_receipt_from_payload(
    value: object,
) -> ComputedEvolutionScalerArtifactReceipt:
    payload = _closed_object(
        value,
        fields=_ARTIFACT_RECEIPT_FIELDS,
        field="artifact_receipt",
    )
    raw_event_receipts = payload["fit_event_receipts"]
    if not isinstance(raw_event_receipts, list):
        raise ValueError("artifact_receipt.fit_event_receipts must be an array")
    return ComputedEvolutionScalerArtifactReceipt(
        oof_fold=_normalize_oof_fold(payload["oof_fold"], field="oof_fold"),
        fit_public_patient_keys=_json_string_tuple(
            payload["fit_public_patient_keys"],
            field="artifact_receipt.fit_public_patient_keys",
        ),
        fit_patient_roster_sha256=_require_sha256(
            payload["fit_patient_roster_sha256"],
            field="artifact_receipt.fit_patient_roster_sha256",
        ),
        patient_count=_json_int(
            payload["patient_count"], field="artifact_receipt.patient_count"
        ),
        fit_event_receipts=tuple(
            _event_receipt_from_payload(value, index=index)
            for index, value in enumerate(raw_event_receipts)
        ),
        fit_event_roster_sha256=_require_sha256(
            payload["fit_event_roster_sha256"],
            field="artifact_receipt.fit_event_roster_sha256",
        ),
        patient_event_roster_sha256=_require_sha256(
            payload["patient_event_roster_sha256"],
            field="artifact_receipt.patient_event_roster_sha256",
        ),
        edf_roster_sha256=_require_sha256(
            payload["edf_roster_sha256"],
            field="artifact_receipt.edf_roster_sha256",
        ),
        signal_roster_sha256=_require_sha256(
            payload["signal_roster_sha256"],
            field="artifact_receipt.signal_roster_sha256",
        ),
        raw_descriptor_roster_sha256=_require_sha256(
            payload["raw_descriptor_roster_sha256"],
            field="artifact_receipt.raw_descriptor_roster_sha256",
        ),
        descriptor_mask_roster_sha256=_require_sha256(
            payload["descriptor_mask_roster_sha256"],
            field="artifact_receipt.descriptor_mask_roster_sha256",
        ),
        complete19_descriptor_mask_sha256=_require_sha256(
            payload["complete19_descriptor_mask_sha256"],
            field="artifact_receipt.complete19_descriptor_mask_sha256",
        ),
        event_count=_json_int(
            payload["event_count"], field="artifact_receipt.event_count"
        ),
        fit_manifest_source_sha256=_require_sha256(
            payload["fit_manifest_source_sha256"],
            field="artifact_receipt.fit_manifest_source_sha256",
        ),
        fit_manifest_bundle_sha256=_require_sha256(
            payload["fit_manifest_bundle_sha256"],
            field="artifact_receipt.fit_manifest_bundle_sha256",
        ),
        split_manifest_sha256=_require_sha256(
            payload["split_manifest_sha256"],
            field="artifact_receipt.split_manifest_sha256",
        ),
        oof_protocol_receipt_sha256=_require_sha256(
            payload["oof_protocol_receipt_sha256"],
            field="artifact_receipt.oof_protocol_receipt_sha256",
        ),
        oof_plan_receipt_sha256=_require_sha256(
            payload["oof_plan_receipt_sha256"],
            field="artifact_receipt.oof_plan_receipt_sha256",
        ),
        held_out_target_roster_sha256=_require_sha256(
            payload["held_out_target_roster_sha256"],
            field="artifact_receipt.held_out_target_roster_sha256",
        ),
        held_out_public_roster_sha256=_require_sha256(
            payload["held_out_public_roster_sha256"],
            field="artifact_receipt.held_out_public_roster_sha256",
        ),
        authorized_source_record_roster_sha256=_require_sha256(
            payload["authorized_source_record_roster_sha256"],
            field="artifact_receipt.authorized_source_record_roster_sha256",
        ),
        descriptor_schema_sha256=_require_sha256(
            payload["descriptor_schema_sha256"],
            field="artifact_receipt.descriptor_schema_sha256",
        ),
        preprocess_schema_sha256=_require_sha256(
            payload["preprocess_schema_sha256"],
            field="artifact_receipt.preprocess_schema_sha256",
        ),
        computation_receipt_sha256=_require_sha256(
            payload["computation_receipt_sha256"],
            field="artifact_receipt.computation_receipt_sha256",
        ),
        scaler_receipt_sha256=_require_sha256(
            payload["scaler_receipt_sha256"],
            field="artifact_receipt.scaler_receipt_sha256",
        ),
        schema_version=_json_string(
            payload["schema_version"], field="artifact_receipt.schema_version"
        ),
    )


def _artifact_from_payload(
    value: Mapping[str, object],
) -> ComputedEvolutionScalerArtifact:
    payload = _closed_object(value, fields=_ARTIFACT_FIELDS, field="artifact")
    if (
        _json_string(payload["schema_version"], field="schema_version")
        != COMPUTED_EVOLUTION_SCALER_ARTIFACT_SCHEMA
    ):
        raise ValueError("Unsupported computed-evolution scaler artifact schema")
    if payload["serialization"] != "canonical_json_utf8_newline_no_pickle":
        raise ValueError("Evolution scaler artifact must use safe canonical JSON")
    receipt = _artifact_receipt_from_payload(payload["artifact_receipt"])
    declared_receipt_sha = _require_sha256(
        payload["artifact_receipt_sha256"], field="artifact_receipt_sha256"
    )
    if declared_receipt_sha != receipt.receipt_sha256:
        raise ValueError("Artifact receipt SHA does not match its payload")
    scaler_receipt = _scaler_receipt_from_payload(payload["scaler_receipt"])
    computation_receipt = _computation_receipt_from_payload(
        payload["computation_receipt"]
    )
    verification_receipt = _verification_receipt_from_payload(
        payload["verification_receipt"]
    )
    scaler = PatientBalancedRobustScaler(
        center=torch.tensor(scaler_receipt.center, dtype=torch.float64),
        iqr=torch.tensor(scaler_receipt.iqr, dtype=torch.float64),
        scale=torch.tensor(scaler_receipt.scale, dtype=torch.float64),
        clip=scaler_receipt.clip,
        receipt=scaler_receipt,
    )
    artifact = ComputedEvolutionScalerArtifact(
        scaler=scaler,
        receipt=receipt,
        computation_receipt=computation_receipt,
        verification_receipt=verification_receipt,
        schema_version=_json_string(payload["schema_version"], field="schema_version"),
    )
    if _canonical_json_bytes(artifact.canonical_payload) != _canonical_json_bytes(
        payload
    ):
        raise ValueError("Artifact payload is not stable under typed reconstruction")
    return artifact


def _load_bound_fit_manifest(
    bundle_directory: str | Path,
) -> tuple[TUSZIctalTrainingManifest, str]:
    bundle = _reject_symlink_components(
        Path(bundle_directory), field="fit manifest bundle"
    )
    _, bundle_sha = _read_stable_file(
        bundle / "manifest.json",
        field="fit manifest envelope",
        max_bytes=_MAX_MANIFEST_ENVELOPE_BYTES,
    )
    _, source_sha = _read_stable_file(
        bundle / "receipt.json",
        field="fit manifest receipt",
        max_bytes=128 * 1024 * 1024,
    )
    manifest = load_tusz_ictal_training_manifest(
        bundle,
        expected_bundle_manifest_sha256=bundle_sha,
        expected_source_manifest_sha256=source_sha,
    )
    return manifest, bundle_sha


def build_computed_evolution_scaler_artifact(
    verified_fit_result: VerifiedComputedEvolutionFitResult,
    *,
    oof_fold: int | None,
    fit_manifest_bundle_directory: str | Path,
    oof_protocol: IctalConceptOOFProtocol,
) -> ComputedEvolutionScalerArtifact:
    """Build a formal artifact only from an independently verified full fit."""

    if not isinstance(verified_fit_result, VerifiedComputedEvolutionFitResult):
        raise TypeError(
            "formal builder requires VerifiedComputedEvolutionFitResult"
        )

    fit_manifest, fit_bundle_sha = _load_bound_fit_manifest(
        fit_manifest_bundle_directory
    )
    scaler, receipt, computation_receipt, core = (
        _build_computed_evolution_scaler_artifact_core_from_manifest(
            verified_fit_result.fit_result,
            oof_fold=oof_fold,
            fit_public_patient_keys=fit_manifest.patient_ids,
            fit_manifest=fit_manifest,
            fit_manifest_bundle_sha256=fit_bundle_sha,
            oof_protocol=oof_protocol,
        )
    )
    verification_receipt = verified_fit_result.verification_receipt
    if _canonical_sha256(core) != (
        verification_receipt.canonical_artifact_core_sha256
    ):
        raise ValueError(
            "Verified replay receipt belongs to a different canonical artifact core"
        )
    return ComputedEvolutionScalerArtifact(
        scaler=scaler,
        receipt=receipt,
        computation_receipt=computation_receipt,
        verification_receipt=verification_receipt,
    )


def load_computed_evolution_scaler_artifact(
    bundle_directory: str | Path,
    *,
    oof_fold: int | None,
    fit_manifest_bundle_directory: str | Path,
    oof_protocol: IctalConceptOOFProtocol,
    expected_fit_public_patient_keys: Sequence[object],
    expected_artifact_sha256: str | None = None,
    expected_scaler: PatientBalancedRobustScaler | None = None,
) -> ComputedEvolutionScalerArtifact:
    """Strictly load against upstream lineage and a numeric trust anchor.

    ``expected_artifact_sha256`` is mandatory because neither protocol/manifest
    reconstruction nor an independently held six-number scaler can authenticate
    the per-event signal/descriptor replay hashes. ``expected_scaler`` adds an
    optional independent numeric check.
    """

    if expected_artifact_sha256 is None:
        raise ValueError(
            "Strict scaler load requires expected_artifact_sha256"
        )
    expected_sha = (
        None
        if expected_artifact_sha256 is None
        else _require_sha256(
            expected_artifact_sha256, field="expected_artifact_sha256"
        )
    )
    if expected_scaler is not None and not isinstance(
        expected_scaler, PatientBalancedRobustScaler
    ):
        raise TypeError("expected_scaler must be a PatientBalancedRobustScaler")
    fold = _normalize_oof_fold(oof_fold, field="oof_fold")
    bundle = _reject_symlink_components(
        Path(bundle_directory), field="evolution scaler bundle"
    )
    if not bundle.is_dir():
        raise FileNotFoundError("Evolution scaler bundle directory does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda item: item.name))
    if (
        len(entries) != 1
        or entries[0].name != COMPUTED_EVOLUTION_SCALER_ARTIFACT_FILENAME
    ):
        raise ValueError("Evolution scaler bundle has missing or unknown entries")
    encoded, artifact_sha = _read_stable_file(
        entries[0], field="evolution scaler artifact", max_bytes=_MAX_ARTIFACT_BYTES
    )
    if expected_sha is not None and artifact_sha != expected_sha:
        raise ValueError("Evolution scaler artifact SHA does not match expected SHA")
    loaded = _artifact_from_payload(
        _parse_strict_canonical_json(encoded, field="evolution scaler artifact")
    )
    if loaded.receipt.oof_fold != fold:
        raise ValueError("Evolution scaler artifact belongs to a different OOF fold")

    fit_manifest, fit_bundle_sha = _load_bound_fit_manifest(
        fit_manifest_bundle_directory
    )
    fit_result = ComputedEvolutionFitResult(
        scaler=loaded.scaler,
        event_receipts=loaded.receipt.fit_event_receipts,
        computation_receipt=loaded.computation_receipt,
    )
    _validate_verified_receipt_against_fit_result(
        fit_result,
        loaded.verification_receipt,
    )
    scaler, receipt, computation_receipt, core = (
        _build_computed_evolution_scaler_artifact_core_from_manifest(
            fit_result,
            oof_fold=fold,
            fit_public_patient_keys=expected_fit_public_patient_keys,
            fit_manifest=fit_manifest,
            fit_manifest_bundle_sha256=fit_bundle_sha,
            oof_protocol=oof_protocol,
        )
    )
    rebuilt = ComputedEvolutionScalerArtifact(
        scaler=scaler,
        receipt=receipt,
        computation_receipt=computation_receipt,
        verification_receipt=loaded.verification_receipt,
    )
    if _canonical_sha256(core) != (
        loaded.verification_receipt.canonical_artifact_core_sha256
    ):
        raise ValueError("Loaded verification receipt does not match rebuilt core")
    if rebuilt.canonical_payload != loaded.canonical_payload:
        raise ValueError(
            "Evolution scaler artifact does not exactly match upstream reconstruction"
        )
    if expected_scaler is not None:
        if expected_scaler.receipt != loaded.scaler.receipt or any(
            not torch.equal(expected, actual)
            for expected, actual in (
                (expected_scaler.center, loaded.scaler.center),
                (expected_scaler.iqr, loaded.scaler.iqr),
                (expected_scaler.scale, loaded.scaler.scale),
            )
        ):
            raise ValueError(
                "Evolution scaler statistics differ from independently rebuilt scaler"
            )
    return loaded


def load_externally_pinned_computed_evolution_scaler_artifact(
    bundle_directory: str | Path,
    *,
    oof_fold: int | None,
    expected_artifact_sha256: str,
) -> ComputedEvolutionScalerArtifact:
    """Load a closed scaler bundle using an external byte-level trust anchor.

    This deliberately narrower loader is for target-free downstream evidence
    production.  It validates the canonical JSON schema, every persisted fit
    and independent-replay receipt, the requested fold, and a mandatory
    externally pinned artifact SHA256.  Unlike
    :func:`load_computed_evolution_scaler_artifact`, it does not reopen the OOF
    protocol or a DeepSOZ-derived registry.  Consequently it is suitable for a
    development evidence producer that is forbidden from reading SOZ target
    vectors, but it is *not* a replacement for the upstream formal
    reconstruction loader when fitting or publishing a new scaler.
    """

    fold = _normalize_oof_fold(oof_fold, field="oof_fold")
    expected_sha = _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    )
    bundle = _reject_symlink_components(
        Path(bundle_directory), field="evolution scaler bundle"
    )
    if not bundle.is_dir():
        raise FileNotFoundError("Evolution scaler bundle directory does not exist")
    entries = tuple(sorted(bundle.iterdir(), key=lambda item: item.name))
    if (
        len(entries) != 1
        or entries[0].name != COMPUTED_EVOLUTION_SCALER_ARTIFACT_FILENAME
    ):
        raise ValueError("Evolution scaler bundle has missing or unknown entries")
    encoded, artifact_sha = _read_stable_file(
        entries[0], field="evolution scaler artifact", max_bytes=_MAX_ARTIFACT_BYTES
    )
    if artifact_sha != expected_sha:
        raise ValueError("Evolution scaler artifact SHA does not match expected SHA")
    loaded = _artifact_from_payload(
        _parse_strict_canonical_json(encoded, field="evolution scaler artifact")
    )
    if loaded.receipt.oof_fold != fold:
        raise ValueError("Evolution scaler artifact belongs to a different OOF fold")
    if loaded.artifact_sha256 != artifact_sha:
        raise ValueError("Evolution scaler canonical reconstruction changed artifact SHA")
    return loaded


__all__ = [
    "COMPUTED_EVOLUTION_COMPUTATION_RECEIPT_SCHEMA",
    "COMPUTED_EVOLUTION_EVENT_RECEIPT_SCHEMA",
    "COMPUTED_EVOLUTION_SCALER_ARTIFACT_FILENAME",
    "COMPUTED_EVOLUTION_SCALER_ARTIFACT_SCHEMA",
    "COMPUTED_EVOLUTION_SCALER_RECEIPT_SCHEMA",
    "EVOLUTION_DESCRIPTOR_SCHEMA_SHA256",
    "ComputedEvolutionScalerArtifact",
    "ComputedEvolutionScalerArtifactReceipt",
    "ComputedEvolutionFitResult",
    "ComputedEvolutionEventReceipt",
    "EvolutionComputationReceipt",
    "SavedComputedEvolutionScalerArtifact",
    "VerifiedComputedEvolutionFitResult",
    "VerifiedEvolutionFitReceipt",
    "build_computed_evolution_scaler_artifact",
    "evolution_preprocess_schema_sha256",
    "load_computed_evolution_scaler_artifact",
    "load_externally_pinned_computed_evolution_scaler_artifact",
    "save_computed_evolution_scaler_artifact",
]
