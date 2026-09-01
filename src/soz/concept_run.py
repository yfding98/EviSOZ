"""Fixed-epoch, leakage-averse orchestration for cached ictal concepts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile

import numpy as np
import torch

from .cached_concept_training import (
    IctalTokenBagDataset,
    train_cached_ictal_epoch,
)
from .concept_training import DEFAULT_EVENT_MICROBATCH_SIZE
from .data.deepsoz import normalize_patient_id
from .data.provenance import patient_roster_sha256
from .models.concept_heads import IctalInvolvementHead


ICTAL_TRAINING_CONFIG_SCHEMA = "soz_ictal_training_config_v2"
ICTAL_TRAINING_RUN_SCHEMA = "soz_ictal_training_run_v5"
ICTAL_HEAD_STATE_HASH_SCHEMA = "soz_ictal_head_state_hash_v1"
ICTAL_TRAINING_RUN_ARTIFACT_SCHEMA = "soz_ictal_training_run_artifact_v4"
ICTAL_TRAINING_RUN_ARTIFACT_FILENAME = "training_run.json"
ICTAL_IDENTITY_SCALER_SCHEMA = "soz_ictal_identity_scaler_v1"
ICTAL_DETERMINISM_POLICY_SCHEMA = "soz_ictal_determinism_policy_v1"
ICTAL_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_RUN_ARTIFACT_BYTES = 16 * 1024 * 1024


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


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
        raise ValueError("Training-run artifact is not canonical JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _parse_canonical_json(raw: bytes) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Training-run artifact is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Training-run artifact must be a JSON object")
    if _canonical_json_bytes(payload) != raw:
        raise ValueError("Training-run artifact bytes are not canonical JSON")
    return payload


def _require_exact_fields(
    payload: dict[str, object], expected: frozenset[str], *, label: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _public_patient_roster_sha256(patient_ids: tuple[str, ...]) -> str:
    if not patient_ids or tuple(sorted(set(patient_ids))) != patient_ids:
        raise ValueError("Concept-training patient roster must be unique and sorted")
    return _canonical_sha256(patient_ids)


ICTAL_IDENTITY_SCALER_SHA256 = _canonical_sha256(
    {
        "schema_version": ICTAL_IDENTITY_SCALER_SCHEMA,
        "input": "ictal_edge_time_logits",
        "transform": "identity_sigmoid_no_fitted_calibrator",
        "fitted_parameters": 0,
    }
)


def _target_patient_roster(
    values: tuple[str, ...], *, field: str
) -> tuple[str, ...]:
    normalized = tuple(sorted(normalize_patient_id(value) for value in values))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must be a non-empty unique target-patient roster")
    return normalized


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def ictal_head_state_sha256(head: IctalInvolvementHead) -> str:
    """Hash the complete head state with canonical tensor framing.

    The digest includes every parameter and topology buffer name, shape,
    dtype, and exact little-endian bytes.  It is intentionally independent of
    pickle and of filesystem serialization details.
    """

    if not isinstance(head, IctalInvolvementHead):
        raise TypeError("head must be IctalInvolvementHead")
    digest = hashlib.sha256()
    digest.update(ICTAL_HEAD_STATE_HASH_SCHEMA.encode("ascii") + b"\0")
    for name, tensor in sorted(head.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"Ictal head state contains non-finite values: {name}")
        native_big_endian = array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and sys.byteorder == "big"
        )
        if native_big_endian:
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        canonical_dtype = array.dtype.newbyteorder("<").str
        metadata = json.dumps(
            {
                "dtype": canonical_dtype,
                "name": name,
                "shape": list(array.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        raw = array.tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class IctalDeterminismPolicyReceipt:
    """Exact deterministic runtime state used by one CPU or CUDA fit."""

    execution_device_type: str
    required_cublas_workspace_config: str
    observed_cublas_workspace_config: str | None
    deterministic_algorithms_enabled: bool
    deterministic_algorithms_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    schema_version: str = ICTAL_DETERMINISM_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ICTAL_DETERMINISM_POLICY_SCHEMA:
            raise ValueError("Unexpected ictal determinism policy schema")
        if self.execution_device_type not in {"cpu", "cuda"}:
            raise ValueError("execution_device_type must be cpu or cuda")
        if self.required_cublas_workspace_config != ICTAL_CUBLAS_WORKSPACE_CONFIG:
            raise ValueError("CUBLAS workspace policy is not the frozen value")
        if self.execution_device_type == "cuda":
            if self.observed_cublas_workspace_config != (
                ICTAL_CUBLAS_WORKSPACE_CONFIG
            ):
                raise ValueError("CUDA run did not observe the frozen CUBLAS workspace")
        elif self.observed_cublas_workspace_config is not None:
            raise ValueError("CPU determinism receipt must not claim a CUBLAS runtime")
        frozen_flags = {
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        for field, expected in frozen_flags.items():
            value = getattr(self, field)
            if not isinstance(value, bool) or value is not expected:
                raise ValueError(f"{field} is frozen to {expected}")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def validate_ictal_cuda_environment() -> str:
    """Fail before CUDA work unless CUBLAS determinism is externally pinned."""

    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed is None:
        raise RuntimeError(
            "CUDA ictal training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before CUDA initialization"
        )
    if observed != ICTAL_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Conflicting CUBLAS_WORKSPACE_CONFIG; required exact value "
            f"{ICTAL_CUBLAS_WORKSPACE_CONFIG!r}, observed {observed!r}"
        )
    return observed


def _capture_ictal_determinism_policy(
    *, execution_device_type: str
) -> IctalDeterminismPolicyReceipt:
    observed_workspace = (
        validate_ictal_cuda_environment()
        if execution_device_type == "cuda"
        else None
    )
    return IctalDeterminismPolicyReceipt(
        execution_device_type=execution_device_type,
        required_cublas_workspace_config=ICTAL_CUBLAS_WORKSPACE_CONFIG,
        observed_cublas_workspace_config=observed_workspace,
        deterministic_algorithms_enabled=(
            torch.are_deterministic_algorithms_enabled()
        ),
        deterministic_algorithms_warn_only=(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
    )


@contextmanager
def ictal_determinism_runtime(
    config: "IctalTrainingConfig", *, execution_device_type: str
):
    """Apply, verify, and exactly restore the frozen deterministic runtime."""

    if not isinstance(config, IctalTrainingConfig):
        raise TypeError("config must be IctalTrainingConfig")
    if execution_device_type not in {"cpu", "cuda"}:
        raise ValueError("execution_device_type must be cpu or cuda")
    if execution_device_type == "cuda":
        validate_ictal_cuda_environment()
    prior = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    try:
        torch.use_deterministic_algorithms(
            config.deterministic_algorithms,
            warn_only=config.deterministic_warn_only,
        )
        torch.backends.cudnn.deterministic = config.cudnn_deterministic
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = config.cuda_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = config.cudnn_allow_tf32
        receipt = _capture_ictal_determinism_policy(
            execution_device_type=execution_device_type
        )
        yield receipt
    finally:
        torch.backends.cudnn.deterministic = prior["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = prior["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = prior[
            "cuda_matmul_allow_tf32"
        ]
        torch.backends.cudnn.allow_tf32 = prior["cudnn_allow_tf32"]
        torch.use_deterministic_algorithms(
            prior["deterministic_algorithms"],
            warn_only=prior["deterministic_warn_only"],
        )


@dataclass(frozen=True)
class IctalTrainingConfig:
    """Predeclared optimization policy with no target-cohort early stopping."""

    seed: int = 20260808
    fixed_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    event_microbatch_size: int = DEFAULT_EVENT_MICROBATCH_SIZE
    optimizer: str = "adamw"
    loss: str = "unweighted_patient_macro_masked_bce"
    checkpoint_selection: str = "fixed_final_epoch_no_target_validation"
    probability_transform: str = "identity_sigmoid_no_fitted_calibrator"
    foundation_policy: str = "frozen_cached_tokens_no_foundation_optimizer"
    cublas_workspace_config: str = ICTAL_CUBLAS_WORKSPACE_CONFIG
    deterministic_algorithms: bool = True
    deterministic_warn_only: bool = False
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    cuda_matmul_allow_tf32: bool = False
    cudnn_allow_tf32: bool = False
    schema_version: str = ICTAL_TRAINING_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ICTAL_TRAINING_CONFIG_SCHEMA:
            raise ValueError("Unexpected ictal training config schema")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.fixed_epochs, bool)
            or not isinstance(self.fixed_epochs, int)
            or self.fixed_epochs < 1
        ):
            raise ValueError("fixed_epochs must be a positive integer")
        for field in ("learning_rate", "max_grad_norm"):
            source = getattr(self, field)
            if isinstance(source, bool) or not isinstance(source, Real):
                raise TypeError(f"{field} must be numeric")
            value = float(source)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if isinstance(self.weight_decay, bool) or not isinstance(
            self.weight_decay, Real
        ):
            raise TypeError("weight_decay must be numeric")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            isinstance(self.event_microbatch_size, bool)
            or not isinstance(self.event_microbatch_size, int)
            or self.event_microbatch_size < 1
        ):
            raise ValueError("event_microbatch_size must be a positive integer")
        frozen = {
            "optimizer": "adamw",
            "loss": "unweighted_patient_macro_masked_bce",
            "checkpoint_selection": "fixed_final_epoch_no_target_validation",
            "probability_transform": "identity_sigmoid_no_fitted_calibrator",
            "foundation_policy": "frozen_cached_tokens_no_foundation_optimizer",
        }
        for field, expected in frozen.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} is frozen to {expected!r}")
        if self.cublas_workspace_config != ICTAL_CUBLAS_WORKSPACE_CONFIG:
            raise ValueError("cublas_workspace_config is frozen to ':4096:8'")
        deterministic_flags = {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        for field, expected in deterministic_flags.items():
            value = getattr(self, field)
            if not isinstance(value, bool) or value is not expected:
                raise ValueError(f"{field} is frozen to {expected}")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class IctalTrainingEpochReceipt:
    epoch: int
    patient_order: tuple[str, ...]
    patient_order_sha256: str
    mean_patient_loss: float
    n_patients: int
    n_events: int
    n_observed_labels: int

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        order = tuple(str(value).strip() for value in self.patient_order)
        if order != self.patient_order or not order or any(not value for value in order):
            raise ValueError("Epoch patient order must contain trimmed non-empty IDs")
        if len(set(order)) != len(order):
            raise ValueError("Epoch patient order cannot contain duplicates")
        order_sha = _require_sha(
            self.patient_order_sha256, field="patient_order_sha256"
        )
        if order_sha != _canonical_sha256(order):
            raise ValueError("Epoch patient-order SHA mismatch")
        if isinstance(self.mean_patient_loss, bool) or not isinstance(
            self.mean_patient_loss, Real
        ):
            raise TypeError("mean_patient_loss must be numeric")
        if not math.isfinite(self.mean_patient_loss) or self.mean_patient_loss < 0:
            raise ValueError("mean_patient_loss must be finite and non-negative")
        for field in ("n_patients", "n_events", "n_observed_labels"):
            _positive_int(getattr(self, field), field=field)
        if self.n_patients != len(order):
            raise ValueError("Epoch patient count must match its patient order")


@dataclass(frozen=True)
class IctalTrainingRunReceipt:
    config: IctalTrainingConfig
    config_sha256: str
    determinism_policy: IctalDeterminismPolicyReceipt
    determinism_policy_sha256: str
    split_manifest_sha256: str
    oof_protocol_receipt_sha256: str
    oof_plan_receipt_sha256: str
    oof_fold: int | None
    training_target_patient_ids: tuple[str, ...]
    held_out_target_patient_ids: tuple[str, ...]
    training_target_roster_sha256: str
    held_out_target_roster_sha256: str
    training_manifest_sha256: str
    token_source_manifest_sha256: str
    foundation_feature_receipt_sha256: str
    formal_token_corpus_verified: bool
    formal_token_corpus_index_sha256: str | None
    formal_token_corpus_training_bundle_manifest_sha256: str | None
    formal_token_corpus_event_roster_sha256: str | None
    formal_token_corpus_patient_roster_sha256: str | None
    formal_token_corpus_tensor_roster_sha256: str | None
    initial_head_state_sha256: str
    final_head_state_sha256: str
    concept_training_patient_ids: tuple[str, ...]
    concept_training_patient_roster_sha256: str
    epochs: tuple[IctalTrainingEpochReceipt, ...]
    selected_epoch: int
    schema_version: str = ICTAL_TRAINING_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ICTAL_TRAINING_RUN_SCHEMA:
            raise ValueError("Unexpected ictal training run schema")
        if not isinstance(self.config, IctalTrainingConfig):
            raise TypeError("config must be IctalTrainingConfig")
        if not isinstance(
            self.determinism_policy, IctalDeterminismPolicyReceipt
        ):
            raise TypeError(
                "determinism_policy must be IctalDeterminismPolicyReceipt"
            )
        for field in (
            "config_sha256",
            "determinism_policy_sha256",
            "split_manifest_sha256",
            "oof_protocol_receipt_sha256",
            "oof_plan_receipt_sha256",
            "training_target_roster_sha256",
            "held_out_target_roster_sha256",
            "training_manifest_sha256",
            "token_source_manifest_sha256",
            "foundation_feature_receipt_sha256",
            "initial_head_state_sha256",
            "final_head_state_sha256",
            "concept_training_patient_roster_sha256",
        ):
            _require_sha(getattr(self, field), field=field)
        if self.config_sha256 != self.config.receipt_sha256:
            raise ValueError("config_sha256 does not match the training config")
        if self.determinism_policy_sha256 != (
            self.determinism_policy.receipt_sha256
        ):
            raise ValueError("determinism_policy_sha256 does not match its receipt")
        deterministic_checks = {
            "required_cublas_workspace_config": (
                self.determinism_policy.required_cublas_workspace_config
                == self.config.cublas_workspace_config
            ),
            "deterministic_algorithms": (
                self.determinism_policy.deterministic_algorithms_enabled
                == self.config.deterministic_algorithms
            ),
            "deterministic_warn_only": (
                self.determinism_policy.deterministic_algorithms_warn_only
                == self.config.deterministic_warn_only
            ),
            "cudnn_deterministic": (
                self.determinism_policy.cudnn_deterministic
                == self.config.cudnn_deterministic
            ),
            "cudnn_benchmark": (
                self.determinism_policy.cudnn_benchmark
                == self.config.cudnn_benchmark
            ),
            "cuda_matmul_allow_tf32": (
                self.determinism_policy.cuda_matmul_allow_tf32
                == self.config.cuda_matmul_allow_tf32
            ),
            "cudnn_allow_tf32": (
                self.determinism_policy.cudnn_allow_tf32
                == self.config.cudnn_allow_tf32
            ),
        }
        failed_determinism = tuple(
            field for field, passed in deterministic_checks.items() if not passed
        )
        if failed_determinism:
            raise ValueError(
                "Determinism policy disagrees with training config: "
                f"{failed_determinism}"
            )
        if self.oof_fold is not None and (
            isinstance(self.oof_fold, bool)
            or not isinstance(self.oof_fold, int)
            or self.oof_fold not in range(5)
        ):
            raise ValueError("oof_fold must be None or an integer in [0,4]")
        training_targets = _target_patient_roster(
            self.training_target_patient_ids,
            field="training_target_patient_ids",
        )
        held_out_targets = _target_patient_roster(
            self.held_out_target_patient_ids,
            field="held_out_target_patient_ids",
        )
        if training_targets != self.training_target_patient_ids:
            raise ValueError("training_target_patient_ids must be canonical")
        if held_out_targets != self.held_out_target_patient_ids:
            raise ValueError("held_out_target_patient_ids must be canonical")
        if set(training_targets) & set(held_out_targets):
            raise ValueError("Training and held-out target patients must be disjoint")
        if self.training_target_roster_sha256 != patient_roster_sha256(
            training_targets
        ):
            raise ValueError("training_target_roster_sha256 does not match its roster")
        if self.held_out_target_roster_sha256 != patient_roster_sha256(
            held_out_targets
        ):
            raise ValueError("held_out_target_roster_sha256 does not match its roster")
        if not isinstance(self.formal_token_corpus_verified, bool):
            raise TypeError("formal_token_corpus_verified must be bool")
        formal_fields = (
            "formal_token_corpus_index_sha256",
            "formal_token_corpus_training_bundle_manifest_sha256",
            "formal_token_corpus_event_roster_sha256",
            "formal_token_corpus_patient_roster_sha256",
            "formal_token_corpus_tensor_roster_sha256",
        )
        if self.formal_token_corpus_verified:
            for field in formal_fields:
                _require_sha(getattr(self, field), field=field)
        elif any(getattr(self, field) is not None for field in formal_fields):
            raise ValueError("Nonformal runs cannot declare formal corpus lineage")
        patients = tuple(sorted(str(value).strip() for value in self.concept_training_patient_ids))
        if patients != self.concept_training_patient_ids or any(not value for value in patients):
            raise ValueError("Concept-training patient IDs must be trimmed and sorted")
        if self.concept_training_patient_roster_sha256 != _public_patient_roster_sha256(
            patients
        ):
            raise ValueError("Concept-training patient roster SHA mismatch")
        if (
            self.formal_token_corpus_verified
            and self.formal_token_corpus_patient_roster_sha256
            != self.concept_training_patient_roster_sha256
        ):
            raise ValueError("Formal corpus patient roster does not match the run")
        if not isinstance(self.epochs, tuple):
            raise TypeError("epochs must be a tuple")
        if any(not isinstance(receipt, IctalTrainingEpochReceipt) for receipt in self.epochs):
            raise TypeError("epochs must contain IctalTrainingEpochReceipt values")
        if len(self.epochs) != self.config.fixed_epochs:
            raise ValueError("Run must contain the predeclared fixed epoch count")
        if tuple(receipt.epoch for receipt in self.epochs) != tuple(
            range(self.config.fixed_epochs)
        ):
            raise ValueError("Epoch receipts must be contiguous and zero-based")
        if (
            isinstance(self.selected_epoch, bool)
            or not isinstance(self.selected_epoch, int)
            or self.selected_epoch != self.config.fixed_epochs - 1
        ):
            raise ValueError("Fixed-epoch policy must select the final epoch")
        if any(receipt.n_patients != len(patients) for receipt in self.epochs):
            raise ValueError("Every epoch must consume the complete patient roster")
        if any(
            len(receipt.patient_order) != len(patients)
            or set(receipt.patient_order) != set(patients)
            for receipt in self.epochs
        ):
            raise ValueError("Every epoch patient order must equal the complete roster")
        if len({receipt.n_events for receipt in self.epochs}) != 1:
            raise ValueError("Event count drifted between fixed training epochs")
        if len({receipt.n_observed_labels for receipt in self.epochs}) != 1:
            raise ValueError("Observed-label count drifted between fixed training epochs")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class IctalTrainingRunArtifact:
    """Strict on-disk receipt artifact used by the concept checkpoint."""

    path: Path
    training_run_receipt: IctalTrainingRunReceipt
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")
        if not isinstance(self.training_run_receipt, IctalTrainingRunReceipt):
            raise TypeError("training_run_receipt must be IctalTrainingRunReceipt")
        _require_sha(self.artifact_sha256, field="artifact_sha256")

    @property
    def training_run_receipt_sha256(self) -> str:
        return self.training_run_receipt.receipt_sha256


_TRAINING_CONFIG_FIELDS = frozenset(
    field.name for field in fields(IctalTrainingConfig)
)
_DETERMINISM_POLICY_FIELDS = frozenset(
    field.name for field in fields(IctalDeterminismPolicyReceipt)
)
_EPOCH_RECEIPT_FIELDS = frozenset(
    field.name for field in fields(IctalTrainingEpochReceipt)
)
_RUN_RECEIPT_FIELDS = frozenset(
    field.name for field in fields(IctalTrainingRunReceipt)
)
_RUN_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "training_run_receipt_sha256",
        "training_run_receipt",
    }
)


def _json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _json_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    return value


def _run_receipt_from_payload(value: object) -> IctalTrainingRunReceipt:
    payload = _json_object(value, field="training_run_receipt")
    _require_exact_fields(
        payload, _RUN_RECEIPT_FIELDS, label="training_run_receipt"
    )
    config_payload = _json_object(payload["config"], field="config")
    _require_exact_fields(
        config_payload, _TRAINING_CONFIG_FIELDS, label="training config"
    )
    config = IctalTrainingConfig(**config_payload)
    determinism_payload = _json_object(
        payload["determinism_policy"], field="determinism_policy"
    )
    _require_exact_fields(
        determinism_payload,
        _DETERMINISM_POLICY_FIELDS,
        label="determinism_policy",
    )
    determinism_policy = IctalDeterminismPolicyReceipt(**determinism_payload)
    epoch_payloads = _json_list(payload["epochs"], field="epochs")
    epochs: list[IctalTrainingEpochReceipt] = []
    for index, raw_epoch in enumerate(epoch_payloads):
        epoch = _json_object(raw_epoch, field=f"epochs[{index}]")
        _require_exact_fields(
            epoch, _EPOCH_RECEIPT_FIELDS, label=f"epochs[{index}]"
        )
        normalized_epoch = dict(epoch)
        normalized_epoch["patient_order"] = tuple(
            _json_list(epoch["patient_order"], field=f"epochs[{index}].patient_order")
        )
        epochs.append(IctalTrainingEpochReceipt(**normalized_epoch))
    normalized = dict(payload)
    normalized["config"] = config
    normalized["determinism_policy"] = determinism_policy
    normalized["concept_training_patient_ids"] = tuple(
        _json_list(
            payload["concept_training_patient_ids"],
            field="concept_training_patient_ids",
        )
    )
    for roster_field in (
        "training_target_patient_ids",
        "held_out_target_patient_ids",
    ):
        normalized[roster_field] = tuple(
            _json_list(payload[roster_field], field=roster_field)
        )
    normalized["epochs"] = tuple(epochs)
    return IctalTrainingRunReceipt(**normalized)


def _run_artifact_payload(
    receipt: IctalTrainingRunReceipt,
) -> dict[str, object]:
    if not isinstance(receipt, IctalTrainingRunReceipt):
        raise TypeError("receipt must be IctalTrainingRunReceipt")
    return {
        "schema_version": ICTAL_TRAINING_RUN_ARTIFACT_SCHEMA,
        "training_run_receipt_sha256": receipt.receipt_sha256,
        "training_run_receipt": asdict(receipt),
    }


def _receipt_from_artifact_payload(
    payload: dict[str, object],
) -> IctalTrainingRunReceipt:
    _require_exact_fields(payload, _RUN_ARTIFACT_FIELDS, label="training-run artifact")
    if payload["schema_version"] != ICTAL_TRAINING_RUN_ARTIFACT_SCHEMA:
        raise ValueError("Unexpected training-run artifact schema")
    declared_sha = _require_sha(
        payload["training_run_receipt_sha256"],
        field="training_run_receipt_sha256",
    )
    receipt = _run_receipt_from_payload(payload["training_run_receipt"])
    if receipt.receipt_sha256 != declared_sha:
        raise ValueError("Training-run receipt SHA mismatch")
    return receipt


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_ictal_training_run_receipt(
    path: str | Path,
    receipt: IctalTrainingRunReceipt,
) -> IctalTrainingRunArtifact:
    """Atomically publish one canonical JSON receipt; overwrite is forbidden."""

    payload = _run_artifact_payload(receipt)
    encoded = _canonical_json_bytes(payload)
    if not 1 <= len(encoded) <= _MAX_RUN_ARTIFACT_BYTES:
        raise ValueError("Serialized training-run artifact has an invalid size")
    reconstructed = _receipt_from_artifact_payload(_parse_canonical_json(encoded))
    if reconstructed != receipt:
        raise ValueError("Training-run receipt is unstable under safe reconstruction")
    artifact_sha = hashlib.sha256(encoded).hexdigest()

    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Training-run artifact bundle already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or target.parent.resolve(strict=True) != target.parent:
        raise ValueError("Training-run artifact parent cannot traverse a symlink")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        artifact_file = temporary / ICTAL_TRAINING_RUN_ARTIFACT_FILENAME
        artifact_file.write_bytes(encoded)
        _fsync_file(artifact_file)
        _fsync_directory(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Training-run artifact bundle already exists: {target}"
            )
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return IctalTrainingRunArtifact(
        path=target,
        training_run_receipt=receipt,
        artifact_sha256=artifact_sha,
    )


def load_ictal_training_run_receipt(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
    expected_training_run_receipt_sha256: str | None = None,
) -> IctalTrainingRunArtifact:
    """Strictly load and verify a canonical training-run receipt artifact."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Training-run artifact bundle must be a regular directory")
    if source.resolve(strict=True) != source:
        raise ValueError("Training-run artifact path may not traverse a symlink")
    entries = tuple(source.iterdir())
    if len(entries) != 1 or entries[0].name != ICTAL_TRAINING_RUN_ARTIFACT_FILENAME:
        raise ValueError("Training-run artifact bundle has missing or unknown files")
    artifact_file = entries[0]
    if artifact_file.is_symlink() or not artifact_file.is_file():
        raise ValueError("Training-run artifact member must be a regular file")
    before = artifact_file.stat()
    if not 1 <= before.st_size <= _MAX_RUN_ARTIFACT_BYTES:
        raise ValueError("Training-run artifact file has an invalid size")
    encoded = artifact_file.read_bytes()
    after = artifact_file.stat()
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
        raise RuntimeError("Training-run artifact changed while it was read")
    artifact_sha = hashlib.sha256(encoded).hexdigest()
    if expected_artifact_sha256 is not None and artifact_sha != _require_sha(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Training-run artifact SHA mismatch")
    receipt = _receipt_from_artifact_payload(_parse_canonical_json(encoded))
    if (
        expected_training_run_receipt_sha256 is not None
        and receipt.receipt_sha256
        != _require_sha(
            expected_training_run_receipt_sha256,
            field="expected_training_run_receipt_sha256",
        )
    ):
        raise ValueError("Training-run receipt does not match the expected SHA")
    return IctalTrainingRunArtifact(
        path=source,
        training_run_receipt=receipt,
        artifact_sha256=artifact_sha,
    )


def train_fixed_epoch_ictal_head(
    head: IctalInvolvementHead,
    dataset: IctalTokenBagDataset,
    *,
    config: IctalTrainingConfig,
    split_manifest_sha256: str,
    oof_protocol_receipt_sha256: str,
    oof_plan_receipt_sha256: str,
    oof_fold: int | None,
    training_target_patient_ids: tuple[str, ...],
    held_out_target_patient_ids: tuple[str, ...],
    training_target_roster_sha256: str,
    held_out_target_roster_sha256: str,
) -> IctalTrainingRunReceipt:
    """Run deterministic fixed epochs and return the checkpoint-bound receipt."""

    if not isinstance(head, IctalInvolvementHead):
        raise TypeError("head must be IctalInvolvementHead")
    if not isinstance(dataset, IctalTokenBagDataset):
        raise TypeError("dataset must be the lazy IctalTokenBagDataset")
    if not dataset.training_authorized:
        raise ValueError("Evaluation-only ictal token corpora cannot be used for training")
    if not isinstance(config, IctalTrainingConfig):
        raise TypeError("config must be IctalTrainingConfig")
    protocol_sha = _require_sha(
        oof_protocol_receipt_sha256, field="oof_protocol_receipt_sha256"
    )
    plan_sha = _require_sha(oof_plan_receipt_sha256, field="oof_plan_receipt_sha256")
    split_sha = _require_sha(
        split_manifest_sha256, field="split_manifest_sha256"
    )
    canonical_training_targets = _target_patient_roster(
        training_target_patient_ids,
        field="training_target_patient_ids",
    )
    canonical_held_out_targets = _target_patient_roster(
        held_out_target_patient_ids,
        field="held_out_target_patient_ids",
    )
    if set(canonical_training_targets) & set(canonical_held_out_targets):
        raise ValueError("Training and held-out target patients must be disjoint")
    if patient_roster_sha256(canonical_training_targets) != _require_sha(
        training_target_roster_sha256,
        field="training_target_roster_sha256",
    ):
        raise ValueError("training_target_roster_sha256 does not match its roster")
    if patient_roster_sha256(canonical_held_out_targets) != _require_sha(
        held_out_target_roster_sha256,
        field="held_out_target_roster_sha256",
    ):
        raise ValueError("held_out_target_roster_sha256 does not match its roster")
    if oof_fold is not None and (
        isinstance(oof_fold, bool)
        or not isinstance(oof_fold, int)
        or oof_fold not in range(5)
    ):
        raise ValueError("oof_fold must be None or an integer in [0,4]")
    head_devices = {parameter.device for parameter in head.parameters()}
    if len(head_devices) != 1:
        raise ValueError("Ictal head parameters must occupy exactly one device")
    head_device = next(iter(head_devices))
    execution_device_type = head_device.type
    if execution_device_type not in {"cpu", "cuda"}:
        raise ValueError("Ictal head device must be cpu or cuda")
    if execution_device_type == "cuda":
        validate_ictal_cuda_environment()
        cuda_index = (
            head_device.index
            if head_device.index is not None
            else torch.cuda.current_device()
        )
        cuda_devices = [cuda_index]
    else:
        cuda_devices = []
    epoch_receipts: list[IctalTrainingEpochReceipt] = []
    canonical_patients = dataset.patient_ids
    with ictal_determinism_runtime(
        config, execution_device_type=execution_device_type
    ) as determinism_policy:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(config.seed)
            if execution_device_type == "cuda":
                torch.cuda.manual_seed_all(config.seed)
            initial_head_sha = ictal_head_state_sha256(head)
            optimizer = torch.optim.AdamW(
                head.parameters(),
                lr=float(config.learning_rate),
                weight_decay=float(config.weight_decay),
            )
            for epoch in range(config.fixed_epochs):
                order = list(canonical_patients)
                random.Random(config.seed + epoch).shuffle(order)
                order_tuple = tuple(order)
                output = train_cached_ictal_epoch(
                    head,
                    dataset,
                    optimizer,
                    patient_order=order_tuple,
                    max_grad_norm=config.max_grad_norm,
                    event_microbatch_size=config.event_microbatch_size,
                )
                # Fail immediately if an optimizer step corrupts any parameter.
                ictal_head_state_sha256(head)
                epoch_receipts.append(
                    IctalTrainingEpochReceipt(
                        epoch=epoch,
                        patient_order=order_tuple,
                        patient_order_sha256=_canonical_sha256(order_tuple),
                        mean_patient_loss=output.mean_patient_loss,
                        n_patients=output.n_patients,
                        n_events=output.n_events,
                        n_observed_labels=output.n_observed_labels,
                    )
                )
            final_head_sha = ictal_head_state_sha256(head)
    return IctalTrainingRunReceipt(
        config=config,
        config_sha256=config.receipt_sha256,
        determinism_policy=determinism_policy,
        determinism_policy_sha256=determinism_policy.receipt_sha256,
        split_manifest_sha256=split_sha,
        oof_protocol_receipt_sha256=protocol_sha,
        oof_plan_receipt_sha256=plan_sha,
        oof_fold=oof_fold,
        training_target_patient_ids=canonical_training_targets,
        held_out_target_patient_ids=canonical_held_out_targets,
        training_target_roster_sha256=training_target_roster_sha256,
        held_out_target_roster_sha256=held_out_target_roster_sha256,
        training_manifest_sha256=dataset.training_manifest_sha256,
        token_source_manifest_sha256=dataset.token_source_manifest_sha256,
        foundation_feature_receipt_sha256=(
            dataset.foundation_feature_receipt_sha256
        ),
        formal_token_corpus_verified=dataset.formal_token_corpus_verified,
        formal_token_corpus_index_sha256=(
            dataset.formal_token_corpus_index_sha256
        ),
        formal_token_corpus_training_bundle_manifest_sha256=(
            dataset.formal_token_corpus_training_bundle_manifest_sha256
        ),
        formal_token_corpus_event_roster_sha256=(
            dataset.formal_token_corpus_event_roster_sha256
        ),
        formal_token_corpus_patient_roster_sha256=(
            dataset.formal_token_corpus_patient_roster_sha256
        ),
        formal_token_corpus_tensor_roster_sha256=(
            dataset.formal_token_corpus_tensor_roster_sha256
        ),
        initial_head_state_sha256=initial_head_sha,
        final_head_state_sha256=final_head_sha,
        concept_training_patient_ids=canonical_patients,
        concept_training_patient_roster_sha256=_public_patient_roster_sha256(
            canonical_patients
        ),
        epochs=tuple(epoch_receipts),
        selected_epoch=config.fixed_epochs - 1,
    )


__all__ = [
    "ICTAL_CUBLAS_WORKSPACE_CONFIG",
    "ICTAL_DETERMINISM_POLICY_SCHEMA",
    "ICTAL_HEAD_STATE_HASH_SCHEMA",
    "ICTAL_IDENTITY_SCALER_SCHEMA",
    "ICTAL_IDENTITY_SCALER_SHA256",
    "ICTAL_TRAINING_CONFIG_SCHEMA",
    "ICTAL_TRAINING_RUN_ARTIFACT_FILENAME",
    "ICTAL_TRAINING_RUN_ARTIFACT_SCHEMA",
    "ICTAL_TRAINING_RUN_SCHEMA",
    "IctalDeterminismPolicyReceipt",
    "IctalTrainingConfig",
    "IctalTrainingEpochReceipt",
    "IctalTrainingRunArtifact",
    "IctalTrainingRunReceipt",
    "ictal_head_state_sha256",
    "ictal_determinism_runtime",
    "load_ictal_training_run_receipt",
    "save_ictal_training_run_receipt",
    "train_fixed_epoch_ictal_head",
    "validate_ictal_cuda_environment",
]
