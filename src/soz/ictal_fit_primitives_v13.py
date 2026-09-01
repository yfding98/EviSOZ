"""Target-agnostic primitives for the process-isolated v13 fit runtime.

This module intentionally imports no dataset package, pandas, DeepSOZ,
native-evaluation code, or full-corpus loader.  The formal v13 trainer uses
it through a minimal package bootstrap so importing ``src.soz.__init__``
cannot make forbidden target sources reachable as an import side effect.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import numpy as np
import torch


ICTAL_TRAINING_CONFIG_SCHEMA = "soz_ictal_training_config_v2"
ICTAL_HEAD_STATE_HASH_SCHEMA = "soz_ictal_head_state_hash_v1"
ICTAL_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
LABRAM_K31_TARGET_SEMANTICS = "tusz_bipolar_edge_time_involvement_not_soz"
LABRAM_K31_OOF_RUN_SCHEMA_V1_2 = "soz_labram_k31_ictal_oof_recovery_run_v1_2"
LABRAM_K31_EXECUTION_RECEIPT_SCHEMA = "soz_labram_k31_execution_receipt_v1"

_SELECTION_RE = re.compile(r"fold([0-4])|final")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXECUTION_FIELDS = frozenset(
    {
        "schema_version",
        "torch_version",
        "cuda_runtime_version",
        "cudnn_version",
        "device_type",
        "device_name",
        "compute_capability",
        "optimizer_class",
        "optimizer_effective_hyperparameters",
        "training_config_sha256",
    }
)
_OPTIMIZER_FIELDS = frozenset(
    {
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def patient_roster(
    values: Sequence[object], *, field: str, allow_empty: bool
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a patient sequence")
    roster = tuple(str(value).strip() for value in values)
    if (not allow_empty and not roster) or any(not value for value in roster):
        raise ValueError(f"{field} must contain non-empty patient IDs")
    if roster != tuple(sorted(roster)) or len(set(roster)) != len(roster):
        raise ValueError(f"{field} must be unique and sorted")
    return roster


def patient_roster_sha256(values: Sequence[object]) -> str:
    roster = patient_roster(values, field="patient_roster", allow_empty=False)
    return canonical_sha256(roster)


def selection(value: object) -> tuple[str, int | None]:
    text = str(value).strip().lower()
    match = _SELECTION_RE.fullmatch(text)
    if match is None:
        raise ValueError("selection must be fold0..fold4 or final")
    return text, None if text == "final" else int(match.group(1))


def safe_new_output(value: str | Path) -> Path:
    target = Path(os.path.abspath(value))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("v13 output requires a concrete path with an existing parent")
    if os.path.lexists(target):
        raise FileExistsError(f"v13 output already exists: {target}")
    return target


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class IctalTrainingConfig:
    """Local closed copy of the already frozen k31 optimization policy."""

    seed: int = 20260808
    fixed_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    event_microbatch_size: int = 4
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
            if not math.isfinite(float(source)) or float(source) <= 0:
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
            "cublas_workspace_config": ICTAL_CUBLAS_WORKSPACE_CONFIG,
        }
        if any(getattr(self, field) != expected for field, expected in frozen.items()):
            raise ValueError("v13 optimization policy changed")
        flags = {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        if any(
            not isinstance(getattr(self, field), bool)
            or getattr(self, field) is not expected
            for field, expected in flags.items()
        ):
            raise ValueError("v13 deterministic policy changed")


def validate_epoch_payload(
    value: object, *, field: str, expected_patient_count: int
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an epoch mapping")
    payload = dict(value)
    expected = {
        "mean_patient_loss",
        "n_patients",
        "n_events",
        "n_observed_labels",
    }
    if set(payload) != expected:
        raise ValueError(f"{field} violates the closed epoch schema")
    loss = payload["mean_patient_loss"]
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(float(loss))
        or float(loss) < 0.0
    ):
        raise ValueError(f"{field}.mean_patient_loss must be finite and non-negative")
    normalized: dict[str, object] = {"mean_patient_loss": float(loss)}
    for name in ("n_patients", "n_events", "n_observed_labels"):
        observed = payload[name]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 1:
            raise ValueError(f"{field}.{name} must be a positive integer")
        normalized[name] = observed
    if normalized["n_patients"] != expected_patient_count:
        raise ValueError(f"{field}.n_patients disagrees with its patient roster")
    return normalized


def ictal_head_state_sha256(head: torch.nn.Module) -> str:
    if not isinstance(head, torch.nn.Module):
        raise TypeError("head must be a torch module")
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
        metadata = json.dumps(
            {
                "dtype": array.dtype.newbyteorder("<").str,
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


def validate_ictal_cuda_environment() -> str:
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed != ICTAL_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUDA v13 training requires CUBLAS_WORKSPACE_CONFIG=':4096:8'"
        )
    return observed


@contextmanager
def ictal_determinism_runtime(
    config: IctalTrainingConfig, *, execution_device_type: str
):
    if not isinstance(config, IctalTrainingConfig):
        raise TypeError("config must be the local v13 IctalTrainingConfig")
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
        yield
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


def validate_execution_receipt(
    value: object, *, training_config: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _EXECUTION_FIELDS:
        raise ValueError("execution_receipt violates its closed schema")
    receipt = dict(value)
    if receipt["schema_version"] != LABRAM_K31_EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("Unsupported execution receipt schema")
    for field in ("torch_version", "device_type", "device_name", "optimizer_class"):
        if not isinstance(receipt[field], str) or not receipt[field]:
            raise ValueError(f"execution_receipt.{field} must be non-empty")
    if receipt["device_type"] not in {"cpu", "cuda"}:
        raise ValueError("execution receipt device_type must be cpu or cuda")
    if receipt["cuda_runtime_version"] is not None and not isinstance(
        receipt["cuda_runtime_version"], str
    ):
        raise TypeError("execution_receipt.cuda_runtime_version must be string or null")
    if receipt["cudnn_version"] is not None and (
        isinstance(receipt["cudnn_version"], bool)
        or not isinstance(receipt["cudnn_version"], int)
        or receipt["cudnn_version"] < 1
    ):
        raise ValueError("execution_receipt.cudnn_version must be positive or null")
    capability = receipt["compute_capability"]
    if capability is not None and (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in capability)
    ):
        raise ValueError("execution_receipt.compute_capability is invalid")
    if receipt["optimizer_class"] != "torch.optim.AdamW":
        raise ValueError("v13 optimizer must be torch.optim.AdamW")
    optimizer = receipt["optimizer_effective_hyperparameters"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != _OPTIMIZER_FIELDS:
        raise ValueError("optimizer effective parameters violate their closed schema")
    optimizer = dict(optimizer)
    if (
        float(optimizer["lr"]) != float(training_config["learning_rate"])
        or float(optimizer["weight_decay"])
        != float(training_config["weight_decay"])
    ):
        raise ValueError("Optimizer receipt disagrees with the training config")
    betas = optimizer["betas"]
    if not isinstance(betas, list) or len(betas) != 2 or any(
        not math.isfinite(float(item)) or not 0.0 <= float(item) < 1.0
        for item in betas
    ):
        raise ValueError("AdamW beta receipt is invalid")
    if not math.isfinite(float(optimizer["eps"])) or float(optimizer["eps"]) <= 0:
        raise ValueError("AdamW eps receipt is invalid")
    for field in ("amsgrad", "maximize", "capturable", "differentiable"):
        if not isinstance(optimizer[field], bool):
            raise TypeError(f"AdamW {field} receipt must be boolean")
    for field in ("foreach", "fused"):
        if optimizer[field] is not None and not isinstance(optimizer[field], bool):
            raise TypeError(f"AdamW {field} receipt must be bool or null")
    if receipt["training_config_sha256"] != canonical_sha256(dict(training_config)):
        raise ValueError("Execution receipt does not bind the training config")
    receipt["optimizer_effective_hyperparameters"] = optimizer
    return receipt


@dataclass(frozen=True)
class VerifiedFitOnlyIctalTargetSnapshotV13:
    path: Path
    manifest_sha256: str
    receipt_sha256: str
    training_manifest_sha256: str
    training_corpus_index_sha256: str
    native_manifest_sha256: str
    native_corpus_index_sha256: str
    training_event_rows: tuple[tuple[str, str], ...]
    native_event_rows: tuple[tuple[str, str], ...]
    training_targets: torch.Tensor
    training_target_mask: torch.Tensor
    native_targets: torch.Tensor
    native_target_mask: torch.Tensor


__all__ = (
    "ICTAL_CUBLAS_WORKSPACE_CONFIG",
    "IctalTrainingConfig",
    "LABRAM_K31_EXECUTION_RECEIPT_SCHEMA",
    "LABRAM_K31_OOF_RUN_SCHEMA_V1_2",
    "LABRAM_K31_TARGET_SEMANTICS",
    "VerifiedFitOnlyIctalTargetSnapshotV13",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_sha256",
    "ictal_determinism_runtime",
    "ictal_head_state_sha256",
    "patient_roster",
    "patient_roster_sha256",
    "require_sha256",
    "safe_new_output",
    "selection",
    "validate_epoch_payload",
    "validate_execution_receipt",
    "validate_ictal_cuda_environment",
)
