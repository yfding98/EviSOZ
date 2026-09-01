"""Post-freeze global calibration for patient-level SOZ logits.

The reasoner is trained on equal-event-mean *raw patient logits*.  Its
uncalibrated training probability is exactly ``sigmoid(raw_patient_logits)``.
Only after the reasoner is frozen may this module fit the prespecified global
affine temperature calibrator on source-development patients.  The calibrator
never receives a model optimizer and cannot send gradients back to the
reasoner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Sequence

import torch
import torch.nn.functional as F

from .data.deepsoz import normalize_patient_id
from .data.provenance import patient_roster_sha256
from .geometry import N_STANDARD_CHANNELS
from .models.reasoner import AdditiveEvidenceReasoner


FROZEN_REASONER_CHECKPOINT_SCHEMA = "soz_frozen_reasoner_checkpoint_v1"
REASONER_CALIBRATION_DATA_SCHEMA = "soz_reasoner_source_dev_logits_v2"
GLOBAL_AFFINE_CALIBRATOR_SCHEMA = "soz_global_affine_calibrator_v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FROZEN_ISSUER = object()
_DATA_ISSUER = object()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(value.shape)}|{value.dtype}".encode("ascii")
    digest.update(metadata)
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _calibration_optimizer_policy(
    *, max_steps: int, learning_rate: float
) -> str:
    """Return the one frozen optimizer-policy spelling accepted in receipts."""

    return (
        f"adam_cpu_float64_steps={max_steps}_lr={float(learning_rate):.12g}_"
        "patient_macro_masked_nll_T[0.05,20]_b[-20,20]_identity_fallback"
    )


def reasoner_state_sha256(model: AdditiveEvidenceReasoner) -> str:
    """Digest the complete named reasoner state without serializing pickle."""

    if not isinstance(model, AdditiveEvidenceReasoner):
        raise TypeError("model must be an AdditiveEvidenceReasoner")
    digest = hashlib.sha256()
    state = model.state_dict()
    if not state:
        raise ValueError("Reasoner state cannot be empty")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"Reasoner state {name} must be finite floating point")
        metadata = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode("ascii")
        digest.update(len(metadata).to_bytes(4, "little"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def uncalibrated_training_probabilities(
    raw_patient_logits: torch.Tensor,
) -> torch.Tensor:
    """The only probability transform permitted during reasoner fitting."""

    if not raw_patient_logits.is_floating_point() or not torch.isfinite(
        raw_patient_logits
    ).all():
        raise ValueError("raw_patient_logits must be finite floating point")
    return torch.sigmoid(raw_patient_logits)


@dataclass(frozen=True)
class FrozenReasonerCheckpointReceipt:
    """Lineage proving which completed reasoner state calibration may consume."""

    state_sha256: str
    training_run_receipt_sha256: str
    evidence_authorization_sha256: str
    source_train_patient_ids: tuple[str, ...]
    source_train_roster_sha256: str
    source_dev_patient_ids: tuple[str, ...]
    source_dev_roster_sha256: str
    parameter_count: int
    training_probability_transform: str = (
        "sigmoid_equal_event_mean_raw_patient_logits"
    )
    calibrator_stage_policy: str = (
        "fit_source_dev_only_after_reasoner_freeze_no_gradient_feedback"
    )
    schema_version: str = FROZEN_REASONER_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "state_sha256",
            "training_run_receipt_sha256",
            "evidence_authorization_sha256",
            "source_train_roster_sha256",
            "source_dev_roster_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        roster = tuple(
            sorted(normalize_patient_id(value) for value in self.source_train_patient_ids)
        )
        if not roster or len(set(roster)) != len(roster):
            raise ValueError("Frozen checkpoint requires a unique source-train roster")
        object.__setattr__(self, "source_train_patient_ids", roster)
        if self.source_train_roster_sha256 != patient_roster_sha256(roster):
            raise ValueError("source_train_roster_sha256 does not match its roster")
        dev_roster = tuple(
            sorted(normalize_patient_id(value) for value in self.source_dev_patient_ids)
        )
        if not dev_roster or len(set(dev_roster)) != len(dev_roster):
            raise ValueError("Frozen checkpoint requires a unique source-dev roster")
        if set(roster) & set(dev_roster):
            raise ValueError("Source-train and source-dev rosters must be disjoint")
        object.__setattr__(self, "source_dev_patient_ids", dev_roster)
        if self.source_dev_roster_sha256 != patient_roster_sha256(dev_roster):
            raise ValueError("source_dev_roster_sha256 does not match its roster")
        if (
            isinstance(self.parameter_count, bool)
            or not isinstance(self.parameter_count, int)
            or self.parameter_count < 1
        ):
            raise ValueError("parameter_count must be a positive integer")
        if self.training_probability_transform != (
            "sigmoid_equal_event_mean_raw_patient_logits"
        ):
            raise ValueError("Reasoner training probability transform cannot change")
        if self.calibrator_stage_policy != (
            "fit_source_dev_only_after_reasoner_freeze_no_gradient_feedback"
        ):
            raise ValueError("Reasoner/calibrator stage boundary cannot be weakened")
        if self.schema_version != FROZEN_REASONER_CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported frozen reasoner checkpoint schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class FrozenReasonerCheckpoint:
    """In-memory freeze capability required by the calibrator fitter."""

    model: AdditiveEvidenceReasoner
    receipt: FrozenReasonerCheckpointReceipt
    _issuer_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer_token is not _FROZEN_ISSUER:
            raise PermissionError("Frozen reasoner capability can only be issued by freezer")
        self.assert_unchanged()
        object.__setattr__(self, "_issuer_token", None)

    def assert_unchanged(self) -> None:
        if self.model.training:
            raise ValueError("Frozen reasoner must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Frozen reasoner parameters cannot require gradients")
        if reasoner_state_sha256(self.model) != self.receipt.state_sha256:
            raise ValueError("Reasoner state changed after calibration freeze")


def freeze_reasoner_checkpoint(
    model: AdditiveEvidenceReasoner,
    *,
    training_run_receipt_sha256: str,
    evidence_authorization_sha256: str,
    source_train_patient_ids: Sequence[object],
    source_dev_patient_ids: Sequence[object],
) -> FrozenReasonerCheckpoint:
    """Freeze a completed source-train reasoner before dev logits are exposed."""

    if not isinstance(model, AdditiveEvidenceReasoner):
        raise TypeError("model must be an AdditiveEvidenceReasoner")
    model.eval()
    model.requires_grad_(False)
    roster = tuple(
        sorted(normalize_patient_id(value) for value in source_train_patient_ids)
    )
    dev_roster = tuple(
        sorted(normalize_patient_id(value) for value in source_dev_patient_ids)
    )
    receipt = FrozenReasonerCheckpointReceipt(
        state_sha256=reasoner_state_sha256(model),
        training_run_receipt_sha256=training_run_receipt_sha256,
        evidence_authorization_sha256=evidence_authorization_sha256,
        source_train_patient_ids=roster,
        source_train_roster_sha256=patient_roster_sha256(roster),
        source_dev_patient_ids=dev_roster,
        source_dev_roster_sha256=patient_roster_sha256(dev_roster),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    return FrozenReasonerCheckpoint(
        model=model,
        receipt=receipt,
        _issuer_token=_FROZEN_ISSUER,
    )


@dataclass(frozen=True)
class ReasonerCalibrationData:
    """Detached source-dev patient logits and their masked benchmark targets."""

    patient_ids: tuple[str, ...]
    raw_patient_logits: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    reasoner_state_sha256: str
    evidence_authorization_sha256: str
    verified_target_v2_receipt_sha256: str
    authorized_dev_cache_receipt_sha256: str
    patient_aggregation_receipt_sha256: str
    raw_logits_sha256: str
    targets_sha256: str
    target_mask_sha256: str
    model_split: str = "source_dev"
    aggregation_policy: str = "equal_event_mean_raw_logits"
    schema_version: str = REASONER_CALIBRATION_DATA_SCHEMA
    _issuer_token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._issuer_token is not _DATA_ISSUER:
            raise PermissionError("Calibration data must be issued by strict builder")
        if self.model_split != "source_dev":
            raise ValueError("Calibrator fitting is restricted to source_dev")
        if self.aggregation_policy != "equal_event_mean_raw_logits":
            raise ValueError("Calibration must consume equal-mean raw patient logits")
        roster = tuple(normalize_patient_id(value) for value in self.patient_ids)
        if len(set(roster)) != len(roster) or len(roster) < 2:
            raise ValueError("Calibration requires unique source-dev patients")
        object.__setattr__(self, "patient_ids", roster)
        expected_shape = (len(roster), N_STANDARD_CHANNELS)
        if tuple(self.raw_patient_logits.shape) != expected_shape:
            raise ValueError("raw_patient_logits must have shape [P,19]")
        if tuple(self.targets.shape) != expected_shape or tuple(self.target_mask.shape) != expected_shape:
            raise ValueError("Calibration targets and mask must have shape [P,19]")
        if self.raw_patient_logits.dtype != torch.float64 or self.targets.dtype != torch.float64:
            raise TypeError("Calibration logits and targets must be CPU float64")
        if self.target_mask.dtype != torch.bool:
            raise TypeError("Calibration target_mask must be bool")
        if any(
            tensor.device.type != "cpu"
            for tensor in (self.raw_patient_logits, self.targets, self.target_mask)
        ):
            raise ValueError("Calibration data must be detached on CPU")
        if self.raw_patient_logits.requires_grad or self.targets.requires_grad:
            raise ValueError("Calibration data cannot carry a reasoner graph")
        if not torch.isfinite(self.raw_patient_logits).all() or not torch.isfinite(
            self.targets[self.target_mask]
        ).all():
            raise ValueError("Calibration values must be finite")
        observed = self.targets[self.target_mask]
        if observed.numel() < 1 or not torch.all((observed == 0) | (observed == 1)):
            raise ValueError("Observed calibration targets must be explicitly binary")
        positive = ((self.targets == 1) & self.target_mask).any(dim=1)
        negative = ((self.targets == 0) & self.target_mask).any(dim=1)
        if not (positive & negative).all():
            raise ValueError(
                "Each calibration patient requires explicit positive and complement-negative support"
            )
        for name in (
            "reasoner_state_sha256",
            "evidence_authorization_sha256",
            "verified_target_v2_receipt_sha256",
            "authorized_dev_cache_receipt_sha256",
            "patient_aggregation_receipt_sha256",
            "raw_logits_sha256",
            "targets_sha256",
            "target_mask_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        if self.raw_logits_sha256 != _tensor_sha256(
            "raw_patient_logits", self.raw_patient_logits
        ):
            raise ValueError("raw_logits_sha256 does not match logits")
        if self.targets_sha256 != _tensor_sha256("targets", self.targets):
            raise ValueError("targets_sha256 does not match targets")
        if self.target_mask_sha256 != _tensor_sha256(
            "target_mask", self.target_mask
        ):
            raise ValueError("target_mask_sha256 does not match mask")
        object.__setattr__(self, "_issuer_token", None)

    @property
    def patient_roster_sha256(self) -> str:
        return patient_roster_sha256(self.patient_ids)


def build_reasoner_calibration_data(
    checkpoint: FrozenReasonerCheckpoint,
    *,
    patient_ids: Sequence[object],
    raw_patient_logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    evidence_authorization_sha256: str,
    verified_target_v2_receipt_sha256: str,
    authorized_dev_cache_receipt_sha256: str,
    patient_aggregation_receipt_sha256: str,
    model_split: str = "source_dev",
) -> ReasonerCalibrationData:
    """Detach source-dev logits and bind all independently issued receipts.

    The three receipt digests are mandatory so a calibration artifact cannot
    be mistaken for an unverified tensor dump.  A production caller must
    obtain them from the verified target-v2 loader, the authorized complete
    source-dev cache roster, and the equal-event patient aggregation run.
    """

    if not isinstance(checkpoint, FrozenReasonerCheckpoint):
        raise TypeError("checkpoint must be a FrozenReasonerCheckpoint")
    checkpoint.assert_unchanged()
    authorization_sha = _require_sha256(
        evidence_authorization_sha256,
        field_name="evidence_authorization_sha256",
    )
    if authorization_sha != checkpoint.receipt.evidence_authorization_sha256:
        raise ValueError("Calibration data uses a different evidence authorization")
    roster = tuple(normalize_patient_id(value) for value in patient_ids)
    if len(set(roster)) != len(roster):
        raise ValueError("Calibration patient_ids cannot contain duplicates")
    if roster != checkpoint.receipt.source_dev_patient_ids:
        raise ValueError(
            "Calibration data must use the complete frozen source-dev roster "
            "in canonical receipt order"
        )
    logits = raw_patient_logits.detach().to(device="cpu", dtype=torch.float64).clone()
    target_values = targets.detach().to(device="cpu", dtype=torch.float64).clone()
    mask = target_mask.detach().to(device="cpu", dtype=torch.bool).clone()
    return ReasonerCalibrationData(
        patient_ids=roster,
        raw_patient_logits=logits,
        targets=target_values,
        target_mask=mask,
        reasoner_state_sha256=checkpoint.receipt.state_sha256,
        evidence_authorization_sha256=authorization_sha,
        verified_target_v2_receipt_sha256=verified_target_v2_receipt_sha256,
        authorized_dev_cache_receipt_sha256=authorized_dev_cache_receipt_sha256,
        patient_aggregation_receipt_sha256=patient_aggregation_receipt_sha256,
        raw_logits_sha256=_tensor_sha256("raw_patient_logits", logits),
        targets_sha256=_tensor_sha256("targets", target_values),
        target_mask_sha256=_tensor_sha256("target_mask", mask),
        model_split=model_split,
        _issuer_token=_DATA_ISSUER,
    )


def _patient_macro_nll(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    elementwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    observed_count = mask.sum(dim=1)
    if (observed_count == 0).any():
        raise ValueError("Every calibration patient requires observed targets")
    patient_loss = (elementwise * mask.to(dtype=elementwise.dtype)).sum(dim=1)
    patient_loss = patient_loss / observed_count.to(dtype=elementwise.dtype)
    return patient_loss.mean()


@dataclass(frozen=True)
class GlobalAffineCalibratorReceipt:
    """Exact source-dev fit receipt for ``sigmoid(logit / T + b)``."""

    frozen_reasoner_receipt_sha256: str
    reasoner_state_sha256: str
    evidence_authorization_sha256: str
    verified_target_v2_receipt_sha256: str
    authorized_dev_cache_receipt_sha256: str
    patient_aggregation_receipt_sha256: str
    source_dev_patient_ids: tuple[str, ...]
    source_dev_roster_sha256: str
    raw_logits_sha256: str
    targets_sha256: str
    target_mask_sha256: str
    temperature: float
    bias: float
    uncalibrated_patient_macro_nll: float
    fitted_patient_macro_nll: float
    optimizer_policy: str
    optimizer_steps: int
    optimizer_learning_rate: float
    fitted_after_reasoner_freeze: bool = True
    target_semantics: str = "deepsoz_benchmark_label_probability_not_biological_soz"
    schema_version: str = GLOBAL_AFFINE_CALIBRATOR_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "frozen_reasoner_receipt_sha256",
            "reasoner_state_sha256",
            "evidence_authorization_sha256",
            "verified_target_v2_receipt_sha256",
            "authorized_dev_cache_receipt_sha256",
            "patient_aggregation_receipt_sha256",
            "source_dev_roster_sha256",
            "raw_logits_sha256",
            "targets_sha256",
            "target_mask_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), field_name=name),
            )
        roster = tuple(
            normalize_patient_id(value) for value in self.source_dev_patient_ids
        )
        if not roster or len(set(roster)) != len(roster):
            raise ValueError("Calibrator requires a unique source-dev roster")
        object.__setattr__(self, "source_dev_patient_ids", roster)
        if self.source_dev_roster_sha256 != patient_roster_sha256(roster):
            raise ValueError("source_dev_roster_sha256 does not match its roster")
        numeric = (
            self.temperature,
            self.bias,
            self.uncalibrated_patient_macro_nll,
            self.fitted_patient_macro_nll,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("Calibrator parameters and losses must be finite")
        if self.temperature <= 0:
            raise ValueError("Calibrator temperature must be positive")
        if self.uncalibrated_patient_macro_nll < 0 or self.fitted_patient_macro_nll < 0:
            raise ValueError("Calibration NLL values cannot be negative")
        if self.fitted_patient_macro_nll > self.uncalibrated_patient_macro_nll + 1e-10:
            raise ValueError("Fitted calibrator cannot be worse than identity on source-dev")
        if (
            isinstance(self.optimizer_steps, bool)
            or not isinstance(self.optimizer_steps, int)
            or self.optimizer_steps < 1
        ):
            raise ValueError("optimizer_steps must be a positive integer")
        if not math.isfinite(float(self.optimizer_learning_rate)) or (
            self.optimizer_learning_rate <= 0
        ):
            raise ValueError("optimizer_learning_rate must be finite and positive")
        expected_optimizer_policy = _calibration_optimizer_policy(
            max_steps=self.optimizer_steps,
            learning_rate=self.optimizer_learning_rate,
        )
        if self.optimizer_policy != expected_optimizer_policy:
            raise ValueError("Calibrator optimizer policy is not the frozen strategy")
        if not self.fitted_after_reasoner_freeze:
            raise ValueError("Calibration before reasoner freeze is forbidden")
        if self.target_semantics != (
            "deepsoz_benchmark_label_probability_not_biological_soz"
        ):
            raise ValueError("Calibrator target claim boundary cannot change")
        if self.schema_version != GLOBAL_AFFINE_CALIBRATOR_SCHEMA:
            raise ValueError("Unsupported global affine calibrator schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class GlobalAffineSOZCalibrator:
    """Frozen two-parameter calibrator with no model-gradient path."""

    receipt: GlobalAffineCalibratorReceipt

    @property
    def temperature(self) -> float:
        return self.receipt.temperature

    @property
    def bias(self) -> float:
        return self.receipt.bias

    def calibrated_probabilities(
        self,
        raw_patient_logits: torch.Tensor,
        checkpoint: FrozenReasonerCheckpoint,
    ) -> torch.Tensor:
        if not isinstance(checkpoint, FrozenReasonerCheckpoint):
            raise TypeError("Calibration application requires frozen checkpoint")
        checkpoint.assert_unchanged()
        if checkpoint.receipt.receipt_sha256 != self.receipt.frozen_reasoner_receipt_sha256:
            raise ValueError("Calibrator belongs to a different frozen reasoner")
        if raw_patient_logits.requires_grad:
            raise ValueError("Calibrator application requires detached raw logits")
        if not raw_patient_logits.is_floating_point() or not torch.isfinite(
            raw_patient_logits
        ).all():
            raise ValueError("raw_patient_logits must be finite floating point")
        return torch.sigmoid(
            raw_patient_logits / float(self.temperature) + float(self.bias)
        )


def fit_global_affine_calibrator(
    checkpoint: FrozenReasonerCheckpoint,
    data: ReasonerCalibrationData,
    *,
    max_steps: int = 300,
    learning_rate: float = 0.05,
) -> GlobalAffineSOZCalibrator:
    """Fit ``T,b`` on detached source-dev logits after reasoner freeze."""

    if not isinstance(checkpoint, FrozenReasonerCheckpoint):
        raise TypeError("checkpoint must be a FrozenReasonerCheckpoint")
    if not isinstance(data, ReasonerCalibrationData):
        raise TypeError("data must be ReasonerCalibrationData")
    checkpoint.assert_unchanged()
    if data.reasoner_state_sha256 != checkpoint.receipt.state_sha256:
        raise ValueError("Calibration logits came from a different reasoner state")
    if data.evidence_authorization_sha256 != checkpoint.receipt.evidence_authorization_sha256:
        raise ValueError("Calibration logits use a different evidence authorization")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    logits = data.raw_patient_logits
    targets = data.targets
    mask = data.target_mask
    with torch.no_grad():
        identity_loss = float(_patient_macro_nll(logits, targets, mask))
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam((log_temperature, bias), lr=float(learning_rate))
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.clamp(
            math.log(0.05), math.log(20.0)
        ).exp()
        calibrated_logits = logits / temperature + bias.clamp(-20.0, 20.0)
        loss = _patient_macro_nll(calibrated_logits, targets, mask)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        fitted_temperature = float(
            log_temperature.clamp(math.log(0.05), math.log(20.0)).exp()
        )
        fitted_bias = float(bias.clamp(-20.0, 20.0))
        fitted_loss = float(
            _patient_macro_nll(
                logits / fitted_temperature + fitted_bias,
                targets,
                mask,
            )
        )
    if not all(
        math.isfinite(value)
        for value in (fitted_temperature, fitted_bias, fitted_loss)
    ) or fitted_loss > identity_loss:
        fitted_temperature = 1.0
        fitted_bias = 0.0
        fitted_loss = identity_loss

    checkpoint.assert_unchanged()
    policy = _calibration_optimizer_policy(
        max_steps=max_steps,
        learning_rate=learning_rate,
    )
    receipt = GlobalAffineCalibratorReceipt(
        frozen_reasoner_receipt_sha256=checkpoint.receipt.receipt_sha256,
        reasoner_state_sha256=checkpoint.receipt.state_sha256,
        evidence_authorization_sha256=checkpoint.receipt.evidence_authorization_sha256,
        verified_target_v2_receipt_sha256=(
            data.verified_target_v2_receipt_sha256
        ),
        authorized_dev_cache_receipt_sha256=(
            data.authorized_dev_cache_receipt_sha256
        ),
        patient_aggregation_receipt_sha256=(
            data.patient_aggregation_receipt_sha256
        ),
        source_dev_patient_ids=data.patient_ids,
        source_dev_roster_sha256=data.patient_roster_sha256,
        raw_logits_sha256=data.raw_logits_sha256,
        targets_sha256=data.targets_sha256,
        target_mask_sha256=data.target_mask_sha256,
        temperature=fitted_temperature,
        bias=fitted_bias,
        uncalibrated_patient_macro_nll=identity_loss,
        fitted_patient_macro_nll=fitted_loss,
        optimizer_policy=policy,
        optimizer_steps=max_steps,
        optimizer_learning_rate=float(learning_rate),
    )
    return GlobalAffineSOZCalibrator(receipt=receipt)


__all__ = [
    "FROZEN_REASONER_CHECKPOINT_SCHEMA",
    "GLOBAL_AFFINE_CALIBRATOR_SCHEMA",
    "REASONER_CALIBRATION_DATA_SCHEMA",
    "FrozenReasonerCheckpoint",
    "FrozenReasonerCheckpointReceipt",
    "GlobalAffineCalibratorReceipt",
    "GlobalAffineSOZCalibrator",
    "ReasonerCalibrationData",
    "build_reasoner_calibration_data",
    "fit_global_affine_calibrator",
    "freeze_reasoner_checkpoint",
    "reasoner_state_sha256",
    "uncalibrated_training_probabilities",
]
