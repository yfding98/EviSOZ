"""Frozen CPU-friendly training and atomic artifact for the I+V candidate.

Source-development evidence is evaluated exactly once after the twentieth
epoch and never participates in optimization, checkpoint selection,
calibration, or threshold selection.  This remains a development artifact and
cannot be loaded by the formal reasoner pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Mapping

import torch

from .development_reasoner import (
    DEVELOPMENT_IV_EXPLANATION_MODE,
    DevelopmentIVAdditiveReasoner,
    DevelopmentReasonerStepOutput,
    VerifiedDevelopmentReasonerDataBundle,
    _canonical_json_bytes,
    _canonical_sha256,
    _file_sha256,
    _fsync_directory,
    _fsync_file,
    _read_capability_file,
    _require_sha256,
    _safe_new_directory,
    _strict_json,
    _tensor_sha256,
    aggregate_numeric_explanations,
    development_reasoner_step,
)
from .formal_reasoner_pipeline import (
    FORMAL_REASONER_FIT_POLICY_SHA256,
    FormalReasonerFitConfig,
)


DEVELOPMENT_REASONER_TRAINING_SCHEMA = "soz_development_iv_training_run_v1"
DEVELOPMENT_REASONER_ARTIFACT_SCHEMA = "soz_development_iv_training_artifact_v1"
DEVELOPMENT_REASONER_MANIFEST_FILENAME = "manifest.json"
DEVELOPMENT_REASONER_CHECKPOINT_FILENAME = "checkpoint.safetensors"
_TRAINING_FILE_SET = {
    DEVELOPMENT_REASONER_MANIFEST_FILENAME,
    DEVELOPMENT_REASONER_CHECKPOINT_FILENAME,
}
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_RUN_MARKER = object()


_DEVELOPMENT_TRAINING_POLICY = {
    "schema_version": DEVELOPMENT_REASONER_TRAINING_SCHEMA,
    "optimizer_policy_sha256": FORMAL_REASONER_FIT_POLICY_SHA256,
    "checkpoint_selection": "final_epoch_20_only",
    "source_dev_access": "one_post_freeze_diagnostic_forward_only",
    "source_dev_used_for_optimization": False,
    "source_dev_used_for_checkpoint_selection": False,
    "source_dev_used_for_threshold_selection": False,
    "threshold_selection": "forbidden",
    "calibration": "not_fitted_in_candidate_training",
    "source_eval_allowed": False,
    "private_allowed": False,
    "formal_promotion": False,
}
DEVELOPMENT_REASONER_TRAINING_POLICY_SHA256 = _canonical_sha256(
    _DEVELOPMENT_TRAINING_POLICY
)


def development_reasoner_state_sha256(
    model_or_state: DevelopmentIVAdditiveReasoner | Mapping[str, torch.Tensor],
) -> str:
    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, DevelopmentIVAdditiveReasoner)
        else model_or_state
    )
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Development reasoner state must be a non-empty mapping")
    return _canonical_sha256(
        {
            str(name): _tensor_sha256(str(name), value)
            for name, value in sorted(state.items())
        }
    )


def _patient_order(
    patient_ids: tuple[str, ...], *, seed: int, epoch: int
) -> tuple[str, ...]:
    order = list(patient_ids)
    random.Random((seed << 20) ^ epoch).shuffle(order)
    return tuple(order)


@dataclass(frozen=True)
class DevelopmentReasonerEpochReceipt:
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
        _require_sha256(self.patient_order_sha256, field_name="patient_order_sha256")
        for name in ("mean_total_loss", "mean_bce_loss", "mean_ranking_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.patient_count < 1 or self.event_count < 1:
            raise ValueError("Every training epoch requires patients and events")


@dataclass(frozen=True)
class DevelopmentReasonerDiagnosticReceipt:
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

    def __post_init__(self) -> None:
        if self.model_split not in {"source_train", "source_dev"}:
            raise ValueError("Candidate diagnostics reject source_eval/private")
        for name in ("total_loss", "bce_loss", "ranking_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.patient_count < 1 or self.event_count < 1:
            raise ValueError("Diagnostic receipt requires patients and events")
        if not 0 <= self.patient_abstain_recommended_count <= self.patient_count:
            raise ValueError("Patient abstention count is invalid")
        _require_sha256(self.patient_logits_sha256, field_name="patient_logits_sha256")
        _require_sha256(self.target_mask_sha256, field_name="target_mask_sha256")
        _require_sha256(
            self.numeric_explanation_receipt_sha256,
            field_name="numeric_explanation_receipt_sha256",
        )


@dataclass(frozen=True)
class DevelopmentReasonerTrainingReceipt:
    evidence_authorization_sha256: str
    verified_target_v2_receipt_sha256: str
    source_train_dataset_sha256: str
    source_dev_dataset_sha256: str
    config: FormalReasonerFitConfig
    config_sha256: str
    training_policy_sha256: str
    initial_state_sha256: str
    final_state_sha256: str
    parameter_count: int
    epochs: tuple[DevelopmentReasonerEpochReceipt, ...]
    final_source_train_diagnostic: DevelopmentReasonerDiagnosticReceipt
    final_source_dev_diagnostic: DevelopmentReasonerDiagnosticReceipt
    source_dev_forward_pass_count: int = 1
    checkpoint_selection: str = "final_epoch_20_only"
    threshold_selected: bool = False
    calibrator_fitted: bool = False
    formal_promotion: bool = False
    formal_reasoner_authorized: bool = False
    source_eval_used: bool = False
    private_used: bool = False
    schema_version: str = DEVELOPMENT_REASONER_TRAINING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "evidence_authorization_sha256",
            "verified_target_v2_receipt_sha256",
            "source_train_dataset_sha256",
            "source_dev_dataset_sha256",
            "config_sha256",
            "training_policy_sha256",
            "initial_state_sha256",
            "final_state_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), field_name=name)
            )
        if not isinstance(self.config, FormalReasonerFitConfig):
            raise TypeError("Candidate training must use FormalReasonerFitConfig")
        if self.config_sha256 != FORMAL_REASONER_FIT_POLICY_SHA256 or self.config.receipt_sha256 != self.config_sha256:
            raise ValueError("Candidate optimizer schedule differs from the frozen formal schedule")
        if self.training_policy_sha256 != DEVELOPMENT_REASONER_TRAINING_POLICY_SHA256:
            raise ValueError("Candidate training policy digest drifted")
        if len(self.epochs) != 20 or tuple(row.epoch_index for row in self.epochs) != tuple(range(20)):
            raise ValueError("Candidate training requires exactly twenty complete epochs")
        if self.parameter_count < 1 or self.parameter_count >= 50_000:
            raise ValueError("Candidate reasoner violates the capacity gate")
        if self.source_dev_forward_pass_count != 1:
            raise ValueError("Source-dev must be evaluated exactly once after training")
        if self.checkpoint_selection != "final_epoch_20_only":
            raise ValueError("Candidate checkpoint selection policy changed")
        if any(
            (
                self.threshold_selected,
                self.calibrator_fitted,
                self.formal_promotion,
                self.formal_reasoner_authorized,
                self.source_eval_used,
                self.private_used,
            )
        ):
            raise ValueError("Candidate training escaped its development-only boundary")
        if self.schema_version != DEVELOPMENT_REASONER_TRAINING_SCHEMA:
            raise ValueError("Unsupported candidate training receipt schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class VerifiedDevelopmentReasonerTrainingRun:
    model: DevelopmentIVAdditiveReasoner = field(repr=False)
    receipt: DevelopmentReasonerTrainingReceipt

    def __init__(
        self,
        *,
        _verification_marker: object,
        model: DevelopmentIVAdditiveReasoner,
        receipt: DevelopmentReasonerTrainingReceipt,
    ) -> None:
        if _verification_marker is not _RUN_MARKER:
            raise TypeError("Candidate training run requires the closed trainer")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "receipt", receipt)
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        if self.model.training:
            raise ValueError("Completed candidate reasoner must remain in eval mode")
        if development_reasoner_state_sha256(self.model) != self.receipt.final_state_sha256:
            raise ValueError("Candidate reasoner state changed after training")


def _diagnostic(
    model: DevelopmentIVAdditiveReasoner,
    bundle: VerifiedDevelopmentReasonerDataBundle,
    *,
    model_split: str,
    device: torch.device,
) -> DevelopmentReasonerDiagnosticReceipt:
    dataset = bundle.source_train if model_split == "source_train" else bundle.source_dev
    batch = dataset.full_batch().to(device)
    model.eval()
    with torch.no_grad():
        step = development_reasoner_step(model, batch)
        explanation = aggregate_numeric_explanations(
            step.reasoner, batch.event_patient_index
        )
    if not torch.allclose(
        explanation.patient_logits, step.patient_logits, atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Final numerical explanation changed patient logits")
    explanation_receipt = _canonical_sha256(
        {
            "explanation_mode": explanation.explanation_mode,
            "llm_used_for_prediction": explanation.llm_used_for_prediction,
            "patient_logits_sha256": _tensor_sha256(
                f"{model_split}_explained_patient_logits",
                explanation.patient_logits.detach().cpu(),
            ),
            "components": {
                name: _tensor_sha256(
                    f"{model_split}_explanation_{name}", value.detach().cpu()
                )
                for name, value in sorted(
                    explanation.component_contributions.items()
                )
            },
        }
    )
    return DevelopmentReasonerDiagnosticReceipt(
        model_split=model_split,
        total_loss=float(step.loss.total.detach().cpu()),
        bce_loss=float(step.loss.bce.detach().cpu()),
        ranking_loss=float(step.loss.ranking.detach().cpu()),
        patient_count=len(batch.patient_ids),
        event_count=int(step.event_counts.sum().item()),
        patient_abstain_recommended_count=int(
            step.patient_abstain_recommended.sum().item()
        ),
        patient_logits_sha256=_tensor_sha256(
            f"{model_split}_final_patient_logits",
            step.patient_logits.detach().cpu(),
        ),
        target_mask_sha256=_tensor_sha256(
            f"{model_split}_target_mask", batch.target_mask.detach().cpu()
        ),
        numeric_explanation_receipt_sha256=explanation_receipt,
    )


def train_development_iv_reasoner(
    data_bundle: VerifiedDevelopmentReasonerDataBundle,
    *,
    device: str | torch.device = "cpu",
) -> VerifiedDevelopmentReasonerTrainingRun:
    """Run the unique twenty-epoch candidate schedule; no selection knobs."""

    if type(data_bundle) is not VerifiedDevelopmentReasonerDataBundle:
        raise TypeError("Candidate training requires the strict target-joined bundle")
    data_bundle.assert_unchanged()
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("Candidate training device must be cpu or cuda")
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
        model = DevelopmentIVAdditiveReasoner(hidden_dim=config.hidden_dim).to(
            execution_device
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial_state = development_reasoner_state_sha256(model)
    epoch_rows: list[DevelopmentReasonerEpochReceipt] = []
    for epoch in range(config.epochs):
        data_bundle.assert_unchanged()
        order = _patient_order(
            data_bundle.source_train.patient_ids, seed=config.seed, epoch=epoch
        )
        totals: list[float] = []
        bces: list[float] = []
        rankings: list[float] = []
        event_count = 0
        model.train()
        for raw_batch in data_bundle.source_train.iter_epoch(order):
            batch = raw_batch.to(execution_device)
            optimizer.zero_grad(set_to_none=True)
            step: DevelopmentReasonerStepOutput = development_reasoner_step(model, batch)
            step.loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            totals.append(float(step.loss.total.detach().cpu()))
            bces.append(float(step.loss.bce.detach().cpu()))
            rankings.append(float(step.loss.ranking.detach().cpu()))
            event_count += int(step.event_counts.sum().item())
        if len(totals) != len(data_bundle.source_train.patient_ids):
            raise RuntimeError("Candidate epoch did not visit every patient once")
        epoch_rows.append(
            DevelopmentReasonerEpochReceipt(
                epoch_index=epoch,
                patient_order_sha256=_canonical_sha256(order),
                mean_total_loss=sum(totals) / len(totals),
                mean_bce_loss=sum(bces) / len(bces),
                mean_ranking_loss=sum(rankings) / len(rankings),
                patient_count=len(totals),
                event_count=event_count,
            )
        )
    model.eval()
    final_state = development_reasoner_state_sha256(model)
    # Both diagnostics occur only after the final state has been fixed.  The
    # source-dev forward is exactly once and has no path back to optimizer or
    # checkpoint selection.
    train_diagnostic = _diagnostic(
        model, data_bundle, model_split="source_train", device=execution_device
    )
    dev_diagnostic = _diagnostic(
        model, data_bundle, model_split="source_dev", device=execution_device
    )
    model.eval()
    data_bundle.assert_unchanged()
    receipt = DevelopmentReasonerTrainingReceipt(
        evidence_authorization_sha256=data_bundle.evidence_authorization_sha256,
        verified_target_v2_receipt_sha256=data_bundle.verified_target_v2_receipt_sha256,
        source_train_dataset_sha256=data_bundle.source_train.receipt_sha256,
        source_dev_dataset_sha256=data_bundle.source_dev.receipt_sha256,
        config=config,
        config_sha256=config.receipt_sha256,
        training_policy_sha256=DEVELOPMENT_REASONER_TRAINING_POLICY_SHA256,
        initial_state_sha256=initial_state,
        final_state_sha256=final_state,
        parameter_count=model.n_trainable_parameters,
        epochs=tuple(epoch_rows),
        final_source_train_diagnostic=train_diagnostic,
        final_source_dev_diagnostic=dev_diagnostic,
    )
    return VerifiedDevelopmentReasonerTrainingRun(
        _verification_marker=_RUN_MARKER, model=model, receipt=receipt
    )


@dataclass(frozen=True)
class PublishedDevelopmentReasonerTrainingArtifact:
    path: Path
    manifest_sha256: str
    training_receipt_sha256: str
    run: VerifiedDevelopmentReasonerTrainingRun = field(repr=False)


def _training_manifest(
    run: VerifiedDevelopmentReasonerTrainingRun,
    *,
    checkpoint_record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": DEVELOPMENT_REASONER_ARTIFACT_SCHEMA,
        "purpose": "labram_iv_development_candidate_reasoner_only",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "training_policy": dict(_DEVELOPMENT_TRAINING_POLICY),
        "training_policy_sha256": DEVELOPMENT_REASONER_TRAINING_POLICY_SHA256,
        "training_receipt": asdict(run.receipt),
        "training_receipt_sha256": run.receipt.receipt_sha256,
        "model_schema_version": "soz_development_iv_additive_reasoner_v1",
        "parameter_count": run.receipt.parameter_count,
        "explanation_mode": DEVELOPMENT_IV_EXPLANATION_MODE,
        "checkpoint_selection": "final_epoch_20_only",
        "source_dev_forward_pass_count": 1,
        "threshold_selected": False,
        "calibrator_fitted": False,
        "formal_promotion": False,
        "formal_reasoner_authorized": False,
        "source_eval_used": False,
        "private_used": False,
        "files": {DEVELOPMENT_REASONER_CHECKPOINT_FILENAME: dict(checkpoint_record)},
    }


def publish_development_reasoner_training_run(
    run: VerifiedDevelopmentReasonerTrainingRun,
    output_directory: str | Path,
) -> PublishedDevelopmentReasonerTrainingArtifact:
    if type(run) is not VerifiedDevelopmentReasonerTrainingRun:
        raise TypeError("Only the closed candidate trainer may publish a run")
    run.assert_unchanged()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for reasoner publication") from exc
    target = _safe_new_directory(
        output_directory, field_name="Development reasoner training output"
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        checkpoint = temporary / DEVELOPMENT_REASONER_CHECKPOINT_FILENAME
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in run.model.state_dict().items()
        }
        save_file(state, str(checkpoint))
        checkpoint_record = {
            "sha256": _file_sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "state_sha256": development_reasoner_state_sha256(state),
        }
        manifest = _training_manifest(run, checkpoint_record=checkpoint_record)
        manifest_raw = _canonical_json_bytes(manifest)
        manifest_path = temporary / DEVELOPMENT_REASONER_MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_raw)
        for path in (checkpoint, manifest_path):
            _fsync_file(path)
        _fsync_directory(temporary)
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_development_reasoner_training_run(
            target,
            expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _training_receipt_from_payload(
    payload: Mapping[str, object],
) -> DevelopmentReasonerTrainingReceipt:
    values = dict(payload)
    try:
        values["config"] = FormalReasonerFitConfig(**values["config"])
        values["epochs"] = tuple(
            DevelopmentReasonerEpochReceipt(**row) for row in values["epochs"]
        )
        values["final_source_train_diagnostic"] = DevelopmentReasonerDiagnosticReceipt(
            **values["final_source_train_diagnostic"]
        )
        values["final_source_dev_diagnostic"] = DevelopmentReasonerDiagnosticReceipt(
            **values["final_source_dev_diagnostic"]
        )
        return DevelopmentReasonerTrainingReceipt(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Candidate training receipt is invalid") from exc


def load_development_reasoner_training_run(
    bundle_directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> PublishedDevelopmentReasonerTrainingArtifact:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for reasoner loading") from exc
    bundle = Path(os.path.abspath(bundle_directory))
    if bundle.is_symlink() or not bundle.is_dir() or {
        path.name for path in bundle.iterdir()
    } != _TRAINING_FILE_SET:
        raise ValueError("Candidate training artifact violates its closed file schema")
    raw = _read_capability_file(
        bundle / DEVELOPMENT_REASONER_MANIFEST_FILENAME,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        field_name="Candidate training manifest",
    )
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != _require_sha256(
        expected_manifest_sha256, field_name="expected_manifest_sha256"
    ):
        raise ValueError("Candidate training manifest SHA mismatch")
    manifest = _strict_json(raw, field_name="Candidate training manifest")
    if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != raw:
        raise ValueError("Candidate training manifest is not canonical JSON")
    fixed = {
        "schema_version": DEVELOPMENT_REASONER_ARTIFACT_SCHEMA,
        "purpose": "labram_iv_development_candidate_reasoner_only",
        "serialization": "canonical_json_plus_safetensors_no_pickle",
        "training_policy": _DEVELOPMENT_TRAINING_POLICY,
        "training_policy_sha256": DEVELOPMENT_REASONER_TRAINING_POLICY_SHA256,
        "model_schema_version": "soz_development_iv_additive_reasoner_v1",
        "explanation_mode": DEVELOPMENT_IV_EXPLANATION_MODE,
        "checkpoint_selection": "final_epoch_20_only",
        "source_dev_forward_pass_count": 1,
        "threshold_selected": False,
        "calibrator_fitted": False,
        "formal_promotion": False,
        "formal_reasoner_authorized": False,
        "source_eval_used": False,
        "private_used": False,
    }
    if any(manifest.get(name) != value for name, value in fixed.items()):
        raise ValueError("Candidate training scientific boundary changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {DEVELOPMENT_REASONER_CHECKPOINT_FILENAME}:
        raise ValueError("Candidate checkpoint file receipt changed")
    record = files[DEVELOPMENT_REASONER_CHECKPOINT_FILENAME]
    if not isinstance(record, dict) or set(record) != {
        "sha256",
        "size_bytes",
        "state_sha256",
    }:
        raise ValueError("Candidate checkpoint record schema changed")
    checkpoint = bundle / DEVELOPMENT_REASONER_CHECKPOINT_FILENAME
    checkpoint_raw = _read_capability_file(
        checkpoint,
        maximum_bytes=_MAX_CHECKPOINT_BYTES,
        field_name="Candidate checkpoint",
    )
    if len(checkpoint_raw) != record["size_bytes"] or hashlib.sha256(
        checkpoint_raw
    ).hexdigest() != record["sha256"]:
        raise ValueError("Candidate checkpoint bytes changed")
    state = load_file(str(checkpoint), device="cpu")
    if development_reasoner_state_sha256(state) != record["state_sha256"]:
        raise ValueError("Candidate checkpoint tensor state changed")
    receipt_payload = manifest.get("training_receipt")
    if not isinstance(receipt_payload, dict):
        raise ValueError("Candidate training receipt is missing")
    receipt = _training_receipt_from_payload(receipt_payload)
    if receipt.receipt_sha256 != manifest.get("training_receipt_sha256"):
        raise ValueError("Candidate training receipt SHA mismatch")
    model = DevelopmentIVAdditiveReasoner(hidden_dim=receipt.config.hidden_dim)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("Candidate checkpoint state schema changed") from exc
    model.eval()
    run = VerifiedDevelopmentReasonerTrainingRun(
        _verification_marker=_RUN_MARKER, model=model, receipt=receipt
    )
    expected_manifest = _training_manifest(run, checkpoint_record=record)
    if _canonical_json_bytes(expected_manifest) != _canonical_json_bytes(manifest):
        raise ValueError("Candidate training manifest differs from reconstructed run")
    return PublishedDevelopmentReasonerTrainingArtifact(
        path=bundle,
        manifest_sha256=manifest_sha,
        training_receipt_sha256=receipt.receipt_sha256,
        run=run,
    )


__all__ = [
    "DEVELOPMENT_REASONER_TRAINING_POLICY_SHA256",
    "DevelopmentReasonerDiagnosticReceipt",
    "DevelopmentReasonerEpochReceipt",
    "DevelopmentReasonerTrainingReceipt",
    "PublishedDevelopmentReasonerTrainingArtifact",
    "VerifiedDevelopmentReasonerTrainingRun",
    "development_reasoner_state_sha256",
    "load_development_reasoner_training_run",
    "publish_development_reasoner_training_run",
    "train_development_iv_reasoner",
]
