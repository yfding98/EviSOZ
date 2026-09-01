"""Formal target-free fitting pipeline for direct temporal-evolution scalers.

One invocation fits exactly one OOF selection (fold 0--4 or final).  It loads
both the final-plan master TUSZ manifest and the selected training manifest
against caller-held SHA-256 anchors, replays every selected event through the
frozen causal EDF loader, computes the six direct descriptors, and publishes
one fold-bound scaler artifact.  Event loss, source drift, receipt drift, and
selection/manifest mismatch fail before the formal output name is published.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from typing import Callable

import numpy as np
import scipy
import torch

from . import evolution as _evolution_source_module
from . import geometry as _geometry_source_module
from .concept_oof import IctalConceptOOFPlan, IctalConceptOOFProtocol
from .data.edf import LoadedEDFEvent, load_standard19_edf_event
from .data.tusz_training import (
    TUSZIctalEventRecord,
    TUSZIctalTrainingManifest,
    derive_tusz_ictal_training_manifest,
    load_tusz_ictal_training_manifest,
    parse_tusz_official_train_path,
    tusz_signal_preflight_receipt_sha256,
)
from .evolution import (
    COMPLETE19_DESCRIPTOR_MASK_SHA256,
    compute_temporal_evolution_descriptors,
    fit_patient_balanced_robust_scaler,
)
from .evolution_io import (
    ComputedEvolutionFitResult,
    ComputedEvolutionEventReceipt,
    EvolutionComputationReceipt,
    SavedComputedEvolutionScalerArtifact,
    VerifiedComputedEvolutionFitResult,
    _build_computed_evolution_scaler_artifact_core_from_manifest,
    _issue_verified_evolution_fit_result,
    _VERIFIED_FIT_ISSUER_TOKEN,
    build_computed_evolution_scaler_artifact,
    load_computed_evolution_scaler_artifact,
    save_computed_evolution_scaler_artifact,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_ENVELOPE_BYTES = 64 * 1024
_MAX_MANIFEST_RECEIPT_BYTES = 128 * 1024 * 1024

ReaderFactory = Callable[[str], object]


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor hash input must be a torch.Tensor")
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _reject_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    return absolute


def _read_stable_regular_file(
    path: Path, *, field: str, max_bytes: int
) -> tuple[bytes, str]:
    source = _reject_symlink_components(path, field=field)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{field} must be a regular file")
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


def _module_source_sha256(module: object, *, field: str) -> str:
    source_value = getattr(module, "__file__", None)
    if not isinstance(source_value, str) or not source_value:
        raise RuntimeError(f"{field} has no auditable source file")
    _, digest = _read_stable_regular_file(
        Path(source_value), field=field, max_bytes=16 * 1024 * 1024
    )
    return digest


def build_evolution_computation_receipt() -> EvolutionComputationReceipt:
    """Bind exact implementation sources, runtime libraries, and CPU policy."""

    return EvolutionComputationReceipt(
        evolution_source_sha256=_module_source_sha256(
            _evolution_source_module, field="evolution.py source"
        ),
        evolution_fit_source_sha256=_module_source_sha256(
            sys.modules[__name__], field="evolution_fit.py source"
        ),
        geometry_source_sha256=_module_source_sha256(
            _geometry_source_module, field="geometry.py source"
        ),
        torch_version=str(torch.__version__),
        numpy_version=str(np.__version__),
        scipy_version=str(scipy.__version__),
        platform_machine=platform.machine() or "unknown_machine",
        torch_num_threads=int(torch.get_num_threads()),
        torch_num_interop_threads=int(torch.get_num_interop_threads()),
    )


@dataclass(frozen=True)
class BoundTUSZManifest:
    """A strictly loaded manifest and the two exact persisted-file hashes."""

    manifest: TUSZIctalTrainingManifest
    bundle_manifest_sha256: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TUSZIctalTrainingManifest):
            raise TypeError("manifest must be a TUSZIctalTrainingManifest")
        for field in ("bundle_manifest_sha256", "source_manifest_sha256"):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.manifest.manifest_sha256 != self.source_manifest_sha256:
            raise ValueError("Bound manifest source SHA disagrees with reconstruction")


def load_bound_tusz_manifest(
    bundle_directory: str | Path,
    *,
    expected_bundle_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    field: str,
) -> BoundTUSZManifest:
    """Load one manifest only when both persisted hashes match trust anchors."""

    bundle = _reject_symlink_components(Path(bundle_directory), field=field)
    _, bundle_sha = _read_stable_regular_file(
        bundle / "manifest.json",
        field=f"{field} manifest.json",
        max_bytes=_MAX_MANIFEST_ENVELOPE_BYTES,
    )
    _, source_sha = _read_stable_regular_file(
        bundle / "receipt.json",
        field=f"{field} receipt.json",
        max_bytes=_MAX_MANIFEST_RECEIPT_BYTES,
    )
    expected_bundle = _require_sha256(
        expected_bundle_manifest_sha256,
        field=f"expected_{field}_bundle_manifest_sha256",
    )
    expected_source = _require_sha256(
        expected_source_manifest_sha256,
        field=f"expected_{field}_source_manifest_sha256",
    )
    if bundle_sha != expected_bundle:
        raise ValueError(f"{field} bundle manifest SHA256 mismatch")
    if source_sha != expected_source:
        raise ValueError(f"{field} source manifest SHA256 mismatch")
    manifest = load_tusz_ictal_training_manifest(
        bundle,
        expected_bundle_manifest_sha256=bundle_sha,
        expected_source_manifest_sha256=source_sha,
    )
    return BoundTUSZManifest(
        manifest=manifest,
        bundle_manifest_sha256=bundle_sha,
        source_manifest_sha256=source_sha,
    )


def _require_fully_preflighted(
    manifest: TUSZIctalTrainingManifest, *, field: str
) -> None:
    if not manifest.preflight_performed:
        raise ValueError(f"{field} must have preflight_performed=True")
    missing = tuple(
        event.event_id
        for event in manifest
        if event.signal_preflight_receipt_sha256 is None
    )
    if missing:
        raise ValueError(
            f"{field} lacks signal-preflight receipts for {missing[:5]}"
        )


def _selection_plan(
    protocol: IctalConceptOOFProtocol, oof_fold: int | None
) -> IctalConceptOOFPlan:
    if not isinstance(protocol, IctalConceptOOFProtocol):
        raise TypeError("oof_protocol must be an IctalConceptOOFProtocol")
    if oof_fold is None:
        return protocol.final_plan
    if isinstance(oof_fold, bool) or not isinstance(oof_fold, int):
        raise ValueError("oof_fold must be an integer in [0,4] or None for final")
    return protocol.for_fold(oof_fold)


def validate_evolution_fit_selection(
    master: TUSZIctalTrainingManifest,
    training: TUSZIctalTrainingManifest,
    *,
    oof_fold: int | None,
    oof_protocol: IctalConceptOOFProtocol,
) -> IctalConceptOOFPlan:
    """Prove that a selected manifest is the exact fold/final fit source."""

    if not isinstance(master, TUSZIctalTrainingManifest) or not isinstance(
        training, TUSZIctalTrainingManifest
    ):
        raise TypeError("master and training must be TUSZ training manifests")
    _require_fully_preflighted(master, field="master manifest")
    _require_fully_preflighted(training, field="training manifest")
    plan = _selection_plan(oof_protocol, oof_fold)
    final_plan = oof_protocol.final_plan

    if master.cohort_receipt.receipt_sha256 != (
        final_plan.training_cohort.receipt.receipt_sha256
    ):
        raise ValueError("Master manifest does not bind the protocol final plan")
    if master.authorized_source_record_sha256s != (
        final_plan.receipt.authorized_record_sha256s
    ):
        raise ValueError("Master authorized source roster differs from final plan")
    if master.derived_from_manifest_sha256 is not None:
        raise ValueError("Master manifest cannot itself be a derived fold manifest")
    if training.preprocess_config != master.preprocess_config:
        raise ValueError("Training preprocessing differs from its master")
    if training.cohort_receipt.receipt_sha256 != (
        plan.training_cohort.receipt.receipt_sha256
    ):
        raise ValueError("Training manifest belongs to a different OOF selection")
    if training.authorized_source_record_sha256s != (
        plan.receipt.authorized_record_sha256s
    ):
        raise ValueError("Training authorized source roster differs from OOF plan")

    if oof_fold is None:
        if training != master or training.manifest_sha256 != master.manifest_sha256:
            raise ValueError("Final evolution scaler must fit the exact master manifest")
    else:
        if training.derived_from_manifest_sha256 != master.manifest_sha256:
            raise ValueError("Fold manifest is not derived from the supplied master")
        if training.discovered_source_count != master.discovered_source_count:
            raise ValueError("Fold discovery count differs from its master")
        if training.duplicate_edf_aliases != master.duplicate_edf_aliases:
            raise ValueError("Fold duplicate-EDF audit differs from its master")
        expected_training = derive_tusz_ictal_training_manifest(
            master, plan.training_cohort
        )
        if training != expected_training:
            raise ValueError(
                "Fold manifest is not the exact attrition-free derivation from master"
            )
        leaked = tuple(
            sorted(set(training.patient_ids) & set(plan.held_out_public_patient_keys))
        )
        if leaked:
            raise ValueError(f"Fold fit manifest contains held-out patients: {leaked}")

    return plan


def _event_source(
    edf_root: str | Path, event: TUSZIctalEventRecord
) -> Path:
    source = parse_tusz_official_train_path(edf_root, event.relative_edf_path)
    checks = {
        "patient_id": source.patient_id == event.patient_id,
        "session_id": source.session_id == event.session_id,
        "montage": source.montage == event.montage,
        "record_id": source.record_id == event.record_id,
        "relative_edf_path": source.relative_edf_path == event.relative_edf_path,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            f"Frozen event {event.event_id} changed source identity fields {failed}"
        )
    return source.edf_path


def _replay_event_signal(
    event: TUSZIctalEventRecord,
    *,
    manifest: TUSZIctalTrainingManifest,
    edf_root: str | Path,
    reader_factory: ReaderFactory | None,
) -> LoadedEDFEvent:
    source = _event_source(edf_root, event)
    loaded = load_standard19_edf_event(
        source,
        event.event_t0_sec,
        config=manifest.preprocess_config,
        reader_factory=reader_factory,
    )
    if loaded.edf_receipt.edf_sha256 != event.edf_sha256:
        raise ValueError(f"Frozen EDF changed after manifest: {event.event_id}")
    replay_sha = tusz_signal_preflight_receipt_sha256(loaded)
    if replay_sha != event.signal_preflight_receipt_sha256:
        raise ValueError(
            f"Signal preflight receipt changed for event {event.event_id}"
        )
    eeg = loaded.window.data
    if (
        tuple(eeg.shape) != (19, 12_000)
        or eeg.dtype != torch.float32
        or not torch.isfinite(eeg).all().item()
    ):
        raise ValueError("Replayed TUSZ event must be finite CPU-compatible [19,12000]")
    return loaded


def _execute_full_raw_edf_replay_fit(
    manifest: TUSZIctalTrainingManifest,
    *,
    edf_root: str | Path,
    split_manifest_sha256: str,
    device: str | torch.device = "cpu",
    reader_factory: ReaderFactory | None = None,
) -> ComputedEvolutionFitResult:
    """Internal implementation shared by two distinct full replay passes."""

    if not isinstance(manifest, TUSZIctalTrainingManifest):
        raise TypeError("manifest must be a TUSZIctalTrainingManifest")
    _require_fully_preflighted(manifest, field="training manifest")
    split_sha = _require_sha256(
        split_manifest_sha256, field="split_manifest_sha256"
    )
    execution_device = torch.device(device)
    if execution_device.type != "cpu" or execution_device.index is not None:
        raise ValueError("Primary evolution fitting is frozen to CPU-only float64")
    computation_receipt = build_evolution_computation_receipt()

    descriptors_by_patient: dict[str, list[torch.Tensor]] = {
        patient_id: [] for patient_id in manifest.patient_ids
    }
    masks_by_patient: dict[str, list[torch.Tensor]] = {
        patient_id: [] for patient_id in manifest.patient_ids
    }
    event_receipts: list[ComputedEvolutionEventReceipt] = []
    replayed_event_ids: list[str] = []
    for event in manifest:
        loaded = _replay_event_signal(
            event,
            manifest=manifest,
            edf_root=edf_root,
            reader_factory=reader_factory,
        )
        eeg_cpu = loaded.window.data.detach().to(dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            computed = compute_temporal_evolution_descriptors(
                eeg_cpu.unsqueeze(0)
            )
        descriptors = computed.descriptors[0]
        mask = computed.mask[0].to(dtype=torch.bool, device="cpu")
        if not mask.all().item():
            raise RuntimeError(
                "Complete physical standard-19 replay produced an unavailable tile"
            )
        if _canonical_tensor_sha256(mask) != COMPLETE19_DESCRIPTOR_MASK_SHA256:
            raise RuntimeError("Complete19 descriptor mask SHA drifted")
        descriptors_by_patient[event.patient_id].append(descriptors)
        masks_by_patient[event.patient_id].append(mask)
        event_receipts.append(
            ComputedEvolutionEventReceipt(
                event_id=event.event_id,
                patient_id=event.patient_id,
                event_record_sha256=event.event_record_sha256,
                edf_sha256=event.edf_sha256,
                signal_content_sha256=event.signal_content_sha256,
                signal_preflight_receipt_sha256=(
                    event.signal_preflight_receipt_sha256
                ),
                signal_window_sha256=_canonical_tensor_sha256(eeg_cpu),
                raw_descriptor_sha256=_canonical_tensor_sha256(descriptors),
                descriptor_mask_sha256=_canonical_tensor_sha256(mask),
            )
        )
        replayed_event_ids.append(event.event_id)

    expected_event_ids = tuple(event.event_id for event in manifest)
    if tuple(replayed_event_ids) != expected_event_ids:
        raise RuntimeError("Evolution replay omitted or reordered a manifest event")
    if len(event_receipts) != len(manifest):
        raise RuntimeError("Evolution replay attrition is forbidden")
    missing_patients = tuple(
        patient_id
        for patient_id in manifest.patient_ids
        if not descriptors_by_patient[patient_id]
    )
    if missing_patients:
        raise RuntimeError(
            f"Evolution replay omitted complete fit patients: {missing_patients}"
        )

    patient_descriptors = [
        torch.stack(descriptors_by_patient[patient_id], dim=0)
        for patient_id in manifest.patient_ids
    ]
    patient_masks = [
        torch.stack(masks_by_patient[patient_id], dim=0)
        for patient_id in manifest.patient_ids
    ]
    scaler = fit_patient_balanced_robust_scaler(
        patient_descriptors,
        patient_masks,
        manifest.patient_ids,
        expected_patient_ids=manifest.patient_ids,
        fit_split="source_train",
        split_manifest_sha256=split_sha,
    )
    canonical_receipts = tuple(
        sorted(event_receipts, key=lambda value: (value.patient_id, value.event_id))
    )
    if build_evolution_computation_receipt() != computation_receipt:
        raise RuntimeError("Evolution implementation/runtime changed during replay")
    return ComputedEvolutionFitResult(
        scaler=scaler,
        event_receipts=canonical_receipts,
        computation_receipt=computation_receipt,
    )


def replay_and_fit_evolution_scaler(
    manifest: TUSZIctalTrainingManifest,
    *,
    edf_root: str | Path,
    split_manifest_sha256: str,
    device: str | torch.device = "cpu",
    reader_factory: ReaderFactory | None = None,
) -> ComputedEvolutionFitResult:
    """Candidate pass: replay every event and fit one exact scaler."""

    return _execute_full_raw_edf_replay_fit(
        manifest,
        edf_root=edf_root,
        split_manifest_sha256=split_manifest_sha256,
        device=device,
        reader_factory=reader_factory,
    )


def _independent_replay_and_fit_evolution_scaler(
    manifest: TUSZIctalTrainingManifest,
    *,
    edf_root: str | Path,
    split_manifest_sha256: str,
    reader_factory: ReaderFactory | None,
) -> ComputedEvolutionFitResult:
    """Verifier pass deliberately bypassing the public candidate producer."""

    return _execute_full_raw_edf_replay_fit(
        manifest,
        edf_root=edf_root,
        split_manifest_sha256=split_manifest_sha256,
        device="cpu",
        reader_factory=reader_factory,
    )


def verify_evolution_fit_result_by_independent_replay(
    candidate: ComputedEvolutionFitResult,
    *,
    master_manifest_bundle: str | Path,
    expected_master_bundle_manifest_sha256: str,
    expected_master_source_manifest_sha256: str,
    training_manifest_bundle: str | Path,
    expected_training_bundle_manifest_sha256: str,
    expected_training_source_manifest_sha256: str,
    oof_fold: int | None,
    oof_protocol: IctalConceptOOFProtocol,
    edf_root: str | Path,
    reader_factory: ReaderFactory | None = None,
) -> VerifiedComputedEvolutionFitResult:
    """Reload raw inputs, perform a distinct full replay, and issue proof."""

    if not isinstance(candidate, ComputedEvolutionFitResult):
        raise TypeError("candidate must be a complete ComputedEvolutionFitResult")
    verified_master = load_bound_tusz_manifest(
        master_manifest_bundle,
        expected_bundle_manifest_sha256=(
            expected_master_bundle_manifest_sha256
        ),
        expected_source_manifest_sha256=(
            expected_master_source_manifest_sha256
        ),
        field="verifier_master_manifest",
    )
    verified_training = load_bound_tusz_manifest(
        training_manifest_bundle,
        expected_bundle_manifest_sha256=(
            expected_training_bundle_manifest_sha256
        ),
        expected_source_manifest_sha256=(
            expected_training_source_manifest_sha256
        ),
        field="verifier_training_manifest",
    )
    validate_evolution_fit_selection(
        verified_master.manifest,
        verified_training.manifest,
        oof_fold=oof_fold,
        oof_protocol=oof_protocol,
    )
    independent = _independent_replay_and_fit_evolution_scaler(
        verified_training.manifest,
        edf_root=edf_root,
        split_manifest_sha256=oof_protocol.receipt.split_manifest_sha256,
        reader_factory=reader_factory,
    )
    if candidate.event_receipts != independent.event_receipts:
        raise ValueError("Independent raw-EDF replay changed event receipts")
    if candidate.scaler.receipt != independent.scaler.receipt:
        raise ValueError("Independent raw-EDF replay changed scaler receipt")

    candidate_components = (
        _build_computed_evolution_scaler_artifact_core_from_manifest(
            candidate,
            oof_fold=oof_fold,
            fit_public_patient_keys=verified_training.manifest.patient_ids,
            fit_manifest=verified_training.manifest,
            fit_manifest_bundle_sha256=(
                verified_training.bundle_manifest_sha256
            ),
            oof_protocol=oof_protocol,
        )
    )
    independent_components = (
        _build_computed_evolution_scaler_artifact_core_from_manifest(
            independent,
            oof_fold=oof_fold,
            fit_public_patient_keys=verified_training.manifest.patient_ids,
            fit_manifest=verified_training.manifest,
            fit_manifest_bundle_sha256=(
                verified_training.bundle_manifest_sha256
            ),
            oof_protocol=oof_protocol,
        )
    )
    candidate_core = candidate_components[3]
    independent_core = independent_components[3]
    if candidate_core != independent_core:
        raise ValueError(
            "Independent raw-EDF replay changed canonical artifact core"
        )
    return _issue_verified_evolution_fit_result(
        candidate,
        independent,
        canonical_artifact_core_sha256=hashlib.sha256(
            json.dumps(
                candidate_core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        issuer_token=_VERIFIED_FIT_ISSUER_TOKEN,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fit_and_publish_tusz_evolution_scaler(
    *,
    master_manifest_bundle: str | Path,
    expected_master_bundle_manifest_sha256: str,
    expected_master_source_manifest_sha256: str,
    training_manifest_bundle: str | Path,
    expected_training_bundle_manifest_sha256: str,
    expected_training_source_manifest_sha256: str,
    oof_fold: int | None,
    oof_protocol: IctalConceptOOFProtocol,
    edf_root: str | Path,
    output_directory: str | Path,
    device: str | torch.device = "cpu",
    reader_factory: ReaderFactory | None = None,
) -> SavedComputedEvolutionScalerArtifact:
    """Fit and atomically publish one strict fold/final evolution scaler."""

    target = _reject_symlink_components(
        Path(output_directory), field="evolution scaler output"
    )
    if target.name in {"", ".", ".."}:
        raise ValueError("Evolution scaler output requires a concrete directory")
    if os.path.lexists(target):
        raise FileExistsError(f"Evolution scaler output already exists: {target}")
    parent = _reject_symlink_components(target.parent, field="output parent")
    if not parent.is_dir():
        raise FileNotFoundError("Evolution scaler output parent does not exist")

    bound_master = load_bound_tusz_manifest(
        master_manifest_bundle,
        expected_bundle_manifest_sha256=(
            expected_master_bundle_manifest_sha256
        ),
        expected_source_manifest_sha256=(
            expected_master_source_manifest_sha256
        ),
        field="master_manifest",
    )
    bound_training = load_bound_tusz_manifest(
        training_manifest_bundle,
        expected_bundle_manifest_sha256=(
            expected_training_bundle_manifest_sha256
        ),
        expected_source_manifest_sha256=(
            expected_training_source_manifest_sha256
        ),
        field="training_manifest",
    )
    validate_evolution_fit_selection(
        bound_master.manifest,
        bound_training.manifest,
        oof_fold=oof_fold,
        oof_protocol=oof_protocol,
    )
    result = replay_and_fit_evolution_scaler(
        bound_training.manifest,
        edf_root=edf_root,
        split_manifest_sha256=oof_protocol.receipt.split_manifest_sha256,
        device=device,
        reader_factory=reader_factory,
    )
    verified_result = verify_evolution_fit_result_by_independent_replay(
        result,
        master_manifest_bundle=master_manifest_bundle,
        expected_master_bundle_manifest_sha256=(
            expected_master_bundle_manifest_sha256
        ),
        expected_master_source_manifest_sha256=(
            expected_master_source_manifest_sha256
        ),
        training_manifest_bundle=training_manifest_bundle,
        expected_training_bundle_manifest_sha256=(
            expected_training_bundle_manifest_sha256
        ),
        expected_training_source_manifest_sha256=(
            expected_training_source_manifest_sha256
        ),
        oof_fold=oof_fold,
        oof_protocol=oof_protocol,
        edf_root=edf_root,
        reader_factory=reader_factory,
    )
    artifact = build_computed_evolution_scaler_artifact(
        verified_result,
        oof_fold=oof_fold,
        fit_manifest_bundle_directory=training_manifest_bundle,
        oof_protocol=oof_protocol,
    )
    if artifact.receipt.fit_manifest_bundle_sha256 != (
        bound_training.bundle_manifest_sha256
    ):
        raise ValueError("Training manifest bundle changed during scaler fitting")
    if artifact.receipt.fit_manifest_source_sha256 != (
        bound_training.source_manifest_sha256
    ):
        raise ValueError("Training manifest source changed during scaler fitting")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    staged_bundle = staging / "artifact"
    published = False
    try:
        saved = save_computed_evolution_scaler_artifact(artifact, staged_bundle)
        load_computed_evolution_scaler_artifact(
            staged_bundle,
            oof_fold=oof_fold,
            fit_manifest_bundle_directory=training_manifest_bundle,
            oof_protocol=oof_protocol,
            expected_fit_public_patient_keys=bound_training.manifest.patient_ids,
            expected_artifact_sha256=saved.artifact_sha256,
            expected_scaler=verified_result.fit_result.scaler,
        )
        _fsync_directory(staging)
        if os.path.lexists(target):
            raise FileExistsError(
                f"Evolution scaler output already exists: {target}"
            )
        os.rename(staged_bundle, target)
        published = True
        _fsync_directory(parent)
        staging.rmdir()
        return SavedComputedEvolutionScalerArtifact(
            path=target,
            artifact_sha256=saved.artifact_sha256,
            artifact_receipt_sha256=saved.artifact_receipt_sha256,
            scaler_receipt_sha256=saved.scaler_receipt_sha256,
            verification_receipt_sha256=(
                saved.verification_receipt_sha256
            ),
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if not published and os.path.lexists(target):
            # ``target`` can only appear after the single atomic rename above.
            # A post-rename failure is surfaced rather than silently removing a
            # potentially valid formal artifact.
            pass


__all__ = [
    "BoundTUSZManifest",
    "build_evolution_computation_receipt",
    "fit_and_publish_tusz_evolution_scaler",
    "load_bound_tusz_manifest",
    "replay_and_fit_evolution_scaler",
    "validate_evolution_fit_selection",
    "verify_evolution_fit_result_by_independent_replay",
]
