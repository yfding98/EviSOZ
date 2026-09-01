"""Process-separated production publishers for neutral v13.1 Stage-A inputs.

These brokers may inspect already-sealed fit/control artifacts or gate signal
tokens, but never execute a model.  Their outputs are the only artifacts read
by ``v13_neutral``.  Gate target values, annotation masks, clinical identity,
training metrics, and source paths are excluded from every projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import save_file as save_safetensors_file

from v13_neutral import control_head_projection as head_schema
from v13_neutral import control_parameters as parameter_schema
from v13_neutral import token_view as token_schema
from v13_neutral.core import (
    TOKEN_EVENT_SHAPE,
    canonical_json_bytes,
    canonical_patient_roster,
    canonical_sha256,
    patient_roster_sha256,
    require_sha256,
)
from v13_neutral.projection import EXPECTED_PRODUCER_ORDER, NeutralK31Projection
from v13_signal_masks.artifact import VerifiedSignalMaskArtifactV13_1


def _new_output(path: str | Path, *, field: str) -> Path:
    target = Path(os.path.abspath(path))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError(f"{field} requires a concrete path with an existing parent")
    for component in (target, *target.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError(f"{field} cannot contain symlink components")
    if os.path.lexists(target):
        raise FileExistsError(f"{field} already exists: {target}")
    return target


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"Output already exists: {target}")
        raise OSError(error, os.strerror(error), str(target))


def _publish_directory(
    target: Path,
    *,
    manifest: Mapping[str, object],
    files: Mapping[str, bytes],
) -> tuple[Path, str]:
    manifest_raw = canonical_json_bytes(dict(manifest))
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.broker-", dir=target.parent))
    moved = False
    try:
        for name, raw in files.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(manifest_raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        _rename_noreplace(staging, target)
        moved = True
        _fsync_directory(target.parent)
    finally:
        if not moved and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target, manifest_sha


def publish_control_parameter_projection_v13_1(
    output_directory: str | Path,
    *,
    k31_projection: NeutralK31Projection,
    fit_only_artifacts: Sequence[object],
) -> tuple[Path, str]:
    """Publish six fit-derived control bundles without any target tensor."""

    if not isinstance(k31_projection, NeutralK31Projection):
        raise TypeError("k31_projection must come from the neutral strict loader")
    fits = tuple(fit_only_artifacts)
    if len(fits) != 6:
        raise ValueError("Parameter projection requires six fit-only artifacts")
    rows = []
    for index, (candidate, fit) in enumerate(zip(k31_projection.producers, fits, strict=True)):
        manifest = getattr(fit, "manifest", None)
        if not isinstance(manifest, Mapping):
            raise TypeError("fit-only artifact lacks a strict manifest")
        selection = EXPECTED_PRODUCER_ORDER[index]
        expected_fold = None if selection == "final" else index
        fit_patients = canonical_patient_roster(manifest.get("fit_patient_ids"), field="fit_patient_ids")
        gate_patients = canonical_patient_roster(
            manifest.get("i_gate_patient_ids_excluded_unopened"),
            field="i_gate_patient_ids_excluded_unopened",
        )
        if (
            manifest.get("schema_version") != "soz_ictal_fit_only_target_artifact_v13_1"
            or manifest.get("selection") != selection
            or manifest.get("oof_fold") != expected_fold
            or manifest.get("matched_k31_checkpoint_sha256") != candidate.checkpoint_sha256
            or manifest.get("matched_k31_manifest_sha256")
            != candidate.manifest.get("legacy_recovery_manifest_sha256")
            or tuple(fit_patients) != candidate.fit_patient_ids
            or manifest.get("fit_patient_roster_sha256") != candidate.fit_patient_roster_sha256
            or tuple(gate_patients) != k31_projection.gate_patient_ids
            or manifest.get("i_gate_target_values_materialized") is not False
            or manifest.get("i_gate_outcomes_opened") is not False
        ):
            raise ValueError(f"Fit-only artifact does not match {selection} projection")
        bundle = parameter_schema.validate_parameter_bundle(
            getattr(fit, "shortcut_parameters", None)
        )
        rows.append(
            {
                "selection": selection,
                "oof_fold": expected_fold,
                "k31_projection_record_sha256": candidate.record_sha256,
                "k31_checkpoint_sha256": candidate.checkpoint_sha256,
                "fit_patient_roster_sha256": candidate.fit_patient_roster_sha256,
                "fit_patient_count": len(candidate.fit_patient_ids),
                "fit_target_projection_manifest_sha256": require_sha256(
                    getattr(fit, "manifest_sha256", None), field="fit.manifest_sha256"
                ),
                "fit_target_projection_receipt_sha256": require_sha256(
                    getattr(fit, "receipt_sha256", None), field="fit.receipt_sha256"
                ),
                "parameter_bundle_sha256": str(bundle["bundle_sha256"]),
                "parameter_bundle": bundle,
            }
        )
    manifest = {
        "schema_version": parameter_schema.SCHEMA,
        "purpose": parameter_schema.PURPOSE,
        "serialization": "single_canonical_json_no_tensor_files",
        "producer_order": list(EXPECTED_PRODUCER_ORDER),
        "producer_count": 6,
        "v5_split_sha256": k31_projection.v5_split_sha256,
        "gate_patient_roster_sha256": str(
            k31_projection.manifest["gate_patient_roster_sha256"]
        ),
        "producers": rows,
        "access_receipt": {
            "schema_version": "soz_ictal_control_parameter_projection_access_v13_1",
            "broker_fit_only_target_artifacts_loaded": True,
            "broker_fit_target_values_loaded": True,
            "broker_fit_target_masks_loaded": True,
            "broker_gate_target_values_loaded": False,
            "broker_gate_target_masks_loaded": False,
            "projection_contains_fit_target_values": False,
            "projection_contains_fit_target_masks": False,
            "projection_contains_event_level_labels": False,
            "projection_contains_source_paths": False,
            "projection_contains_source_broker_callable": False,
            "training_performed": False,
            "evaluation_performed": False,
        },
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "v13_execution_hold": True,
    }
    published = _publish_directory(
        _new_output(output_directory, field="parameter projection"),
        manifest=manifest,
        files={},
    )
    parameter_schema.load_control_parameter_projection(
        published[0], expected_manifest_sha256=published[1]
    )
    return published


def _stable_checkpoint_bytes(artifact: object, *, field: str) -> tuple[bytes, str]:
    manifest = getattr(artifact, "manifest", None)
    path = getattr(artifact, "path", None)
    if not isinstance(manifest, Mapping) or not isinstance(path, Path):
        raise TypeError(f"{field} is not a strict control artifact")
    filename = manifest.get("checkpoint_filename")
    if filename != "model.safetensors":
        raise ValueError(f"{field} checkpoint filename changed")
    source = path / filename
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError(f"{field} checkpoint is not a regular file")
    before = source.stat()
    raw = source.read_bytes()
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{field} checkpoint changed while read")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest.get("checkpoint_sha256"):
        raise ValueError(f"{field} checkpoint SHA mismatch")
    return raw, digest


def publish_control_head_projection_v13_1(
    output_directory: str | Path,
    *,
    k31_projection: NeutralK31Projection,
    capacity_artifacts: Sequence[object],
    naked_artifacts: Sequence[object],
) -> tuple[Path, str]:
    """Project twelve trained controls into inference-only safetensors."""

    if not isinstance(k31_projection, NeutralK31Projection):
        raise TypeError("k31_projection must come from the neutral strict loader")
    capacities = tuple(capacity_artifacts)
    nakeds = tuple(naked_artifacts)
    if len(capacities) != 6 or len(nakeds) != 6:
        raise ValueError("Control-head projection requires six artifacts of each kind")
    rows = []
    files: dict[str, bytes] = {}
    for index, (candidate, capacity, naked) in enumerate(
        zip(k31_projection.producers, capacities, nakeds, strict=True)
    ):
        selection = EXPECTED_PRODUCER_ORDER[index]
        fold = None if selection == "final" else index
        capacity_manifest = getattr(capacity, "manifest", None)
        naked_manifest = getattr(naked, "manifest", None)
        if not isinstance(capacity_manifest, Mapping) or not isinstance(naked_manifest, Mapping):
            raise TypeError("Control artifacts require strict manifests")
        common = {
            "selection": selection,
            "oof_fold": fold,
            "matched_k31_checkpoint_sha256": candidate.checkpoint_sha256,
            "matched_k31_manifest_sha256": candidate.manifest.get(
                "legacy_recovery_manifest_sha256"
            ),
            "training_public_roster_sha256": candidate.fit_patient_roster_sha256,
            "i_gate_patient_roster_sha256": k31_projection.manifest[
                "gate_patient_roster_sha256"
            ],
            "i_gate_outcomes_opened": False,
            "i_gate_target_values_materialized": False,
            "native_evaluation_performed": False,
        }
        if any(capacity_manifest.get(key) != value for key, value in common.items()) or any(
            naked_manifest.get(key) != value for key, value in common.items()
        ):
            raise ValueError(f"Control artifacts do not match {selection} lineage")
        if (
            capacity_manifest.get("schema_version")
            != "soz_labram_ictal_capacity_matched_channel_control_v13_1"
            or capacity_manifest.get("capacity_matched_to_k31") is not True
            or naked_manifest.get("schema_version")
            != "soz_labram_ictal_independent_control_v13_1"
            or naked_manifest.get("capacity_matched_to_k31") is not False
        ):
            raise ValueError("Control artifact kind changed")
        capacity_raw, capacity_sha = _stable_checkpoint_bytes(
            capacity, field=f"{selection} capacity"
        )
        naked_raw, naked_sha = _stable_checkpoint_bytes(
            naked, field=f"{selection} naked"
        )
        capacity_name = f"{selection}.capacity.safetensors"
        naked_name = f"{selection}.naked.safetensors"
        files[capacity_name] = capacity_raw
        files[naked_name] = naked_raw
        row = {
            "selection": selection,
            "oof_fold": fold,
            "k31_projection_record_sha256": candidate.record_sha256,
            "fit_patient_roster_sha256": candidate.fit_patient_roster_sha256,
            "fit_patient_count": len(candidate.fit_patient_ids),
            "target_semantics": head_schema.EXPECTED_TARGET_SEMANTICS,
            "capacity_source_manifest_sha256": require_sha256(
                getattr(capacity, "manifest_sha256", None),
                field="capacity.manifest_sha256",
            ),
            "capacity_checkpoint_filename": capacity_name,
            "capacity_checkpoint_sha256": capacity_sha,
            "capacity_head_state_sha256": require_sha256(
                capacity_manifest.get("head_state_sha256"), field="capacity.head_state_sha256"
            ),
            "capacity_head_config": {
                "token_dim": 200,
                "hidden_dim": 128,
                "type": "capacity_matched_channel_residual",
            },
            "naked_source_manifest_sha256": require_sha256(
                getattr(naked, "manifest_sha256", None),
                field="naked.manifest_sha256",
            ),
            "naked_checkpoint_filename": naked_name,
            "naked_checkpoint_sha256": naked_sha,
            "naked_head_state_sha256": require_sha256(
                naked_manifest.get("head_state_sha256"), field="naked.head_state_sha256"
            ),
            "naked_head_config": {
                "token_dim": 200,
                "hidden_dim": 128,
                "type": "naked_independent_second",
            },
        }
        row["projection_record_sha256"] = canonical_sha256(row)
        rows.append(row)
    manifest = {
        "schema_version": head_schema.SCHEMA,
        "purpose": head_schema.PURPOSE,
        "serialization": "canonical_json_plus_twelve_safetensors_no_pickle",
        "producer_order": list(EXPECTED_PRODUCER_ORDER),
        "producer_count": 6,
        "v5_split_sha256": k31_projection.v5_split_sha256,
        "gate_patient_roster_sha256": k31_projection.manifest[
            "gate_patient_roster_sha256"
        ],
        "producers": rows,
        "broker_receipt": {
            "schema_version": "soz_labram_ictal_control_head_projection_broker_v13_1",
            "control_training_manifests_loaded": True,
            "control_checkpoint_weights_loaded": True,
            "fit_target_artifacts_loaded": False,
            "fit_target_values_loaded": False,
            "fit_target_masks_loaded": False,
            "gate_signal_or_tokens_loaded": False,
            "gate_target_values_loaded": False,
            "gate_target_masks_loaded": False,
            "forward_performed": False,
            "evaluation_performed": False,
            "projection_contains_training_metrics": False,
            "projection_contains_target_values": False,
            "projection_contains_target_masks": False,
            "projection_contains_source_paths": False,
        },
        "formal_promotion": False,
        "authorized_for_formal_evidence_or_reasoner": False,
        "v13_execution_hold": True,
    }
    published = _publish_directory(
        _new_output(output_directory, field="control-head projection"),
        manifest=manifest,
        files=files,
    )
    head_schema.load_control_head_projection(
        published[0], expected_manifest_sha256=published[1]
    )
    return published


@dataclass(frozen=True)
class GateTokenBrokerEventV13_1:
    patient_id: str
    event_id: str
    event_record_sha256: str
    preprocess_receipt_sha256: str
    source_bundle_manifest_sha256: str
    source_tensor_sha256: str
    tokens: torch.Tensor


@dataclass(frozen=True)
class GateTokenBrokerSourceV13_1:
    source_master_index_sha256: str
    source_master_manifest_sha256: str
    preprocessing_selection_artifact_sha256: str
    preprocessing_protocol_receipt_sha256: str
    foundation_feature_receipt_sha256: str
    foundation_checkpoint_sha256: str
    foundation_modeling_sha256: str
    events: tuple[GateTokenBrokerEventV13_1, ...]
    non_gate_token_values_loaded: bool = False


@dataclass(frozen=True)
class SignalMaskBrokerSourceV13_1:
    event_rows: tuple[tuple[str, str], ...]
    deployment_mask: torch.Tensor
    phase_mask: torch.Tensor
    signal_preflight_artifact_sha256: str
    signal_preflight_receipt_sha256: str
    timeline_context_receipt_sha256: str
    mask_derivation_receipt_sha256: str
    token_lineage_receipt_sha256: str
    signal_mask_artifact_manifest_sha256: str
    deployment_mask_derived_from_signal_samples: bool = True
    phase_mask_derived_from_signal_timeline: bool = True
    per_edge_annotation_loaded: bool = False
    annotation_target_mask_loaded: bool = False
    all_true_fallback_used: bool = False


def _publish_gate_token_view(
    output_directory: str | Path,
    *,
    k31_projection: NeutralK31Projection,
    token_source: GateTokenBrokerSourceV13_1,
    signal_masks: SignalMaskBrokerSourceV13_1,
    expected_event_count: int,
) -> tuple[Path, str]:
    if not isinstance(k31_projection, NeutralK31Projection):
        raise TypeError("k31_projection must come from the neutral strict loader")
    if not isinstance(token_source, GateTokenBrokerSourceV13_1) or not isinstance(
        signal_masks, SignalMaskBrokerSourceV13_1
    ):
        raise TypeError("Gate view requires closed broker source records")
    events = tuple(token_source.events)
    if len(events) != expected_event_count:
        raise ValueError("Gate token event count changed")
    rows = tuple((event.event_id, event.patient_id) for event in events)
    if rows != signal_masks.event_rows:
        raise ValueError("Signal masks and token events do not align")
    patients = tuple(sorted({event.patient_id for event in events}))
    if patients != k31_projection.gate_patient_ids:
        raise ValueError("Gate token patient roster differs from k31 projection")
    if token_source.non_gate_token_values_loaded is not False:
        raise ValueError("Gate-view broker opened non-gate token values")
    if (
        signal_masks.deployment_mask_derived_from_signal_samples is not True
        or signal_masks.phase_mask_derived_from_signal_timeline is not True
        or signal_masks.per_edge_annotation_loaded is not False
        or signal_masks.annotation_target_mask_loaded is not False
        or signal_masks.all_true_fallback_used is not False
    ):
        raise ValueError("Signal-mask source violates the annotation firewall")
    for field in (
        "signal_preflight_artifact_sha256", "signal_preflight_receipt_sha256",
        "timeline_context_receipt_sha256", "mask_derivation_receipt_sha256",
        "token_lineage_receipt_sha256", "signal_mask_artifact_manifest_sha256",
    ):
        require_sha256(getattr(signal_masks, field), field=field)
    token_tensors = []
    event_payloads = []
    for event in events:
        if not event.event_id.startswith(f"{event.patient_id}_"):
            raise ValueError("Gate broker event identity changed")
        for field in (
            "event_record_sha256", "preprocess_receipt_sha256",
            "source_bundle_manifest_sha256", "source_tensor_sha256",
        ):
            require_sha256(getattr(event, field), field=field)
        value = event.tokens.detach().cpu().to(torch.float32).contiguous()
        if tuple(value.shape) != TOKEN_EVENT_SHAPE or not torch.isfinite(value).all():
            raise ValueError("Gate broker token shape/value changed")
        token_tensors.append(value)
        event_payloads.append(
            {
                "patient_id": event.patient_id,
                "event_id": event.event_id,
                "event_record_sha256": event.event_record_sha256,
                "preprocess_receipt_sha256": event.preprocess_receipt_sha256,
                "source_bundle_manifest_sha256": event.source_bundle_manifest_sha256,
                "source_tensor_sha256": event.source_tensor_sha256,
            }
        )
    if tuple((row["patient_id"], row["event_id"]) for row in event_payloads) != tuple(
        sorted((row["patient_id"], row["event_id"]) for row in event_payloads)
    ):
        raise ValueError("Gate broker event order changed")
    tokens = torch.stack(token_tensors, dim=0)
    deployment = signal_masks.deployment_mask.detach().cpu().to(torch.bool).contiguous()
    phase = signal_masks.phase_mask.detach().cpu().to(torch.bool).contiguous()
    if tuple(deployment.shape) != (expected_event_count, 20, 60) or tuple(phase.shape) != (expected_event_count, 15):
        raise ValueError("Gate signal masks have the wrong shape")
    target = _new_output(output_directory, field="physical gate token view")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tensor-", dir=target.parent))
    try:
        tensor_path = staging / token_schema.TENSOR_FILENAME
        save_safetensors_file(
            {
                token_schema.TOKEN_NAME: tokens,
                token_schema.DEPLOYMENT_MASK_NAME: deployment,
                token_schema.PHASE_MASK_NAME: phase,
            },
            str(tensor_path),
        )
        _fsync_file(tensor_path)
        tensor_raw = tensor_path.read_bytes()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    manifest = {
        "schema_version": token_schema.SCHEMA,
        "purpose": token_schema.PURPOSE,
        "serialization": "canonical_json_plus_single_safetensors_no_pickle",
        "v5_split_sha256": k31_projection.v5_split_sha256,
        "source_master_index_sha256": require_sha256(
            token_source.source_master_index_sha256, field="source_master_index_sha256"
        ),
        "source_master_manifest_sha256": require_sha256(
            token_source.source_master_manifest_sha256, field="source_master_manifest_sha256"
        ),
        "preprocessing_selection_artifact_sha256": require_sha256(
            token_source.preprocessing_selection_artifact_sha256,
            field="preprocessing_selection_artifact_sha256",
        ),
        "preprocessing_protocol_receipt_sha256": require_sha256(
            token_source.preprocessing_protocol_receipt_sha256,
            field="preprocessing_protocol_receipt_sha256",
        ),
        "preprocessing_selected_arm_id": "C-CAR19",
        "foundation_feature_receipt_sha256": require_sha256(
            token_source.foundation_feature_receipt_sha256,
            field="foundation_feature_receipt_sha256",
        ),
        "foundation_checkpoint_sha256": token_source.foundation_checkpoint_sha256,
        "foundation_modeling_sha256": token_source.foundation_modeling_sha256,
        "patient_ids": list(patients),
        "patient_roster_sha256": patient_roster_sha256(patients),
        "patient_count": len(patients),
        "event_count": len(events),
        "event_roster_sha256": canonical_sha256(event_payloads),
        "events": event_payloads,
        "tensor": {
            "filename": token_schema.TENSOR_FILENAME,
            "file_sha256": hashlib.sha256(tensor_raw).hexdigest(),
            "file_size_bytes": len(tensor_raw),
            "tokens_shape": list(tokens.shape),
            "tokens_dtype": "float32",
            "tokens_value_sha256": token_schema.tensor_value_sha256(
                token_schema.TOKEN_NAME, tokens
            ),
            "deployment_mask_shape": list(deployment.shape),
            "deployment_mask_dtype": "bool",
            "deployment_mask_value_sha256": token_schema.tensor_value_sha256(
                token_schema.DEPLOYMENT_MASK_NAME, deployment
            ),
            "phase_mask_shape": list(phase.shape),
            "phase_mask_dtype": "bool",
            "phase_mask_value_sha256": token_schema.tensor_value_sha256(
                token_schema.PHASE_MASK_NAME, phase
            ),
        },
        "signal_mask_lineage": {
            "schema_version": "soz_labram_k31_signal_mask_lineage_v13_1",
            "signal_preflight_artifact_sha256": signal_masks.signal_preflight_artifact_sha256,
            "signal_preflight_receipt_sha256": signal_masks.signal_preflight_receipt_sha256,
            "timeline_context_receipt_sha256": signal_masks.timeline_context_receipt_sha256,
            "mask_derivation_receipt_sha256": signal_masks.mask_derivation_receipt_sha256,
            "token_lineage_receipt_sha256": signal_masks.token_lineage_receipt_sha256,
            "signal_mask_artifact_manifest_sha256": (
                signal_masks.signal_mask_artifact_manifest_sha256
            ),
            "deployment_mask_derived_from_signal_samples": True,
            "phase_mask_derived_from_signal_timeline": True,
            "per_edge_annotation_loaded": False,
            "annotation_target_mask_loaded": False,
            "all_true_fallback_used": False,
        },
        "access_receipt": {
            "schema_version": "soz_labram_k31_physical_gate_token_view_access_v13_1",
            "broker_master_signal_index_loaded": True,
            "broker_gate_token_values_loaded": True,
            "broker_non_gate_token_values_loaded": False,
            "broker_signal_availability_loaded": True,
            "broker_target_values_loaded": False,
            "broker_target_masks_loaded": False,
            "broker_clinical_identity_loaded": False,
            "broker_per_edge_annotation_loaded": False,
            "physical_view_contains_only_gate_events": True,
            "physical_view_contains_source_paths": False,
            "physical_view_contains_labels": False,
            "physical_view_contains_target_masks": False,
            "training_performed": False,
            "evaluation_performed": False,
        },
        "v13_execution_hold": True,
    }
    published = _publish_directory(
        target,
        manifest=manifest,
        files={token_schema.TENSOR_FILENAME: tensor_raw},
    )
    token_schema.load_gate_token_view(
        published[0], expected_manifest_sha256=published[1]
    )
    return published


def publish_gate_token_view_v13_1(
    output_directory: str | Path,
    *,
    k31_projection: NeutralK31Projection,
    token_source: GateTokenBrokerSourceV13_1,
    signal_masks: VerifiedSignalMaskArtifactV13_1,
) -> tuple[Path, str]:
    """Publish the physical gate view from a strict-reloaded mask artifact.

    The public production path deliberately does not accept a forgeable pair
    of arbitrary boolean tensors.  Synthetic tests retain a private seam
    below, while production must consume the closed issuer capability and
    cross-check its token-lineage receipt event by event.
    """

    if not isinstance(signal_masks, VerifiedSignalMaskArtifactV13_1):
        raise TypeError("production gate view requires the strict signal-mask artifact")
    if signal_masks.manifest.get("source_master_index_sha256") != token_source.source_master_index_sha256:
        raise ValueError("signal-mask and gate-token master indices disagree")
    mask_events = tuple(signal_masks.events)
    if len(mask_events) != len(token_source.events):
        raise ValueError("signal-mask and gate-token event counts disagree")
    for mask_event, token_event in zip(mask_events, token_source.events, strict=True):
        if (
            mask_event.patient_id != token_event.patient_id
            or mask_event.event_id != token_event.event_id
            or mask_event.event_record_sha256 != token_event.event_record_sha256
            or mask_event.preprocess_receipt_sha256 != token_event.preprocess_receipt_sha256
            or mask_event.source_bundle_manifest_sha256
            != token_event.source_bundle_manifest_sha256
            or mask_event.source_tensor_sha256 != token_event.source_tensor_sha256
        ):
            raise ValueError("signal-mask and gate-token per-event lineage disagree")
    receipts = signal_masks.receipts
    broker_masks = SignalMaskBrokerSourceV13_1(
        event_rows=signal_masks.event_rows,
        deployment_mask=signal_masks.deployment_mask,
        phase_mask=signal_masks.phase_mask,
        signal_preflight_artifact_sha256=str(
            signal_masks.manifest["signal_preflight_artifact_sha256"]
        ),
        signal_preflight_receipt_sha256=str(
            receipts["signal_preflight_receipt_sha256"]
        ),
        timeline_context_receipt_sha256=str(
            receipts["timeline_context_receipt_sha256"]
        ),
        mask_derivation_receipt_sha256=str(
            receipts["mask_derivation_receipt_sha256"]
        ),
        token_lineage_receipt_sha256=str(
            receipts["token_lineage_receipt_sha256"]
        ),
        signal_mask_artifact_manifest_sha256=signal_masks.manifest_sha256,
    )
    return _publish_gate_token_view(
        output_directory,
        k31_projection=k31_projection,
        token_source=token_source,
        signal_masks=broker_masks,
        expected_event_count=token_schema.EXPECTED_EVENT_COUNT,
    )


def _publish_gate_token_view_for_synthetic_test(
    output_directory: str | Path,
    *,
    k31_projection: NeutralK31Projection,
    token_source: GateTokenBrokerSourceV13_1,
    signal_masks: SignalMaskBrokerSourceV13_1,
) -> tuple[Path, str]:
    return _publish_gate_token_view(
        output_directory,
        k31_projection=k31_projection,
        token_source=token_source,
        signal_masks=signal_masks,
        expected_event_count=len(token_source.events),
    )


__all__ = (
    "GateTokenBrokerEventV13_1",
    "GateTokenBrokerSourceV13_1",
    "SignalMaskBrokerSourceV13_1",
    "publish_control_head_projection_v13_1",
    "publish_control_parameter_projection_v13_1",
    "publish_gate_token_view_v13_1",
)
