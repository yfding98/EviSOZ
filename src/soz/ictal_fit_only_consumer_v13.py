"""Consumer-only strict loader for physical v13 fit-target artifacts.

Unlike :mod:`ictal_fit_only_targets_v13`, this module contains no source
snapshot index, full-manifest broker, row extraction, or materialization
capability.  Formal control trainers import only this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .ictal_fit_primitives_v13 import (
    IctalTrainingConfig,
    LABRAM_K31_TARGET_SEMANTICS,
    VerifiedFitOnlyIctalTargetSnapshotV13,
    canonical_json_bytes,
    patient_roster,
    patient_roster_sha256,
    require_sha256,
    selection,
)


FIT_ONLY_TARGET_SCHEMA_V13 = "soz_ictal_fit_only_target_artifact_v13_1"
FIT_ONLY_TARGET_RECEIPT_SCHEMA_V13 = "soz_ictal_fit_only_target_receipt_v13_1"
FIT_ONLY_TARGET_MANIFEST = "manifest.json"
FIT_ONLY_TARGET_RECEIPT = "receipt.json"
FIT_ONLY_TARGETS = "fit_targets.npy"
FIT_ONLY_TARGET_MASK = "fit_target_mask.npy"
FIT_ONLY_SHORTCUT_SCHEMA_V13 = "soz_ictal_fit_only_shortcut_parameters_v13_1"
TIME_ONLY_ALGORITHM_V13 = (
    "laplace_smoothed_training_prevalence_by_relative_second_v1"
)
MASK_ONLY_ALGORITHM_V13 = (
    "laplace_smoothed_training_prevalence_by_event_mask_density_quartile_v1"
)
PREVALENCE_ONLY_ALGORITHM_V13 = (
    "laplace_smoothed_training_global_observed_cell_prevalence_v1"
)
UNKNOWN_TARGET_CELL_POLICY_V13 = "masked_unknown_excluded_never_imputed_negative"
MASK_DENSITY_BOUNDARIES_V13 = (0.25, 0.50, 0.75)
FIT_ONLY_TARGET_FILES = frozenset(
    {
        FIT_ONLY_TARGET_MANIFEST,
        FIT_ONLY_TARGET_RECEIPT,
        FIT_ONLY_TARGETS,
        FIT_ONLY_TARGET_MASK,
    }
)
_MAX_JSON_BYTES = 16 * 1024 * 1024
_EVENT_SHAPE = (20, 60)
_FIT_ONLY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "selection",
        "oof_fold",
        "target_semantics",
        "source_access_policy",
        "matched_k31_manifest_sha256",
        "matched_k31_checkpoint_sha256",
        "matched_training_config",
        "training_manifest_bundle_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "source_target_snapshot_manifest_sha256",
        "source_target_snapshot_receipt_sha256",
        "source_native_evaluation_manifest_sha256",
        "source_native_evaluation_corpus_index_sha256",
        "source_training_targets_declared_file_sha256",
        "source_training_target_mask_declared_file_sha256",
        "source_full_tensor_hashes_computed",
        "source_full_arrays_loaded",
        "source_full_arrays_mapped",
        "np_load_invoked_on_source",
        "fit_rows_only_pread",
        "i_gate_target_row_byte_ranges_read",
        "i_gate_target_values_materialized",
        "i_gate_outcomes_opened",
        "deepsoz_target_source_loaded",
        "deepsoz_soz_labels_used",
        "private_labels_used",
        "missing_tusz_cells_imputed_as_negative",
        "fit_patient_ids",
        "fit_patient_roster_sha256",
        "i_gate_patient_ids_excluded_unopened",
        "i_gate_patient_roster_sha256",
        "fit_event_count",
        "fit_event_rows",
        "selected_source_row_indices_sha256",
        "selected_target_byte_ranges_sha256",
        "selected_mask_byte_ranges_sha256",
        "fit_gate_source_row_intersection_count",
        "broker_full_training_manifest_metadata_loaded",
        "broker_gate_target_derived_hashes_counts_loaded",
        "broker_legacy_k31_full_manifest_loaded",
        "broker_legacy_k31_native_roster_metrics_metadata_loaded",
        "broker_legacy_k31_checkpoint_weights_loaded",
        "broker_gate_target_values_read",
        "broker_gate_target_masks_read",
        "trainer_requires_full_training_manifest",
        "trainer_requires_full_k31_manifest",
        "fit_only_shortcut_parameters",
        "tensor_files",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, object], str]:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError(f"Strict JSON must be a regular absolute file: {source.name}")
    before = source.stat()
    if before.st_size < 1 or before.st_size > _MAX_JSON_BYTES:
        raise ValueError(f"Strict JSON has an invalid size: {source.name}")
    raw = source.read_bytes()
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Strict JSON changed while read: {source.name}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != require_sha256(
        expected_sha256, field=f"expected_{source.name}_sha256"
    ):
        raise ValueError(f"Strict JSON SHA mismatch: {source.name}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Strict JSON is invalid: {source.name}") from exc
    canonical = canonical_json_bytes(payload)
    if not isinstance(payload, dict) or raw not in {canonical, canonical + b"\n"}:
        raise ValueError(f"Strict JSON is not canonical: {source.name}")
    return payload, digest


def _event_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _artifact_tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(value.shape)}|{value.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = value.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _laplace_logit(*, positive_count: int, observed_count: int) -> float:
    if (
        isinstance(positive_count, bool)
        or not isinstance(positive_count, int)
        or isinstance(observed_count, bool)
        or not isinstance(observed_count, int)
        or positive_count < 0
        or observed_count < 0
        or positive_count > observed_count
    ):
        raise ValueError("Laplace counts must satisfy 0 <= positive <= observed")
    probability = (positive_count + 1.0) / (observed_count + 2.0)
    value = math.log(probability / (1.0 - probability))
    if not math.isfinite(value):
        raise ValueError("Laplace-smoothed logit must be finite")
    return value


def _shortcut_tensor_inputs(
    targets: torch.Tensor, target_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(targets, torch.Tensor) or not isinstance(
        target_mask, torch.Tensor
    ):
        raise TypeError("shortcut targets and mask must be tensors")
    values = targets.detach().cpu().contiguous()
    mask = target_mask.detach().cpu().contiguous()
    if (
        values.ndim != 3
        or tuple(values.shape[1:]) != _EVENT_SHAPE
        or values.shape[0] < 1
        or mask.shape != values.shape
        or not values.is_floating_point()
        or mask.dtype != torch.bool
    ):
        raise ValueError("shortcut tensors must have shape [E,20,60] and fixed dtypes")
    if not torch.isfinite(values).all():
        raise ValueError("shortcut targets must be finite")
    observed = values[mask]
    if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("shortcut observed targets must be binary")
    return values, mask


def _hashed_parameter_section(core: Mapping[str, object]) -> dict[str, object]:
    payload = dict(core)
    payload["parameters_sha256"] = _canonical_sha256(payload)
    return payload


def compute_fit_only_shortcut_parameters_v13(
    targets: torch.Tensor, target_mask: torch.Tensor
) -> dict[str, object]:
    values, mask = _shortcut_tensor_inputs(targets, target_mask)
    positives = mask & (values == 1)
    event_count, edge_count, second_count = (int(item) for item in values.shape)
    time_rows = []
    for relative_second in range(second_count):
        observed_count = int(mask[:, :, relative_second].sum().item())
        positive_count = int(positives[:, :, relative_second].sum().item())
        time_rows.append(
            {
                "relative_second": relative_second,
                "observed_cell_count": observed_count,
                "positive_cell_count": positive_count,
                "logit": _laplace_logit(
                    positive_count=positive_count,
                    observed_count=observed_count,
                ),
            }
        )
    time_only = _hashed_parameter_section(
        {
            "algorithm_id": TIME_ONLY_ALGORITHM_V13,
            "relative_second_count": second_count,
            "parameter_rows": time_rows,
        }
    )
    global_observed = int(mask.sum().item())
    global_positive = int(positives.sum().item())
    global_logit = _laplace_logit(
        positive_count=global_positive, observed_count=global_observed
    )
    prevalence_only = _hashed_parameter_section(
        {
            "algorithm_id": PREVALENCE_ONLY_ALGORITHM_V13,
            "observed_cell_count": global_observed,
            "positive_cell_count": global_positive,
            "logit": global_logit,
        }
    )
    densities = mask.reshape(event_count, edge_count * second_count).to(
        torch.float64
    ).mean(dim=1)
    assignments = torch.bucketize(
        densities,
        torch.tensor(MASK_DENSITY_BOUNDARIES_V13, dtype=torch.float64),
        right=False,
    )
    bin_rows = []
    for bin_index in range(len(MASK_DENSITY_BOUNDARIES_V13) + 1):
        selected = assignments == bin_index
        selected_event_count = int(selected.sum().item())
        if selected_event_count:
            observed_count = int(mask[selected].sum().item())
            positive_count = int(positives[selected].sum().item())
            fallback = False
            logit = _laplace_logit(
                positive_count=positive_count, observed_count=observed_count
            )
        else:
            observed_count = 0
            positive_count = 0
            fallback = True
            logit = global_logit
        bin_rows.append(
            {
                "bin_index": bin_index,
                "event_count": selected_event_count,
                "observed_cell_count": observed_count,
                "positive_cell_count": positive_count,
                "fallback_to_global": fallback,
                "logit": logit,
            }
        )
    mask_only = _hashed_parameter_section(
        {
            "algorithm_id": MASK_ONLY_ALGORITHM_V13,
            "event_mask_density_denominator": edge_count * second_count,
            "bucket_boundaries": list(MASK_DENSITY_BOUNDARIES_V13),
            "torch_bucketize_right": False,
            "global_observed_cell_count": global_observed,
            "global_positive_cell_count": global_positive,
            "global_logit": global_logit,
            "bin_rows": bin_rows,
        }
    )
    core = {
        "schema_version": FIT_ONLY_SHORTCUT_SCHEMA_V13,
        "unknown_target_cell_policy": UNKNOWN_TARGET_CELL_POLICY_V13,
        "fit_event_count": event_count,
        "time_only": time_only,
        "prevalence_only": prevalence_only,
        "mask_only": mask_only,
    }
    return {**core, "bundle_sha256": _canonical_sha256(core)}


def _require_finite_json_numbers(value: object, *, field: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"fit-only shortcut parameter is non-finite: {field}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_json_numbers(item, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json_numbers(item, field=f"{field}[{index}]")


def validate_fit_only_shortcut_parameters_v13(
    value: object, *, targets: torch.Tensor, target_mask: torch.Tensor
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("fit_only_shortcut_parameters must be a mapping")
    payload = dict(value)
    _require_finite_json_numbers(payload, field="fit_only_shortcut_parameters")
    expected = compute_fit_only_shortcut_parameters_v13(targets, target_mask)
    if set(payload) != set(expected):
        raise ValueError("fit-only shortcut bundle violates its closed schema")
    for name in ("time_only", "prevalence_only", "mask_only"):
        section = payload.get(name)
        expected_section = expected[name]
        if not isinstance(section, Mapping) or not isinstance(expected_section, Mapping):
            raise TypeError(f"fit-only shortcut section must be a mapping: {name}")
        record = dict(section)
        if set(record) != set(expected_section):
            raise ValueError(f"fit-only shortcut section violates its closed schema: {name}")
        declared = require_sha256(
            record.pop("parameters_sha256"),
            field=f"fit_only_shortcut_parameters.{name}.parameters_sha256",
        )
        if _canonical_sha256(record) != declared:
            raise ValueError(f"fit-only shortcut section SHA mismatch: {name}")
    bundle = dict(payload)
    declared_bundle = require_sha256(
        bundle.pop("bundle_sha256"),
        field="fit_only_shortcut_parameters.bundle_sha256",
    )
    if _canonical_sha256(bundle) != declared_bundle:
        raise ValueError("fit-only shortcut bundle SHA mismatch")
    if canonical_json_bytes(payload) != canonical_json_bytes(expected):
        raise ValueError("fit-only shortcut parameters failed exact fit-row recomputation")
    return expected


@dataclass(frozen=True)
class FitOnlyTargetEventV13:
    event_id: str
    patient_id: str
    source_row_index: int
    event_record_sha256: str
    preprocess_receipt_sha256: str
    target_sha256: str
    target_mask_sha256: str


@dataclass(frozen=True)
class LoadedFitOnlyTargetArtifactV13:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    receipt_sha256: str
    events: tuple[FitOnlyTargetEventV13, ...]
    snapshot: VerifiedFitOnlyIctalTargetSnapshotV13
    shortcut_parameters: Mapping[str, object]


def _load_artifact_array(
    path: Path, *, record: object, name: str, filename: str
) -> torch.Tensor:
    if not isinstance(record, Mapping):
        raise TypeError(f"fit-only tensor record must be a mapping: {name}")
    payload = dict(record)
    if set(payload) != {
        "filename",
        "shape",
        "dtype",
        "file_size_bytes",
        "file_sha256",
        "tensor_sha256",
    } or payload["filename"] != filename:
        raise ValueError(f"fit-only tensor record changed: {name}")
    source = path / filename
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError(f"fit-only tensor must be regular: {filename}")
    raw = source.read_bytes()
    if (
        len(raw) != payload["file_size_bytes"]
        or hashlib.sha256(raw).hexdigest()
        != require_sha256(payload["file_sha256"], field=f"{name}.file_sha256")
    ):
        raise ValueError(f"fit-only tensor file receipt mismatch: {name}")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"fit-only tensor is not safe NumPy: {name}") from exc
    tensor = torch.from_numpy(np.ascontiguousarray(array)).contiguous()
    if list(tensor.shape) != payload["shape"] or str(tensor.dtype) != payload["dtype"]:
        raise ValueError(f"fit-only tensor shape/dtype changed: {name}")
    if _artifact_tensor_sha256(name, tensor) != require_sha256(
        payload["tensor_sha256"], field=f"{name}.tensor_sha256"
    ):
        raise ValueError(f"fit-only tensor value receipt mismatch: {name}")
    return tensor


def load_fit_only_target_artifact_v13(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> LoadedFitOnlyTargetArtifactV13:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("fit-only target artifact must be a regular absolute directory")
    if {entry.name for entry in source.iterdir()} != FIT_ONLY_TARGET_FILES:
        raise ValueError("fit-only target artifact has missing or unknown files")
    manifest, manifest_sha = _strict_json(
        source / FIT_ONLY_TARGET_MANIFEST,
        expected_sha256=expected_manifest_sha256,
    )
    receipt, receipt_sha = _strict_json(
        source / FIT_ONLY_TARGET_RECEIPT,
        expected_sha256=expected_receipt_sha256,
    )
    if set(manifest) != _FIT_ONLY_MANIFEST_FIELDS:
        raise ValueError("fit-only target manifest violates its closed schema")
    if receipt != {
        "schema_version": FIT_ONLY_TARGET_RECEIPT_SCHEMA_V13,
        "artifact_sha256": manifest_sha,
        "artifact_size_bytes": (source / FIT_ONLY_TARGET_MANIFEST).stat().st_size,
    }:
        raise ValueError("fit-only target receipt changed")
    fixed = {
        "schema_version": FIT_ONLY_TARGET_SCHEMA_V13,
        "target_semantics": LABRAM_K31_TARGET_SEMANTICS,
        "source_access_policy": "npy_header_then_explicit_fit_c_order_pread_rows_v1",
        "source_full_tensor_hashes_computed": False,
        "source_full_arrays_loaded": False,
        "source_full_arrays_mapped": False,
        "np_load_invoked_on_source": False,
        "fit_rows_only_pread": True,
        "i_gate_target_row_byte_ranges_read": False,
        "i_gate_target_values_materialized": False,
        "i_gate_outcomes_opened": False,
        "deepsoz_target_source_loaded": False,
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "missing_tusz_cells_imputed_as_negative": False,
        "fit_gate_source_row_intersection_count": 0,
        "broker_full_training_manifest_metadata_loaded": True,
        "broker_gate_target_derived_hashes_counts_loaded": True,
        "broker_legacy_k31_full_manifest_loaded": True,
        "broker_legacy_k31_native_roster_metrics_metadata_loaded": True,
        "broker_legacy_k31_checkpoint_weights_loaded": True,
        "broker_gate_target_values_read": False,
        "broker_gate_target_masks_read": False,
        "trainer_requires_full_training_manifest": False,
        "trainer_requires_full_k31_manifest": False,
    }
    if any(manifest.get(field) != value for field, value in fixed.items()):
        raise ValueError("fit-only target artifact changed an access boundary")
    normalized_selection, fold = selection(manifest.get("selection"))
    if manifest.get("oof_fold") != fold:
        raise ValueError("fit-only selection/fold mismatch")
    fit = patient_roster(
        manifest.get("fit_patient_ids"), field="fit_patient_ids", allow_empty=False
    )
    gate = patient_roster(
        manifest.get("i_gate_patient_ids_excluded_unopened"),
        field="i_gate_patient_ids_excluded_unopened",
        allow_empty=False,
    )
    if (
        len(gate) != 12
        or set(fit) & set(gate)
        or patient_roster_sha256(fit) != manifest.get("fit_patient_roster_sha256")
        or patient_roster_sha256(gate)
        != manifest.get("i_gate_patient_roster_sha256")
    ):
        raise ValueError("fit-only target patient firewall failed")
    hash_fields = (
        "matched_k31_manifest_sha256",
        "matched_k31_checkpoint_sha256",
        "training_manifest_bundle_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "source_target_snapshot_manifest_sha256",
        "source_target_snapshot_receipt_sha256",
        "source_native_evaluation_manifest_sha256",
        "source_native_evaluation_corpus_index_sha256",
        "source_training_targets_declared_file_sha256",
        "source_training_target_mask_declared_file_sha256",
        "selected_source_row_indices_sha256",
        "selected_target_byte_ranges_sha256",
        "selected_mask_byte_ranges_sha256",
    )
    for field in hash_fields:
        require_sha256(manifest.get(field), field=field)
    if manifest.get("matched_training_config") != asdict(IctalTrainingConfig()):
        raise ValueError("fit-only target matched training config changed")
    event_rows = manifest.get("fit_event_rows")
    if not isinstance(event_rows, list) or len(event_rows) != manifest.get(
        "fit_event_count"
    ):
        raise ValueError("fit-only event roster changed")
    normalized = []
    for row in event_rows:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("fit-only event row schema changed")
        event_id, patient_id, source_index, record_sha, pre_sha, target_sha, mask_sha = row
        if (
            not isinstance(event_id, str)
            or not isinstance(patient_id, str)
            or patient_id not in set(fit)
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise ValueError("fit-only event row identity is invalid")
        normalized.append(
            FitOnlyTargetEventV13(
                event_id=event_id,
                patient_id=patient_id,
                source_row_index=source_index,
                event_record_sha256=require_sha256(
                    record_sha, field="event.event_record_sha256"
                ),
                preprocess_receipt_sha256=require_sha256(
                    pre_sha, field="event.preprocess_receipt_sha256"
                ),
                target_sha256=require_sha256(target_sha, field="event.target_sha256"),
                target_mask_sha256=require_sha256(mask_sha, field="event.mask_sha256"),
            )
        )
    if (
        len({row.event_id for row in normalized}) != len(normalized)
        or tuple(row.source_row_index for row in normalized)
        != tuple(sorted(row.source_row_index for row in normalized))
        or {row.patient_id for row in normalized} != set(fit)
        or _canonical_sha256(tuple(row.source_row_index for row in normalized))
        != manifest["selected_source_row_indices_sha256"]
    ):
        raise ValueError("fit-only event/source-row roster failed")
    tensor_files = manifest.get("tensor_files")
    if not isinstance(tensor_files, Mapping) or set(tensor_files) != {
        "fit_targets",
        "fit_target_mask",
    }:
        raise ValueError("fit-only tensor roster changed")
    targets = _load_artifact_array(
        source,
        record=tensor_files["fit_targets"],
        name="fit_targets",
        filename=FIT_ONLY_TARGETS,
    ).to(torch.float32)
    mask = _load_artifact_array(
        source,
        record=tensor_files["fit_target_mask"],
        name="fit_target_mask",
        filename=FIT_ONLY_TARGET_MASK,
    ).to(torch.bool)
    if tuple(targets.shape) != (len(normalized), 20, 60) or mask.shape != targets.shape:
        raise ValueError("fit-only target tensors have the wrong shape")
    if not torch.isfinite(targets).all():
        raise ValueError("fit-only targets are non-finite")
    observed = targets[mask]
    if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("fit-only observed targets are not binary")
    for row, target, row_mask in zip(normalized, targets, mask, strict=True):
        if (
            _event_tensor_sha256(target) != row.target_sha256
            or _event_tensor_sha256(row_mask) != row.target_mask_sha256
        ):
            raise ValueError(f"fit-only event value receipt mismatch: {row.event_id}")
    shortcuts = validate_fit_only_shortcut_parameters_v13(
        manifest.get("fit_only_shortcut_parameters"),
        targets=targets,
        target_mask=mask,
    )
    empty_targets = torch.empty((0, 20, 60), dtype=torch.float32)
    empty_mask = torch.empty((0, 20, 60), dtype=torch.bool)
    snapshot = VerifiedFitOnlyIctalTargetSnapshotV13(
        path=source,
        manifest_sha256=manifest_sha,
        receipt_sha256=receipt_sha,
        training_manifest_sha256=str(manifest["training_manifest_sha256"]),
        training_corpus_index_sha256=str(manifest["training_corpus_index_sha256"]),
        native_manifest_sha256=str(manifest["source_native_evaluation_manifest_sha256"]),
        native_corpus_index_sha256=str(
            manifest["source_native_evaluation_corpus_index_sha256"]
        ),
        training_event_rows=tuple((row.event_id, row.patient_id) for row in normalized),
        native_event_rows=tuple(),
        training_targets=targets.contiguous(),
        training_target_mask=mask.contiguous(),
        native_targets=empty_targets,
        native_target_mask=empty_mask,
    )
    return LoadedFitOnlyTargetArtifactV13(
        path=source,
        manifest={**manifest, "selection": normalized_selection},
        manifest_sha256=manifest_sha,
        receipt_sha256=receipt_sha,
        events=tuple(normalized),
        snapshot=snapshot,
        shortcut_parameters=shortcuts,
    )


__all__ = (
    "FIT_ONLY_TARGET_MANIFEST",
    "FIT_ONLY_TARGET_MASK",
    "FIT_ONLY_TARGET_RECEIPT",
    "FIT_ONLY_TARGET_RECEIPT_SCHEMA_V13",
    "FIT_ONLY_TARGET_SCHEMA_V13",
    "FIT_ONLY_TARGETS",
    "FitOnlyTargetEventV13",
    "LoadedFitOnlyTargetArtifactV13",
    "compute_fit_only_shortcut_parameters_v13",
    "load_fit_only_target_artifact_v13",
    "validate_fit_only_shortcut_parameters_v13",
)
