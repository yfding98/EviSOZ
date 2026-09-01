"""Atomic formal prediction/control artifacts for ictal producer promotion.

Public materializers accept only strictly loaded production/data artifacts.
They never accept caller-provided logits, targets, masks, event identities, or
control parameters.  Every loader replays the closed tensor bundle and the
fixed control algorithm before issuing an opaque promotion capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from .cached_concept_training import IctalTokenBagDataset
from .concept_oof import IctalConceptOOFProtocolArtifact
from .concept_token_io import load_labram_concept_tokens
from .concept_metrics import IctalConceptMetrics, patient_macro_ictal_metrics
from .data.tusz_training import TUSZIctalTrainingManifest
from .formal_token_corpus import VerifiedFormalTokenCorpusArtifact
from .ictal_native_eval import (
    VerifiedIctalNativeEvalManifestArtifact,
    VerifiedIctalNativeEvalTokenCorpusArtifact,
    build_ictal_native_eval_token_bag_dataset,
)
from .ictal_production import (
    ICTAL_NATIVE_TARGET_SEMANTICS,
    ICTAL_PRODUCTION_CONFIG,
    LoadedIctalProductionRun,
    load_ictal_production_run,
)
from .models.concept_heads import IctalInvolvementHead
from .tusz_token_dataset import build_tusz_ictal_token_bag_dataset


ICTAL_NATIVE_PREDICTION_ARTIFACT_SCHEMA = (
    "soz_ictal_native_prediction_artifact_v1"
)
ICTAL_NATIVE_PREDICTION_RECEIPT_SCHEMA = (
    "soz_ictal_native_prediction_bundle_receipt_v1"
)
ICTAL_CONTROL_PREDICTION_ARTIFACT_SCHEMA = (
    "soz_ictal_control_prediction_artifact_v1"
)
ICTAL_CONTROL_PREDICTION_RECEIPT_SCHEMA = (
    "soz_ictal_control_prediction_bundle_receipt_v1"
)
ICTAL_SCALE_PROBE_ARTIFACT_SCHEMA = "soz_ictal_shared_dev_scale_probe_artifact_v1"
ICTAL_SCALE_PROBE_RECEIPT_SCHEMA = "soz_ictal_shared_dev_scale_probe_receipt_v1"
ICTAL_FOLD_ID_PROBE_ARTIFACT_SCHEMA = (
    "soz_ictal_signal_eligible_fold_identity_probe_artifact_v1"
)
ICTAL_FOLD_ID_PROBE_RECEIPT_SCHEMA = (
    "soz_ictal_signal_eligible_fold_identity_probe_receipt_v1"
)
ICTAL_NATIVE_PREDICTION_MANIFEST_FILENAME = "manifest.json"
ICTAL_NATIVE_PREDICTION_RECEIPT_FILENAME = "receipt.json"
ICTAL_CONTROL_PREDICTION_MANIFEST_FILENAME = "manifest.json"
ICTAL_CONTROL_PREDICTION_RECEIPT_FILENAME = "receipt.json"
ICTAL_SCALE_PROBE_MANIFEST_FILENAME = "manifest.json"
ICTAL_SCALE_PROBE_RECEIPT_FILENAME = "receipt.json"
ICTAL_FOLD_ID_PROBE_MANIFEST_FILENAME = "manifest.json"
ICTAL_FOLD_ID_PROBE_RECEIPT_FILENAME = "receipt.json"
ICTAL_TIME_ONLY_CONTROL = "time_only"
ICTAL_MASK_ONLY_CONTROL = "mask_only"
ICTAL_TIME_ONLY_CONTROL_ALGORITHM = (
    "laplace_smoothed_training_prevalence_by_relative_second_v1"
)
ICTAL_MASK_ONLY_CONTROL_ALGORITHM = (
    "laplace_smoothed_training_prevalence_by_event_mask_density_quartile_v1"
)

_NATIVE_TENSOR_FILENAMES = {
    "full_native_logits": "full_native_logits.npy",
    "native_targets": "native_targets.npy",
    "native_target_mask": "native_target_mask.npy",
    "training_targets": "training_targets.npy",
    "training_target_mask": "training_target_mask.npy",
}
_CONTROL_TENSOR_FILENAME = "control_logits.npy"
_SCALE_SCORE_FILENAME = "shared_dev_scores.npy"
_SCALE_MASK_FILENAME = "deployment_mask.npy"
_FOLD_SCORE_FILENAME = "oof_scores.npy"
_FOLD_MASK_FILENAME = "deployment_mask.npy"
_FOLD_PHASE_FILENAME = "ictal_phase_mask.npy"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SELECTION_RE = re.compile(r"fold[0-4]|final")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_NATIVE_MARKER = object()
_CONTROL_MARKER = object()
_SCALE_PROBE_MARKER = object()
_FOLD_ID_PROBE_MARKER = object()
_LAPLACE_ALPHA = 1.0
_MASK_DENSITY_BOUNDARIES = (0.25, 0.5, 0.75)
_TENSOR_RECORD_FIELDS = frozenset(
    {
        "filename",
        "file_sha256",
        "file_size_bytes",
        "tensor_sha256",
        "shape",
        "dtype",
    }
)
_NATIVE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "selection",
        "production_run_manifest_sha256",
        "checkpoint_manifest_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "native_evaluation_manifest_sha256",
        "native_evaluation_corpus_index_sha256",
        "training_public_patient_ids",
        "native_public_patient_ids",
        "training_event_rows",
        "native_event_rows",
        "tensor_files",
        "native_grid_receipt_sha256",
        "native_support_receipt_sha256",
        "native_fidelity_receipt_sha256",
        "target_semantics",
        "deepsoz_soz_labels_used",
        "private_labels_used",
        "missing_tusz_bins_imputed_as_negative",
    }
)
_NATIVE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_sha256",
        "selection",
        "production_run_manifest_sha256",
        "native_grid_receipt_sha256",
        "native_support_receipt_sha256",
        "native_fidelity_receipt_sha256",
    }
)
_CONTROL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "selection",
        "control_type",
        "control_algorithm",
        "production_run_manifest_sha256",
        "checkpoint_manifest_sha256",
        "native_prediction_artifact_sha256",
        "native_prediction_bundle_receipt_sha256",
        "native_grid_receipt_sha256",
        "native_event_rows",
        "native_public_patient_ids",
        "native_targets_sha256",
        "native_target_mask_sha256",
        "training_targets_sha256",
        "training_target_mask_sha256",
        "fit_parameters",
        "control_logits",
        "control_metrics",
        "control_run_sha256",
        "evaluated_observed_label_count",
        "target_semantics",
        "deepsoz_soz_labels_used",
        "private_labels_used",
        "held_out_targets_used_for_control_fit",
        "missing_tusz_bins_imputed_as_negative",
    }
)
_CONTROL_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_sha256",
        "selection",
        "control_type",
        "production_run_manifest_sha256",
        "native_prediction_artifact_sha256",
        "native_prediction_bundle_receipt_sha256",
        "native_grid_receipt_sha256",
        "control_logits_sha256",
        "control_run_sha256",
    }
)
_PRODUCER_BINDING_FIELDS = frozenset(
    {"selection", "production_run_manifest_sha256", "checkpoint_manifest_sha256"}
)
_SCALE_PROBE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "producer_bindings",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "timeline_context_receipt_sha256",
        "signal_preflight_receipt_sha256",
        "event_registry_sha256",
        "token_corpus_index_sha256",
        "token_corpus_manifest_receipt_sha256",
        "token_corpus_tensor_roster_sha256",
        "foundation_feature_receipt_sha256",
        "event_rows",
        "scores",
        "deployment_mask",
        "scale_alignment_receipt_sha256",
        "target_semantics",
        "score_transform",
        "native_or_soz_labels_used",
        "private_labels_used",
        "source_target_mask_used",
        "source_eval_events_used",
    }
)
_SCALE_PROBE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_sha256",
        "oof_protocol_receipt_sha256",
        "timeline_context_receipt_sha256",
        "token_corpus_index_sha256",
        "scores_sha256",
        "deployment_mask_sha256",
        "scale_alignment_receipt_sha256",
    }
)
_FOLD_ID_PROBE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "producer_bindings",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "timeline_context_receipt_sha256",
        "signal_preflight_receipt_sha256",
        "event_registry_sha256",
        "token_corpus_index_sha256",
        "token_corpus_master_source_manifest_sha256",
        "token_corpus_tensor_roster_sha256",
        "foundation_feature_receipt_sha256",
        "event_rows",
        "scores",
        "deployment_mask",
        "ictal_phase_mask",
        "fold_identity_receipt_sha256",
        "feature_policy",
        "target_semantics",
        "source_target_mask_used",
        "deepsoz_soz_labels_used",
        "private_labels_used",
        "source_dev_events_used",
        "source_eval_events_used",
        "final_producer_used",
    }
)
_FOLD_ID_PROBE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_sha256",
        "oof_protocol_receipt_sha256",
        "timeline_context_receipt_sha256",
        "token_corpus_index_sha256",
        "scores_sha256",
        "deployment_mask_sha256",
        "ictal_phase_mask_sha256",
        "fold_identity_receipt_sha256",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Ictal prediction artifact is not canonical JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _selection(value: object) -> str:
    normalized = str(value).strip().lower()
    if not _SELECTION_RE.fullmatch(normalized):
        raise ValueError("selection must be fold0..fold4 or final")
    return normalized


def _control_type(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {ICTAL_TIME_ONLY_CONTROL, ICTAL_MASK_ONLY_CONTROL}:
        raise ValueError("control_type must be time_only or mask_only")
    return normalized


def _roster(values: Sequence[object], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{field} must be non-empty")
    if normalized != tuple(sorted(normalized)) or len(set(normalized)) != len(
        normalized
    ):
        raise ValueError(f"{field} must be unique and sorted")
    return normalized


def _event_rows(
    values: Sequence[object], *, field: str
) -> tuple[tuple[str, str], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    rows: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{field} rows must be [event_id, patient_id]")
        event_id, patient_id = (str(item).strip() for item in value)
        if not event_id or not patient_id:
            raise ValueError(f"{field} identities cannot be blank")
        rows.append((event_id, patient_id))
    normalized = tuple(rows)
    if not normalized or len({row[0] for row in normalized}) != len(normalized):
        raise ValueError(f"{field} requires unique non-empty event IDs")
    if normalized != tuple(sorted(normalized, key=lambda row: (row[1], row[0]))):
        raise ValueError(
            f"{field} must use canonical patient/event order"
        )
    return normalized


def _tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(value.shape)}|{value.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = value.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_output(path: str | Path) -> Path:
    target = Path(os.path.abspath(path))
    if target.name in {"", ".", ".."}:
        raise ValueError("Prediction artifact output requires a concrete directory")
    for component in (target.parent, *target.parent.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Prediction artifact output cannot traverse symlinks")
    if not target.parent.is_dir():
        raise FileNotFoundError("Prediction artifact output parent does not exist")
    if os.path.lexists(target):
        raise FileExistsError(f"Prediction artifact output already exists: {target}")
    return target


def _strict_directory(path: str | Path, expected_files: set[str]) -> Path:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("Prediction artifact bundle must be a regular directory")
    if {item.name for item in source.iterdir()} != expected_files:
        raise ValueError("Prediction artifact bundle has missing or unknown files")
    return source


def _read_regular_bytes(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    """Read one descriptor once, refusing a final-component symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{path.name} must be a regular unsymlinked file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{path.name} must be a regular file")
        limit = None if maximum_bytes is None else maximum_bytes + 1
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(limit)
        if maximum_bytes is not None and len(raw) > maximum_bytes:
            raise ValueError(f"{path.name} exceeds its maximum size")
        if metadata.st_size != len(raw):
            raise ValueError(f"{path.name} changed while it was read")
        return raw
    finally:
        os.close(descriptor)


def _parse_canonical_json(
    path: Path,
    *,
    expected_fields: frozenset[str],
    raw: bytes | None = None,
) -> dict:
    if raw is None:
        raw = _read_regular_bytes(path, maximum_bytes=_MAX_JSON_BYTES)
    if not 1 <= len(raw) <= _MAX_JSON_BYTES:
        raise ValueError(f"{path.name} has an invalid size")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{path.name} violates its closed schema")
    if _canonical_json_bytes(value) != raw:
        raise ValueError(f"{path.name} is not canonical JSON")
    return value


def _write_tensor(path: Path, name: str, tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    if value.requires_grad or not (
        value.is_floating_point() or value.dtype == torch.bool
    ):
        raise TypeError("Prediction tensors must be detached float/bool tensors")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("Prediction tensor contains non-finite values")
    array = value.numpy()
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    _fsync_file(path)
    raw = path.read_bytes()
    return {
        "filename": path.name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "file_size_bytes": len(raw),
        "tensor_sha256": _tensor_sha256(name, value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _read_tensor(
    source: Path,
    *,
    name: str,
    record: object,
    expected_filename: str,
) -> torch.Tensor:
    if not isinstance(record, dict) or set(record) != _TENSOR_RECORD_FIELDS:
        raise ValueError(f"Tensor record {name} violates its closed schema")
    if record.get("filename") != expected_filename:
        raise ValueError(f"Tensor record {name} changed its filename")
    path = source / expected_filename
    raw = _read_regular_bytes(path)
    if hashlib.sha256(raw).hexdigest() != _require_sha256(
        record.get("file_sha256"), field=f"{name}.file_sha256"
    ):
        raise ValueError(f"Tensor file {name} SHA mismatch")
    size = record.get("file_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size != len(raw):
        raise ValueError(f"Tensor file {name} size mismatch")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Tensor file {name} is not a safe NPY array") from exc
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise ValueError(f"Tensor file {name} has an unsafe dtype")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).clone().contiguous()
    canonical_npy = io.BytesIO()
    np.save(canonical_npy, tensor.numpy(), allow_pickle=False)
    if canonical_npy.getvalue() != raw:
        raise ValueError(f"Tensor file {name} is not canonical deterministic NPY")
    if list(tensor.shape) != record.get("shape") or str(tensor.dtype) != record.get(
        "dtype"
    ):
        raise ValueError(f"Tensor file {name} metadata mismatch")
    if _tensor_sha256(name, tensor) != _require_sha256(
        record.get("tensor_sha256"), field=f"{name}.tensor_sha256"
    ):
        raise ValueError(f"Tensor file {name} logical SHA mismatch")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"Tensor file {name} contains non-finite values")
    return tensor


def _strict_replay_run(run: LoadedIctalProductionRun) -> LoadedIctalProductionRun:
    if not isinstance(run, LoadedIctalProductionRun):
        raise TypeError("production_run must be a strictly loaded run")
    replay = load_ictal_production_run(
        run.path, expected_manifest_sha256=run.manifest_sha256
    )
    if (
        replay.manifest != run.manifest
        or replay.checkpoint.manifest_sha256 != run.checkpoint.manifest_sha256
        or replay.checkpoint.checkpoint_sha256 != run.checkpoint.checkpoint_sha256
    ):
        raise ValueError("Production run changed after strict loading")
    return replay


@dataclass(frozen=True)
class _PredictionGrid:
    native_logits: torch.Tensor
    native_targets: torch.Tensor
    native_target_mask: torch.Tensor
    native_event_rows: tuple[tuple[str, str], ...]
    training_targets: torch.Tensor
    training_target_mask: torch.Tensor
    training_event_rows: tuple[tuple[str, str], ...]


def _head_device(head: IctalInvolvementHead) -> torch.device:
    if not isinstance(head, IctalInvolvementHead):
        raise TypeError("Strict production checkpoint lacks an ictal head")
    devices = {parameter.device for parameter in head.parameters()}
    if len(devices) != 1:
        raise ValueError("Ictal prediction head must occupy one device")
    return next(iter(devices))


def _collect_targets(
    dataset: IctalTokenBagDataset, patient_ids: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[str, str], ...]]:
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    rows: list[tuple[str, str]] = []
    for bag in dataset.iter_subset(patient_ids):
        targets.append(bag.targets.detach().cpu().to(torch.float32).clone())
        masks.append(bag.target_mask.detach().cpu().to(torch.bool).clone())
        rows.extend((event_id, bag.patient_id) for event_id in bag.event_ids)
    values = torch.cat(targets, dim=0).contiguous()
    mask = torch.cat(masks, dim=0).contiguous()
    values[~mask] = 0
    return values, mask, tuple(rows)


@torch.no_grad()
def _collect_native_predictions(
    head: IctalInvolvementHead,
    dataset: IctalTokenBagDataset,
    patient_ids: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[str, str], ...]]:
    device = _head_device(head)
    head.eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    rows: list[tuple[str, str]] = []
    microbatch = ICTAL_PRODUCTION_CONFIG.event_microbatch_size
    for bag in dataset.iter_subset(patient_ids):
        for start in range(0, len(bag.event_ids), microbatch):
            stop = min(start + microbatch, len(bag.event_ids))
            tokens = torch.stack(
                [event.tokens for event in bag.token_events[start:stop]], dim=0
            ).to(device=device)
            prediction = head(tokens.detach()).detach().cpu().to(torch.float32)
            logits.append(prediction)
            targets.append(bag.targets[start:stop].detach().cpu().to(torch.float32))
            masks.append(bag.target_mask[start:stop].detach().cpu().to(torch.bool))
            rows.extend(
                (event_id, bag.patient_id) for event_id in bag.event_ids[start:stop]
            )
    full_logits = torch.cat(logits, dim=0).contiguous()
    values = torch.cat(targets, dim=0).contiguous()
    mask = torch.cat(masks, dim=0).contiguous()
    values[~mask] = 0
    return full_logits, values, mask, tuple(rows)


def _manifest_roster(run: LoadedIctalProductionRun, field: str) -> tuple[str, ...]:
    value = run.manifest.get(field)
    if not isinstance(value, list):
        raise TypeError(f"Production {field} must be a JSON array")
    return _roster(value, field=field)


def _build_datasets(
    run: LoadedIctalProductionRun,
    *,
    training_manifest: TUSZIctalTrainingManifest,
    training_corpus: VerifiedFormalTokenCorpusArtifact,
    native_evaluation_manifest: (
        TUSZIctalTrainingManifest | VerifiedIctalNativeEvalManifestArtifact
    ),
    native_evaluation_corpus: (
        VerifiedFormalTokenCorpusArtifact
        | VerifiedIctalNativeEvalTokenCorpusArtifact
    ),
    edf_root: str | Path,
) -> tuple[IctalTokenBagDataset, IctalTokenBagDataset]:
    manifest = run.manifest
    if not isinstance(training_manifest, TUSZIctalTrainingManifest) or not isinstance(
        training_corpus, VerifiedFormalTokenCorpusArtifact
    ):
        raise TypeError("Training prediction inputs must be strict formal artifacts")
    if (
        training_manifest.manifest_sha256
        != manifest.get("training_manifest_sha256")
        or training_corpus.index_sha256
        != manifest.get("training_corpus_index_sha256")
    ):
        raise ValueError("Training prediction inputs differ from the production run")
    training_dataset = build_tusz_ictal_token_bag_dataset(
        training_manifest, edf_root, training_corpus
    )
    if isinstance(
        native_evaluation_manifest, VerifiedIctalNativeEvalManifestArtifact
    ):
        if not isinstance(
            native_evaluation_corpus, VerifiedIctalNativeEvalTokenCorpusArtifact
        ):
            raise TypeError("Evaluation-only manifest requires evaluation-only corpus")
        native_manifest_sha = native_evaluation_manifest.receipt_sha256
        native_dataset = build_ictal_native_eval_token_bag_dataset(
            native_evaluation_manifest, edf_root, native_evaluation_corpus
        )
    elif isinstance(native_evaluation_manifest, TUSZIctalTrainingManifest):
        if not isinstance(native_evaluation_corpus, VerifiedFormalTokenCorpusArtifact):
            raise TypeError("Fold native manifest requires formal token corpus")
        native_manifest_sha = native_evaluation_manifest.manifest_sha256
        native_dataset = build_tusz_ictal_token_bag_dataset(
            native_evaluation_manifest, edf_root, native_evaluation_corpus
        )
    else:
        raise TypeError("Unsupported native evaluation manifest artifact")
    if (
        native_manifest_sha != manifest.get("native_evaluation_manifest_sha256")
        or native_evaluation_corpus.index_sha256
        != manifest.get("native_evaluation_corpus_index_sha256")
    ):
        raise ValueError("Native prediction inputs differ from the production run")
    if training_dataset.foundation_feature_receipt_sha256 != (
        native_dataset.foundation_feature_receipt_sha256
    ):
        raise ValueError("Training/native prediction datasets use different foundations")
    return training_dataset, native_dataset


def _generate_prediction_grid(
    run: LoadedIctalProductionRun,
    training_dataset: IctalTokenBagDataset,
    native_dataset: IctalTokenBagDataset,
) -> _PredictionGrid:
    training_patients = _manifest_roster(run, "training_source_public_patient_ids")
    native_patients = _manifest_roster(run, "native_evaluation_public_patient_ids")
    if tuple(training_dataset.patient_ids) != training_patients:
        raise ValueError("Training dataset roster differs from production fit roster")
    if not set(native_patients) <= set(native_dataset.patient_ids):
        raise ValueError("Native dataset omits production evaluation patients")
    training_targets, training_mask, training_rows = _collect_targets(
        training_dataset, training_patients
    )
    native_logits, native_targets, native_mask, native_rows = (
        _collect_native_predictions(
            run.checkpoint.head, native_dataset, native_patients
        )
    )
    return _PredictionGrid(
        native_logits=native_logits,
        native_targets=native_targets,
        native_target_mask=native_mask,
        native_event_rows=native_rows,
        training_targets=training_targets,
        training_target_mask=training_mask,
        training_event_rows=training_rows,
    )


def _require_exact_prediction_replay(
    stored: _PredictionGrid,
    replayed: _PredictionGrid,
) -> None:
    """Bind stored bytes to the fixed checkpoint and strict source artifacts."""

    if (
        stored.native_event_rows != replayed.native_event_rows
        or stored.training_event_rows != replayed.training_event_rows
    ):
        raise ValueError(
            "Native prediction event roster differs from strict source replay"
        )
    for field in (
        "native_logits",
        "native_targets",
        "native_target_mask",
        "training_targets",
        "training_target_mask",
    ):
        if not torch.equal(getattr(stored, field), getattr(replayed, field)):
            raise ValueError(
                f"Native prediction {field} differs from strict checkpoint/source replay"
            )


def _verify_native_grid(run: LoadedIctalProductionRun, grid: _PredictionGrid):
    from .ictal_promotion import verify_ictal_native_evaluation_tensors

    return verify_ictal_native_evaluation_tensors(
        production_run=run,
        full_native_logits=grid.native_logits,
        native_targets=grid.native_targets,
        native_target_mask=grid.native_target_mask,
        native_event_ids=tuple(row[0] for row in grid.native_event_rows),
        native_public_patient_ids=tuple(row[1] for row in grid.native_event_rows),
        training_targets=grid.training_targets,
        training_target_mask=grid.training_target_mask,
        training_event_ids=tuple(row[0] for row in grid.training_event_rows),
        training_public_patient_ids=tuple(row[1] for row in grid.training_event_rows),
    )


def _native_grid_receipt(grid: _PredictionGrid) -> str:
    return _canonical_sha256(
        {
            "schema_version": "soz_ictal_native_prediction_grid_v1",
            "native_event_rows": grid.native_event_rows,
            "training_event_rows": grid.training_event_rows,
            "full_native_logits_sha256": _tensor_sha256(
                "full_native_logits", grid.native_logits
            ),
            "native_targets_sha256": _tensor_sha256(
                "native_targets", grid.native_targets
            ),
            "native_target_mask_sha256": _tensor_sha256(
                "native_target_mask", grid.native_target_mask
            ),
            "training_targets_sha256": _tensor_sha256(
                "training_targets", grid.training_targets
            ),
            "training_target_mask_sha256": _tensor_sha256(
                "training_target_mask", grid.training_target_mask
            ),
        }
    )


def _native_expected_files() -> set[str]:
    return {
        ICTAL_NATIVE_PREDICTION_MANIFEST_FILENAME,
        ICTAL_NATIVE_PREDICTION_RECEIPT_FILENAME,
        *_NATIVE_TENSOR_FILENAMES.values(),
    }


def _read_native_bundle(
    path: str | Path,
    *,
    production_run: LoadedIctalProductionRun,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[Path, str, str, dict, _PredictionGrid, object]:
    run = _strict_replay_run(production_run)
    source = _strict_directory(path, _native_expected_files())
    manifest_path = source / ICTAL_NATIVE_PREDICTION_MANIFEST_FILENAME
    receipt_path = source / ICTAL_NATIVE_PREDICTION_RECEIPT_FILENAME
    manifest_raw = _read_regular_bytes(
        manifest_path, maximum_bytes=_MAX_JSON_BYTES
    )
    receipt_raw = _read_regular_bytes(
        receipt_path, maximum_bytes=_MAX_JSON_BYTES
    )
    artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Native prediction artifact SHA mismatch")
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Native prediction bundle receipt SHA mismatch")
    manifest = _parse_canonical_json(
        manifest_path,
        expected_fields=_NATIVE_MANIFEST_FIELDS,
        raw=manifest_raw,
    )
    receipt = _parse_canonical_json(
        receipt_path,
        expected_fields=_NATIVE_RECEIPT_FIELDS,
        raw=receipt_raw,
    )
    if manifest.get("schema_version") != ICTAL_NATIVE_PREDICTION_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported native prediction artifact schema")
    if receipt.get("schema_version") != ICTAL_NATIVE_PREDICTION_RECEIPT_SCHEMA:
        raise ValueError("Unsupported native prediction receipt schema")
    selection = _selection(manifest.get("selection"))
    if selection != _selection(run.manifest.get("selection")):
        raise ValueError("Native prediction selection changed")
    bindings = {
        "production_run_manifest_sha256": run.manifest_sha256,
        "checkpoint_manifest_sha256": run.checkpoint.manifest_sha256,
        "training_manifest_sha256": run.manifest.get("training_manifest_sha256"),
        "training_corpus_index_sha256": run.manifest.get(
            "training_corpus_index_sha256"
        ),
        "native_evaluation_manifest_sha256": run.manifest.get(
            "native_evaluation_manifest_sha256"
        ),
        "native_evaluation_corpus_index_sha256": run.manifest.get(
            "native_evaluation_corpus_index_sha256"
        ),
    }
    for field, expected in bindings.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Native prediction changed production binding {field}")
    training_patients = _roster(
        manifest.get("training_public_patient_ids"),
        field="training_public_patient_ids",
    )
    native_patients = _roster(
        manifest.get("native_public_patient_ids"),
        field="native_public_patient_ids",
    )
    if training_patients != _manifest_roster(
        run, "training_source_public_patient_ids"
    ) or native_patients != _manifest_roster(
        run, "native_evaluation_public_patient_ids"
    ):
        raise ValueError("Native prediction patient roster changed")
    training_rows = _event_rows(
        manifest.get("training_event_rows"), field="training_event_rows"
    )
    native_rows = _event_rows(
        manifest.get("native_event_rows"), field="native_event_rows"
    )
    if tuple(sorted({row[1] for row in training_rows})) != training_patients or tuple(
        sorted({row[1] for row in native_rows})
    ) != native_patients:
        raise ValueError("Native prediction event/patient rosters disagree")
    tensor_records = manifest.get("tensor_files")
    if not isinstance(tensor_records, dict) or set(tensor_records) != set(
        _NATIVE_TENSOR_FILENAMES
    ):
        raise ValueError("Native prediction tensor roster changed")
    tensors = {
        name: _read_tensor(
            source,
            name=name,
            record=tensor_records[name],
            expected_filename=filename,
        )
        for name, filename in _NATIVE_TENSOR_FILENAMES.items()
    }
    grid = _PredictionGrid(
        native_logits=tensors["full_native_logits"],
        native_targets=tensors["native_targets"],
        native_target_mask=tensors["native_target_mask"],
        native_event_rows=native_rows,
        training_targets=tensors["training_targets"],
        training_target_mask=tensors["training_target_mask"],
        training_event_rows=training_rows,
    )
    if grid.native_logits.ndim != 4 or tuple(grid.native_logits.shape[1:]) != (
        20,
        60,
        1,
    ):
        raise ValueError("Native logits must have shape [E,20,60,1]")
    for values, mask, rows, label in (
        (
            grid.native_targets,
            grid.native_target_mask,
            grid.native_event_rows,
            "native",
        ),
        (
            grid.training_targets,
            grid.training_target_mask,
            grid.training_event_rows,
            "training",
        ),
    ):
        if (
            values.dtype != torch.float32
            or mask.dtype != torch.bool
            or tuple(values.shape) != tuple(mask.shape)
            or tuple(values.shape[1:]) != (20, 60)
            or values.shape[0] != len(rows)
        ):
            raise ValueError(f"{label} target grid has invalid shape or dtype")
        observed = values[mask]
        if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
            raise ValueError(f"{label} observed targets must be binary")
        if not torch.equal(values[~mask], torch.zeros_like(values[~mask])):
            raise ValueError(f"{label} unknown targets must use canonical zero fill")
    grid_sha = _native_grid_receipt(grid)
    if manifest.get("native_grid_receipt_sha256") != grid_sha:
        raise ValueError("Native prediction grid receipt SHA mismatch")
    verified = _verify_native_grid(run, grid)
    if manifest.get("native_support_receipt_sha256") != (
        verified.support_receipt_sha256
    ) or manifest.get("native_fidelity_receipt_sha256") != (
        verified.fidelity_receipt_sha256
    ):
        raise ValueError("Native prediction replay receipts changed")
    if (
        manifest.get("target_semantics") != ICTAL_NATIVE_TARGET_SEMANTICS
        or manifest.get("deepsoz_soz_labels_used") is not False
        or manifest.get("private_labels_used") is not False
        or manifest.get("missing_tusz_bins_imputed_as_negative") is not False
    ):
        raise ValueError("Native prediction contains forbidden target semantics")
    expected_receipt = {
        "schema_version": ICTAL_NATIVE_PREDICTION_RECEIPT_SCHEMA,
        "artifact_sha256": artifact_sha,
        "selection": selection,
        "production_run_manifest_sha256": run.manifest_sha256,
        "native_grid_receipt_sha256": grid_sha,
        "native_support_receipt_sha256": verified.support_receipt_sha256,
        "native_fidelity_receipt_sha256": verified.fidelity_receipt_sha256,
    }
    if receipt != expected_receipt:
        raise ValueError("Native prediction receipt does not bind its artifact")
    return source, artifact_sha, receipt_sha, manifest, grid, verified


@dataclass(frozen=True, init=False)
class VerifiedIctalNativePredictionArtifact:
    path: Path
    artifact_sha256: str
    receipt_sha256: str
    selection: str
    production_run: LoadedIctalProductionRun
    production_run_manifest_sha256: str
    native_grid_receipt_sha256: str
    native_evaluation: object

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        artifact_sha256: str,
        receipt_sha256: str,
        production_run: LoadedIctalProductionRun,
        native_grid_receipt_sha256: str,
        native_evaluation: object,
    ) -> None:
        if _verification_marker is not _NATIVE_MARKER:
            raise TypeError(
                "VerifiedIctalNativePredictionArtifact can only be issued by "
                "the strict native-prediction loader"
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self, "artifact_sha256", _require_sha256(artifact_sha256, field="artifact_sha256")
        )
        object.__setattr__(
            self, "receipt_sha256", _require_sha256(receipt_sha256, field="receipt_sha256")
        )
        object.__setattr__(self, "selection", _selection(production_run.manifest.get("selection")))
        object.__setattr__(self, "production_run", production_run)
        object.__setattr__(
            self, "production_run_manifest_sha256", production_run.manifest_sha256
        )
        object.__setattr__(
            self,
            "native_grid_receipt_sha256",
            _require_sha256(native_grid_receipt_sha256, field="native_grid_receipt_sha256"),
        )
        object.__setattr__(self, "native_evaluation", native_evaluation)

    def assert_unchanged(self) -> None:
        _, artifact_sha, receipt_sha, _, grid, verified = _read_native_bundle(
            self.path,
            production_run=self.production_run,
            expected_artifact_sha256=self.artifact_sha256,
            expected_receipt_sha256=self.receipt_sha256,
        )
        if (
            artifact_sha != self.artifact_sha256
            or receipt_sha != self.receipt_sha256
            or _native_grid_receipt(grid) != self.native_grid_receipt_sha256
            or verified.support_receipt_sha256
            != self.native_evaluation.support_receipt_sha256
            or verified.fidelity_receipt_sha256
            != self.native_evaluation.fidelity_receipt_sha256
        ):
            raise ValueError("Verified native prediction artifact changed after load")


def relocate_verified_ictal_native_prediction_artifact(
    artifact: VerifiedIctalNativePredictionArtifact,
    path: str | Path,
) -> VerifiedIctalNativePredictionArtifact:
    """Rebind a strict-loaded artifact after an atomic parent-directory move.

    The native materializer has already regenerated the checkpoint/source grid
    before it can issue ``artifact``.  A parent-directory ``os.rename`` changes
    only its pathname.  This boundary re-reads every final bundle member,
    validates canonical hashes and logical tensors, recomputes the native
    evaluation receipts, and requires the complete identity to match the
    previously issued opaque capability.  It accepts no caller tensors,
    labels, scores, manifests, or hashes.
    """

    if not isinstance(artifact, VerifiedIctalNativePredictionArtifact):
        raise TypeError(
            "artifact must be issued by the strict native-prediction loader"
        )
    source, artifact_sha, receipt_sha, _, grid, verified = _read_native_bundle(
        path,
        production_run=artifact.production_run,
        expected_artifact_sha256=artifact.artifact_sha256,
        expected_receipt_sha256=artifact.receipt_sha256,
    )
    if (
        artifact_sha != artifact.artifact_sha256
        or receipt_sha != artifact.receipt_sha256
        or _native_grid_receipt(grid) != artifact.native_grid_receipt_sha256
        or verified.support_receipt_sha256
        != artifact.native_evaluation.support_receipt_sha256
        or verified.fidelity_receipt_sha256
        != artifact.native_evaluation.fidelity_receipt_sha256
    ):
        raise ValueError("Relocated native prediction differs from verified bytes")
    return VerifiedIctalNativePredictionArtifact(
        _verification_marker=_NATIVE_MARKER,
        path=source,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
        production_run=artifact.production_run,
        native_grid_receipt_sha256=artifact.native_grid_receipt_sha256,
        native_evaluation=verified,
    )


def load_ictal_native_prediction_artifact(
    path: str | Path,
    *,
    production_run: LoadedIctalProductionRun,
    training_manifest: TUSZIctalTrainingManifest,
    training_corpus: VerifiedFormalTokenCorpusArtifact,
    native_evaluation_manifest: (
        TUSZIctalTrainingManifest | VerifiedIctalNativeEvalManifestArtifact
    ),
    native_evaluation_corpus: (
        VerifiedFormalTokenCorpusArtifact
        | VerifiedIctalNativeEvalTokenCorpusArtifact
    ),
    edf_root: str | Path,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedIctalNativePredictionArtifact:
    """Strictly reload and exactly regenerate the checkpoint/source grid."""

    source, artifact_sha, receipt_sha, manifest, stored_grid, verified = _read_native_bundle(
        path,
        production_run=production_run,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    replay = _strict_replay_run(production_run)
    training_dataset, native_dataset = _build_datasets(
        replay,
        training_manifest=training_manifest,
        training_corpus=training_corpus,
        native_evaluation_manifest=native_evaluation_manifest,
        native_evaluation_corpus=native_evaluation_corpus,
        edf_root=edf_root,
    )
    replayed_grid = _generate_prediction_grid(
        replay, training_dataset, native_dataset
    )
    _require_exact_prediction_replay(stored_grid, replayed_grid)
    return VerifiedIctalNativePredictionArtifact(
        _verification_marker=_NATIVE_MARKER,
        path=source,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
        production_run=replay,
        native_grid_receipt_sha256=str(manifest["native_grid_receipt_sha256"]),
        native_evaluation=verified,
    )


def materialize_ictal_native_prediction_artifact(
    *,
    production_run: LoadedIctalProductionRun,
    training_manifest: TUSZIctalTrainingManifest,
    training_corpus: VerifiedFormalTokenCorpusArtifact,
    native_evaluation_manifest: (
        TUSZIctalTrainingManifest | VerifiedIctalNativeEvalManifestArtifact
    ),
    native_evaluation_corpus: (
        VerifiedFormalTokenCorpusArtifact
        | VerifiedIctalNativeEvalTokenCorpusArtifact
    ),
    edf_root: str | Path,
    output_directory: str | Path,
) -> VerifiedIctalNativePredictionArtifact:
    """Run the strict checkpoint itself and atomically publish its native grid."""

    run = _strict_replay_run(production_run)
    training_dataset, native_dataset = _build_datasets(
        run,
        training_manifest=training_manifest,
        training_corpus=training_corpus,
        native_evaluation_manifest=native_evaluation_manifest,
        native_evaluation_corpus=native_evaluation_corpus,
        edf_root=edf_root,
    )
    grid = _generate_prediction_grid(run, training_dataset, native_dataset)
    verified = _verify_native_grid(run, grid)
    target = _safe_output(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        tensor_values = {
            "full_native_logits": grid.native_logits,
            "native_targets": grid.native_targets,
            "native_target_mask": grid.native_target_mask,
            "training_targets": grid.training_targets,
            "training_target_mask": grid.training_target_mask,
        }
        tensor_records = {
            name: _write_tensor(
                temporary / _NATIVE_TENSOR_FILENAMES[name], name, tensor
            )
            for name, tensor in tensor_values.items()
        }
        grid_sha = _native_grid_receipt(grid)
        manifest_payload = {
            "schema_version": ICTAL_NATIVE_PREDICTION_ARTIFACT_SCHEMA,
            "selection": _selection(run.manifest.get("selection")),
            "production_run_manifest_sha256": run.manifest_sha256,
            "checkpoint_manifest_sha256": run.checkpoint.manifest_sha256,
            "training_manifest_sha256": run.manifest["training_manifest_sha256"],
            "training_corpus_index_sha256": run.manifest[
                "training_corpus_index_sha256"
            ],
            "native_evaluation_manifest_sha256": run.manifest[
                "native_evaluation_manifest_sha256"
            ],
            "native_evaluation_corpus_index_sha256": run.manifest[
                "native_evaluation_corpus_index_sha256"
            ],
            "training_public_patient_ids": list(
                _manifest_roster(run, "training_source_public_patient_ids")
            ),
            "native_public_patient_ids": list(
                _manifest_roster(run, "native_evaluation_public_patient_ids")
            ),
            "training_event_rows": [list(row) for row in grid.training_event_rows],
            "native_event_rows": [list(row) for row in grid.native_event_rows],
            "tensor_files": tensor_records,
            "native_grid_receipt_sha256": grid_sha,
            "native_support_receipt_sha256": verified.support_receipt_sha256,
            "native_fidelity_receipt_sha256": verified.fidelity_receipt_sha256,
            "target_semantics": ICTAL_NATIVE_TARGET_SEMANTICS,
            "deepsoz_soz_labels_used": False,
            "private_labels_used": False,
            "missing_tusz_bins_imputed_as_negative": False,
        }
        manifest_raw = _canonical_json_bytes(manifest_payload)
        artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
        receipt_payload = {
            "schema_version": ICTAL_NATIVE_PREDICTION_RECEIPT_SCHEMA,
            "artifact_sha256": artifact_sha,
            "selection": manifest_payload["selection"],
            "production_run_manifest_sha256": run.manifest_sha256,
            "native_grid_receipt_sha256": grid_sha,
            "native_support_receipt_sha256": verified.support_receipt_sha256,
            "native_fidelity_receipt_sha256": verified.fidelity_receipt_sha256,
        }
        receipt_raw = _canonical_json_bytes(receipt_payload)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        manifest_path = temporary / ICTAL_NATIVE_PREDICTION_MANIFEST_FILENAME
        receipt_path = temporary / ICTAL_NATIVE_PREDICTION_RECEIPT_FILENAME
        manifest_path.write_bytes(manifest_raw)
        receipt_path.write_bytes(receipt_raw)
        _fsync_file(manifest_path)
        _fsync_file(receipt_path)
        _fsync_directory(temporary)
        _, _, _, _, staged_grid, _ = _read_native_bundle(
            temporary,
            production_run=run,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
        _require_exact_prediction_replay(staged_grid, grid)
        if os.path.lexists(target):
            raise FileExistsError(f"Prediction artifact output already exists: {target}")
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_ictal_native_prediction_artifact(
            target,
            production_run=run,
            training_manifest=training_manifest,
            training_corpus=training_corpus,
            native_evaluation_manifest=native_evaluation_manifest,
            native_evaluation_corpus=native_evaluation_corpus,
            edf_root=edf_root,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _load_native_grid(
    artifact: VerifiedIctalNativePredictionArtifact,
) -> _PredictionGrid:
    if not isinstance(artifact, VerifiedIctalNativePredictionArtifact):
        raise TypeError("native_prediction must be issued by the strict loader")
    _, _, _, _, grid, _ = _read_native_bundle(
        artifact.path,
        production_run=artifact.production_run,
        expected_artifact_sha256=artifact.artifact_sha256,
        expected_receipt_sha256=artifact.receipt_sha256,
    )
    return grid


def _smoothed_logit(positive: float, observed: float) -> float:
    probability = (positive + _LAPLACE_ALPHA) / (
        observed + 2.0 * _LAPLACE_ALPHA
    )
    return math.log(probability / (1.0 - probability))


def _time_only_control(grid: _PredictionGrid) -> tuple[torch.Tensor, dict]:
    observed = grid.training_target_mask.sum(dim=(0, 1)).to(torch.float64)
    positive = (
        grid.training_targets * grid.training_target_mask
    ).sum(dim=(0, 1)).to(torch.float64)
    logits = torch.tensor(
        [
            _smoothed_logit(float(pos), float(count))
            for pos, count in zip(positive.tolist(), observed.tolist())
        ],
        dtype=torch.float32,
    )
    prediction = logits.view(1, 1, 60, 1).expand(
        grid.native_targets.shape[0], 20, 60, 1
    ).clone()
    parameters = {
        "smoothing_alpha": _LAPLACE_ALPHA,
        "relative_second_logits": logits.tolist(),
        "training_observed_counts_by_second": [int(value) for value in observed],
        "training_positive_counts_by_second": [int(value) for value in positive],
    }
    return prediction, parameters


def _mask_density_bins(mask: torch.Tensor) -> torch.Tensor:
    density = mask.to(torch.float64).mean(dim=(1, 2))
    boundaries = torch.tensor(_MASK_DENSITY_BOUNDARIES, dtype=torch.float64)
    return torch.bucketize(density, boundaries, right=False)


def _mask_only_control(grid: _PredictionGrid) -> tuple[torch.Tensor, dict]:
    train_bins = _mask_density_bins(grid.training_target_mask)
    global_observed = int(grid.training_target_mask.sum().item())
    global_positive = int(
        (grid.training_targets * grid.training_target_mask).sum().item()
    )
    global_logit = _smoothed_logit(global_positive, global_observed)
    bin_logits: list[float] = []
    bin_observed: list[int] = []
    bin_positive: list[int] = []
    for bin_index in range(4):
        selected = train_bins == bin_index
        if selected.any():
            selected_mask = grid.training_target_mask[selected]
            observed = int(selected_mask.sum().item())
            positive = int(
                (grid.training_targets[selected] * selected_mask).sum().item()
            )
            logit = _smoothed_logit(positive, observed)
        else:
            observed = 0
            positive = 0
            logit = global_logit
        bin_logits.append(logit)
        bin_observed.append(observed)
        bin_positive.append(positive)
    native_bins = _mask_density_bins(grid.native_target_mask)
    event_logits = torch.tensor(
        [bin_logits[int(index)] for index in native_bins.tolist()],
        dtype=torch.float32,
    )
    prediction = event_logits.view(-1, 1, 1, 1).expand(
        grid.native_targets.shape[0], 20, 60, 1
    ).clone()
    parameters = {
        "smoothing_alpha": _LAPLACE_ALPHA,
        "mask_density_boundaries": list(_MASK_DENSITY_BOUNDARIES),
        "global_training_logit": global_logit,
        "bin_logits": bin_logits,
        "training_observed_counts_by_bin": bin_observed,
        "training_positive_counts_by_bin": bin_positive,
    }
    return prediction, parameters


def _control_prediction(
    control_type: str, grid: _PredictionGrid
) -> tuple[torch.Tensor, dict, str]:
    normalized = _control_type(control_type)
    if normalized == ICTAL_TIME_ONLY_CONTROL:
        logits, parameters = _time_only_control(grid)
        return logits, parameters, ICTAL_TIME_ONLY_CONTROL_ALGORITHM
    logits, parameters = _mask_only_control(grid)
    return logits, parameters, ICTAL_MASK_ONLY_CONTROL_ALGORITHM


def _patient_index(rows: tuple[tuple[str, str], ...]) -> torch.Tensor:
    roster = tuple(sorted({row[1] for row in rows}))
    index = {patient: position for position, patient in enumerate(roster)}
    return torch.tensor([index[row[1]] for row in rows], dtype=torch.long)


def _metrics_payload(metrics: IctalConceptMetrics) -> dict:
    return asdict(metrics)


def _metrics_from_payload(value: object) -> IctalConceptMetrics:
    if not isinstance(value, dict) or set(value) != set(
        IctalConceptMetrics.__dataclass_fields__
    ):
        raise ValueError("Control metrics violate their closed schema")
    try:
        return IctalConceptMetrics(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Control metrics are invalid") from exc


def _control_run_sha(
    *,
    control_type: str,
    algorithm: str,
    native_artifact: VerifiedIctalNativePredictionArtifact,
    grid: _PredictionGrid,
    fit_parameters: Mapping[str, object],
    control_logits: torch.Tensor,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "soz_ictal_control_run_v1",
            "control_type": control_type,
            "control_algorithm": algorithm,
            "native_prediction_artifact_sha256": native_artifact.artifact_sha256,
            "native_prediction_bundle_receipt_sha256": native_artifact.receipt_sha256,
            "native_grid_receipt_sha256": native_artifact.native_grid_receipt_sha256,
            "training_targets_sha256": _tensor_sha256(
                "training_targets", grid.training_targets
            ),
            "training_target_mask_sha256": _tensor_sha256(
                "training_target_mask", grid.training_target_mask
            ),
            "native_target_mask_sha256": _tensor_sha256(
                "native_target_mask", grid.native_target_mask
            ),
            "fit_parameters": fit_parameters,
            "control_logits_sha256": _tensor_sha256(
                f"{control_type}_logits", control_logits
            ),
        }
    )


def _control_expected_files() -> set[str]:
    return {
        ICTAL_CONTROL_PREDICTION_MANIFEST_FILENAME,
        ICTAL_CONTROL_PREDICTION_RECEIPT_FILENAME,
        _CONTROL_TENSOR_FILENAME,
    }


def _read_control_bundle(
    path: str | Path,
    *,
    native_prediction: VerifiedIctalNativePredictionArtifact,
    expected_control_type: str,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[Path, str, str, dict, torch.Tensor, IctalConceptMetrics]:
    if not isinstance(native_prediction, VerifiedIctalNativePredictionArtifact):
        raise TypeError("native_prediction must be a strict opaque artifact")
    native_prediction.assert_unchanged()
    grid = _load_native_grid(native_prediction)
    control_type = _control_type(expected_control_type)
    source = _strict_directory(path, _control_expected_files())
    manifest_path = source / ICTAL_CONTROL_PREDICTION_MANIFEST_FILENAME
    receipt_path = source / ICTAL_CONTROL_PREDICTION_RECEIPT_FILENAME
    manifest_raw = _read_regular_bytes(
        manifest_path, maximum_bytes=_MAX_JSON_BYTES
    )
    receipt_raw = _read_regular_bytes(
        receipt_path, maximum_bytes=_MAX_JSON_BYTES
    )
    artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Control prediction artifact SHA mismatch")
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Control prediction receipt SHA mismatch")
    manifest = _parse_canonical_json(
        manifest_path,
        expected_fields=_CONTROL_MANIFEST_FIELDS,
        raw=manifest_raw,
    )
    receipt = _parse_canonical_json(
        receipt_path,
        expected_fields=_CONTROL_RECEIPT_FIELDS,
        raw=receipt_raw,
    )
    if manifest.get("schema_version") != ICTAL_CONTROL_PREDICTION_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported control prediction artifact schema")
    if receipt.get("schema_version") != ICTAL_CONTROL_PREDICTION_RECEIPT_SCHEMA:
        raise ValueError("Unsupported control prediction receipt schema")
    if _control_type(manifest.get("control_type")) != control_type:
        raise ValueError("Control prediction type changed")
    expected_algorithm = (
        ICTAL_TIME_ONLY_CONTROL_ALGORITHM
        if control_type == ICTAL_TIME_ONLY_CONTROL
        else ICTAL_MASK_ONLY_CONTROL_ALGORITHM
    )
    if manifest.get("control_algorithm") != expected_algorithm:
        raise ValueError("Control prediction algorithm changed")
    run = native_prediction.production_run
    bindings = {
        "selection": native_prediction.selection,
        "production_run_manifest_sha256": run.manifest_sha256,
        "checkpoint_manifest_sha256": run.checkpoint.manifest_sha256,
        "native_prediction_artifact_sha256": native_prediction.artifact_sha256,
        "native_prediction_bundle_receipt_sha256": native_prediction.receipt_sha256,
        "native_grid_receipt_sha256": native_prediction.native_grid_receipt_sha256,
        "native_targets_sha256": _tensor_sha256(
            "native_targets", grid.native_targets
        ),
        "native_target_mask_sha256": _tensor_sha256(
            "native_target_mask", grid.native_target_mask
        ),
        "training_targets_sha256": _tensor_sha256(
            "training_targets", grid.training_targets
        ),
        "training_target_mask_sha256": _tensor_sha256(
            "training_target_mask", grid.training_target_mask
        ),
    }
    for field, expected in bindings.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Control prediction changed binding {field}")
    rows = _event_rows(manifest.get("native_event_rows"), field="native_event_rows")
    if rows != grid.native_event_rows:
        raise ValueError("Control prediction changed native event grid")
    patients = _roster(
        manifest.get("native_public_patient_ids"), field="native_public_patient_ids"
    )
    if patients != tuple(sorted({row[1] for row in grid.native_event_rows})):
        raise ValueError("Control prediction changed native patient roster")
    expected_logits, expected_parameters, _ = _control_prediction(control_type, grid)
    if manifest.get("fit_parameters") != expected_parameters:
        raise ValueError("Control prediction fit parameters were not replayed")
    logits = _read_tensor(
        source,
        name=f"{control_type}_logits",
        record=manifest.get("control_logits"),
        expected_filename=_CONTROL_TENSOR_FILENAME,
    )
    if not torch.equal(logits, expected_logits):
        raise ValueError("Control logits differ from the fixed replay algorithm")
    metrics = patient_macro_ictal_metrics(
        logits,
        grid.native_targets,
        grid.native_target_mask,
        _patient_index(grid.native_event_rows),
    )
    stored_metrics = _metrics_from_payload(manifest.get("control_metrics"))
    if asdict(metrics) != asdict(stored_metrics):
        raise ValueError("Control prediction metrics changed")
    control_run_sha = _control_run_sha(
        control_type=control_type,
        algorithm=expected_algorithm,
        native_artifact=native_prediction,
        grid=grid,
        fit_parameters=expected_parameters,
        control_logits=logits,
    )
    if manifest.get("control_run_sha256") != control_run_sha:
        raise ValueError("Control run receipt SHA mismatch")
    if manifest.get("evaluated_observed_label_count") != int(
        grid.native_target_mask.sum().item()
    ):
        raise ValueError("Control prediction changed observed native cells")
    if (
        manifest.get("target_semantics") != ICTAL_NATIVE_TARGET_SEMANTICS
        or manifest.get("deepsoz_soz_labels_used") is not False
        or manifest.get("private_labels_used") is not False
        or manifest.get("held_out_targets_used_for_control_fit") is not False
        or manifest.get("missing_tusz_bins_imputed_as_negative") is not False
    ):
        raise ValueError("Control prediction contains forbidden target semantics")
    expected_receipt = {
        "schema_version": ICTAL_CONTROL_PREDICTION_RECEIPT_SCHEMA,
        "artifact_sha256": artifact_sha,
        "selection": native_prediction.selection,
        "control_type": control_type,
        "production_run_manifest_sha256": run.manifest_sha256,
        "native_prediction_artifact_sha256": native_prediction.artifact_sha256,
        "native_prediction_bundle_receipt_sha256": native_prediction.receipt_sha256,
        "native_grid_receipt_sha256": native_prediction.native_grid_receipt_sha256,
        "control_logits_sha256": _tensor_sha256(
            f"{control_type}_logits", logits
        ),
        "control_run_sha256": control_run_sha,
    }
    if receipt != expected_receipt:
        raise ValueError("Control receipt does not bind its artifact")
    return source, artifact_sha, receipt_sha, manifest, logits, metrics


@dataclass(frozen=True, init=False)
class VerifiedIctalControlPredictionArtifact:
    path: Path
    artifact_sha256: str
    receipt_sha256: str
    selection: str
    control_type: str
    control_run_sha256: str
    native_prediction_artifact_sha256: str
    native_prediction_bundle_receipt_sha256: str
    native_grid_receipt_sha256: str
    logits_sha256: str
    metrics: IctalConceptMetrics
    native_prediction: VerifiedIctalNativePredictionArtifact

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        artifact_sha256: str,
        receipt_sha256: str,
        native_prediction: VerifiedIctalNativePredictionArtifact,
        control_type: str,
        control_run_sha256: str,
        logits_sha256: str,
        metrics: IctalConceptMetrics,
    ) -> None:
        if _verification_marker is not _CONTROL_MARKER:
            raise TypeError(
                "VerifiedIctalControlPredictionArtifact can only be issued by "
                "the strict control loader"
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "artifact_sha256", _require_sha256(artifact_sha256, field="artifact_sha256"))
        object.__setattr__(self, "receipt_sha256", _require_sha256(receipt_sha256, field="receipt_sha256"))
        object.__setattr__(self, "selection", native_prediction.selection)
        object.__setattr__(self, "control_type", _control_type(control_type))
        object.__setattr__(self, "control_run_sha256", _require_sha256(control_run_sha256, field="control_run_sha256"))
        object.__setattr__(self, "native_prediction_artifact_sha256", native_prediction.artifact_sha256)
        object.__setattr__(self, "native_prediction_bundle_receipt_sha256", native_prediction.receipt_sha256)
        object.__setattr__(self, "native_grid_receipt_sha256", native_prediction.native_grid_receipt_sha256)
        object.__setattr__(self, "logits_sha256", _require_sha256(logits_sha256, field="logits_sha256"))
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "native_prediction", native_prediction)

    def assert_unchanged(self) -> None:
        _, artifact_sha, receipt_sha, manifest, logits, metrics = _read_control_bundle(
            self.path,
            native_prediction=self.native_prediction,
            expected_control_type=self.control_type,
            expected_artifact_sha256=self.artifact_sha256,
            expected_receipt_sha256=self.receipt_sha256,
        )
        if (
            artifact_sha != self.artifact_sha256
            or receipt_sha != self.receipt_sha256
            or manifest["control_run_sha256"] != self.control_run_sha256
            or _tensor_sha256(f"{self.control_type}_logits", logits)
            != self.logits_sha256
            or asdict(metrics) != asdict(self.metrics)
        ):
            raise ValueError("Verified control prediction changed after load")


def load_ictal_control_prediction_artifact(
    path: str | Path,
    *,
    native_prediction: VerifiedIctalNativePredictionArtifact,
    expected_control_type: str,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedIctalControlPredictionArtifact:
    source, artifact_sha, receipt_sha, manifest, logits, metrics = (
        _read_control_bundle(
            path,
            native_prediction=native_prediction,
            expected_control_type=expected_control_type,
            expected_artifact_sha256=expected_artifact_sha256,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    )
    control_type = _control_type(manifest["control_type"])
    return VerifiedIctalControlPredictionArtifact(
        _verification_marker=_CONTROL_MARKER,
        path=source,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
        native_prediction=native_prediction,
        control_type=control_type,
        control_run_sha256=str(manifest["control_run_sha256"]),
        logits_sha256=_tensor_sha256(f"{control_type}_logits", logits),
        metrics=metrics,
    )


def _materialize_control(
    *,
    native_prediction: VerifiedIctalNativePredictionArtifact,
    expected_native_prediction_artifact_sha256: str,
    expected_native_prediction_receipt_sha256: str,
    control_type: str,
    output_directory: str | Path,
) -> VerifiedIctalControlPredictionArtifact:
    if not isinstance(native_prediction, VerifiedIctalNativePredictionArtifact):
        raise TypeError("native_prediction must be issued by the strict loader")
    if native_prediction.artifact_sha256 != _require_sha256(
        expected_native_prediction_artifact_sha256,
        field="expected_native_prediction_artifact_sha256",
    ) or native_prediction.receipt_sha256 != _require_sha256(
        expected_native_prediction_receipt_sha256,
        field="expected_native_prediction_receipt_sha256",
    ):
        raise ValueError("Control producer received the wrong native artifact")
    native_prediction.assert_unchanged()
    grid = _load_native_grid(native_prediction)
    normalized = _control_type(control_type)
    logits, parameters, algorithm = _control_prediction(normalized, grid)
    metrics = patient_macro_ictal_metrics(
        logits,
        grid.native_targets,
        grid.native_target_mask,
        _patient_index(grid.native_event_rows),
    )
    control_run_sha = _control_run_sha(
        control_type=normalized,
        algorithm=algorithm,
        native_artifact=native_prediction,
        grid=grid,
        fit_parameters=parameters,
        control_logits=logits,
    )
    target = _safe_output(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        tensor_record = _write_tensor(
            temporary / _CONTROL_TENSOR_FILENAME,
            f"{normalized}_logits",
            logits,
        )
        manifest_payload = {
            "schema_version": ICTAL_CONTROL_PREDICTION_ARTIFACT_SCHEMA,
            "selection": native_prediction.selection,
            "control_type": normalized,
            "control_algorithm": algorithm,
            "production_run_manifest_sha256": native_prediction.production_run_manifest_sha256,
            "checkpoint_manifest_sha256": native_prediction.production_run.checkpoint.manifest_sha256,
            "native_prediction_artifact_sha256": native_prediction.artifact_sha256,
            "native_prediction_bundle_receipt_sha256": native_prediction.receipt_sha256,
            "native_grid_receipt_sha256": native_prediction.native_grid_receipt_sha256,
            "native_event_rows": [list(row) for row in grid.native_event_rows],
            "native_public_patient_ids": list(
                tuple(sorted({row[1] for row in grid.native_event_rows}))
            ),
            "native_targets_sha256": _tensor_sha256(
                "native_targets", grid.native_targets
            ),
            "native_target_mask_sha256": _tensor_sha256(
                "native_target_mask", grid.native_target_mask
            ),
            "training_targets_sha256": _tensor_sha256(
                "training_targets", grid.training_targets
            ),
            "training_target_mask_sha256": _tensor_sha256(
                "training_target_mask", grid.training_target_mask
            ),
            "fit_parameters": parameters,
            "control_logits": tensor_record,
            "control_metrics": _metrics_payload(metrics),
            "control_run_sha256": control_run_sha,
            "evaluated_observed_label_count": int(
                grid.native_target_mask.sum().item()
            ),
            "target_semantics": ICTAL_NATIVE_TARGET_SEMANTICS,
            "deepsoz_soz_labels_used": False,
            "private_labels_used": False,
            "held_out_targets_used_for_control_fit": False,
            "missing_tusz_bins_imputed_as_negative": False,
        }
        manifest_raw = _canonical_json_bytes(manifest_payload)
        artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
        receipt_payload = {
            "schema_version": ICTAL_CONTROL_PREDICTION_RECEIPT_SCHEMA,
            "artifact_sha256": artifact_sha,
            "selection": native_prediction.selection,
            "control_type": normalized,
            "production_run_manifest_sha256": native_prediction.production_run_manifest_sha256,
            "native_prediction_artifact_sha256": native_prediction.artifact_sha256,
            "native_prediction_bundle_receipt_sha256": native_prediction.receipt_sha256,
            "native_grid_receipt_sha256": native_prediction.native_grid_receipt_sha256,
            "control_logits_sha256": _tensor_sha256(
                f"{normalized}_logits", logits
            ),
            "control_run_sha256": control_run_sha,
        }
        receipt_raw = _canonical_json_bytes(receipt_payload)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        manifest_path = temporary / ICTAL_CONTROL_PREDICTION_MANIFEST_FILENAME
        receipt_path = temporary / ICTAL_CONTROL_PREDICTION_RECEIPT_FILENAME
        manifest_path.write_bytes(manifest_raw)
        receipt_path.write_bytes(receipt_raw)
        _fsync_file(manifest_path)
        _fsync_file(receipt_path)
        _fsync_directory(temporary)
        load_ictal_control_prediction_artifact(
            temporary,
            native_prediction=native_prediction,
            expected_control_type=normalized,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
        if os.path.lexists(target):
            raise FileExistsError(f"Prediction artifact output already exists: {target}")
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return load_ictal_control_prediction_artifact(
            target,
            native_prediction=native_prediction,
            expected_control_type=normalized,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def materialize_ictal_time_only_control_artifact(
    *,
    native_prediction: VerifiedIctalNativePredictionArtifact,
    expected_native_prediction_artifact_sha256: str,
    expected_native_prediction_receipt_sha256: str,
    output_directory: str | Path,
) -> VerifiedIctalControlPredictionArtifact:
    return _materialize_control(
        native_prediction=native_prediction,
        expected_native_prediction_artifact_sha256=(
            expected_native_prediction_artifact_sha256
        ),
        expected_native_prediction_receipt_sha256=(
            expected_native_prediction_receipt_sha256
        ),
        control_type=ICTAL_TIME_ONLY_CONTROL,
        output_directory=output_directory,
    )


def materialize_ictal_mask_only_control_artifact(
    *,
    native_prediction: VerifiedIctalNativePredictionArtifact,
    expected_native_prediction_artifact_sha256: str,
    expected_native_prediction_receipt_sha256: str,
    output_directory: str | Path,
) -> VerifiedIctalControlPredictionArtifact:
    return _materialize_control(
        native_prediction=native_prediction,
        expected_native_prediction_artifact_sha256=(
            expected_native_prediction_artifact_sha256
        ),
        expected_native_prediction_receipt_sha256=(
            expected_native_prediction_receipt_sha256
        ),
        control_type=ICTAL_MASK_ONLY_CONTROL,
        output_directory=output_directory,
    )


def verified_shortcut_probe_from_artifacts(
    *,
    native_prediction: VerifiedIctalNativePredictionArtifact,
    time_only_control: VerifiedIctalControlPredictionArtifact,
    mask_only_control: VerifiedIctalControlPredictionArtifact,
):
    """Issue a diagnostic shortcut receipt solely from three strict bundles."""

    from .ictal_promotion import verify_ictal_shortcut_prediction_tensors

    if not isinstance(native_prediction, VerifiedIctalNativePredictionArtifact):
        raise TypeError("native_prediction must be a strict opaque artifact")
    for control, expected_type in (
        (time_only_control, ICTAL_TIME_ONLY_CONTROL),
        (mask_only_control, ICTAL_MASK_ONLY_CONTROL),
    ):
        if not isinstance(control, VerifiedIctalControlPredictionArtifact):
            raise TypeError("Shortcut controls must be strict opaque artifacts")
        if control.control_type != expected_type:
            raise ValueError("Shortcut control type is swapped")
        if (
            control.native_prediction_artifact_sha256
            != native_prediction.artifact_sha256
            or control.native_prediction_bundle_receipt_sha256
            != native_prediction.receipt_sha256
            or control.native_grid_receipt_sha256
            != native_prediction.native_grid_receipt_sha256
        ):
            raise ValueError("Shortcut control uses another native prediction grid")
        control.assert_unchanged()
    native_prediction.assert_unchanged()
    grid = _load_native_grid(native_prediction)
    _, _, _, _, time_logits, _ = _read_control_bundle(
        time_only_control.path,
        native_prediction=native_prediction,
        expected_control_type=ICTAL_TIME_ONLY_CONTROL,
        expected_artifact_sha256=time_only_control.artifact_sha256,
        expected_receipt_sha256=time_only_control.receipt_sha256,
    )
    _, _, _, _, mask_logits, _ = _read_control_bundle(
        mask_only_control.path,
        native_prediction=native_prediction,
        expected_control_type=ICTAL_MASK_ONLY_CONTROL,
        expected_artifact_sha256=mask_only_control.artifact_sha256,
        expected_receipt_sha256=mask_only_control.receipt_sha256,
    )
    return verify_ictal_shortcut_prediction_tensors(
        production_run=native_prediction.production_run,
        native_evaluation=native_prediction.native_evaluation,
        full_logits=grid.native_logits,
        time_only_logits=time_logits,
        mask_only_logits=mask_logits,
        native_targets=grid.native_targets,
        native_target_mask=grid.native_target_mask,
        native_event_ids=tuple(row[0] for row in grid.native_event_rows),
        native_public_patient_ids=tuple(row[1] for row in grid.native_event_rows),
        time_only_control_run_sha256=time_only_control.control_run_sha256,
        mask_only_control_run_sha256=mask_only_control.control_run_sha256,
    )


_PROBE_SELECTIONS = ("fold0", "fold1", "fold2", "fold3", "fold4", "final")


@dataclass(frozen=True)
class _ScaleProbeGrid:
    scores: torch.Tensor
    deployment_mask: torch.Tensor
    event_rows: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class _FoldIdProbeGrid:
    scores: torch.Tensor
    deployment_mask: torch.Tensor
    phase_mask: torch.Tensor
    event_rows: tuple[tuple[str, str, str, int], ...]


def _strict_probe_runs(
    production_runs: Sequence[LoadedIctalProductionRun],
) -> tuple[LoadedIctalProductionRun, ...]:
    if isinstance(production_runs, (str, bytes)):
        raise TypeError("production_runs must be a sequence")
    indexed: dict[str, LoadedIctalProductionRun] = {}
    for candidate in production_runs:
        replay = _strict_replay_run(candidate)
        selection = _selection(replay.manifest.get("selection"))
        if selection in indexed:
            raise ValueError(f"Duplicate ictal probe producer: {selection}")
        indexed[selection] = replay
    if tuple(indexed) != _PROBE_SELECTIONS and set(indexed) != set(
        _PROBE_SELECTIONS
    ):
        raise ValueError("Ictal probes require exactly fold0..fold4 and final")
    return tuple(indexed[selection] for selection in _PROBE_SELECTIONS)


def _run_index(
    production_runs: Sequence[LoadedIctalProductionRun],
) -> dict[str, LoadedIctalProductionRun]:
    return {
        _selection(run.manifest.get("selection")): run for run in production_runs
    }


def _require_protocol_run_binding(
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
) -> None:
    if not isinstance(oof_protocol, IctalConceptOOFProtocolArtifact):
        raise TypeError("oof_protocol must be a verified artifact")
    receipt_sha = oof_protocol.protocol.receipt.receipt_sha256
    for run in production_runs:
        if (
            run.manifest.get("oof_protocol_artifact_sha256")
            != oof_protocol.artifact_sha256
            or run.manifest.get("oof_protocol_receipt_sha256") != receipt_sha
        ):
            raise ValueError("Ictal probe producer uses another OOF protocol")


def _strict_corpus_layout(
    path: Path,
    *,
    expected_index_sha256: str,
    event_ids: Sequence[str],
) -> Path:
    source = _strict_directory(path, {"index.json", "events"})
    raw = _read_regular_bytes(source / "index.json", maximum_bytes=64 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != _require_sha256(
        expected_index_sha256, field="expected_index_sha256"
    ):
        raise ValueError("Probe token-corpus index changed after strict loading")
    event_root = source / "events"
    if event_root.is_symlink() or not event_root.is_dir() or event_root.resolve() != event_root:
        raise ValueError("Probe token-corpus events must be a regular directory")
    if {entry.name for entry in event_root.iterdir()} != set(event_ids):
        raise ValueError("Probe token-corpus event directories changed")
    return event_root


def _load_native_probe_tokens(
    corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
) -> dict[str, object]:
    if not isinstance(corpus, VerifiedIctalNativeEvalTokenCorpusArtifact):
        raise TypeError("Scale probe requires a strict evaluation-only token corpus")
    event_ids = tuple(binding.event_id for binding in corpus.events)
    event_root = _strict_corpus_layout(
        corpus.path,
        expected_index_sha256=corpus.index_sha256,
        event_ids=event_ids,
    )
    tokens: dict[str, object] = {}
    for binding in corpus.events:
        expected_path = event_root / binding.event_id
        if binding.bundle_path != expected_path or expected_path.is_symlink():
            raise ValueError("Scale-probe token path changed its canonical event binding")
        token = load_labram_concept_tokens(
            expected_path,
            expected_manifest_sha256=binding.bundle_manifest_sha256,
        )
        checks = (
            token.event_id == binding.event_id,
            token.tensor_sha256 == binding.tensor_sha256,
            token.event_record_sha256 == binding.evaluation_event_record_sha256,
            token.source_concept_manifest_sha256
            == corpus.manifest_receipt_sha256,
            token.foundation_feature_receipt_sha256
            == corpus.foundation_feature_receipt_sha256,
            token.foundation_checkpoint_sha256
            == corpus.foundation_checkpoint_sha256,
        )
        if not all(checks):
            raise ValueError("Scale-probe token changed after corpus verification")
        tokens[binding.event_id] = token
    return tokens


def _load_formal_probe_tokens(
    corpus: VerifiedFormalTokenCorpusArtifact,
) -> tuple[dict[str, object], str]:
    if not isinstance(corpus, VerifiedFormalTokenCorpusArtifact):
        raise TypeError("Fold-ID probe requires a strict formal master corpus")
    event_ids = tuple(binding.event_id for binding in corpus.events)
    event_root = _strict_corpus_layout(
        corpus.path,
        expected_index_sha256=corpus.index_sha256,
        event_ids=event_ids,
    )
    tokens: dict[str, object] = {}
    foundation_receipt_sha: str | None = None
    foundation_checkpoint_sha: str | None = None
    for binding in corpus.events:
        expected_path = event_root / binding.event_id
        if binding.bundle_path != expected_path or expected_path.is_symlink():
            raise ValueError("Fold-ID token path changed its canonical event binding")
        token = load_labram_concept_tokens(
            expected_path,
            expected_manifest_sha256=binding.bundle_manifest_sha256,
        )
        if (
            token.event_id != binding.event_id
            or token.tensor_sha256 != binding.tensor_sha256
            or token.source_concept_manifest_sha256
            != corpus.training_source_manifest_sha256
        ):
            raise ValueError("Fold-ID token changed after corpus verification")
        if foundation_receipt_sha is None:
            foundation_receipt_sha = token.foundation_feature_receipt_sha256
            foundation_checkpoint_sha = token.foundation_checkpoint_sha256
        elif (
            token.foundation_feature_receipt_sha256 != foundation_receipt_sha
            or token.foundation_checkpoint_sha256 != foundation_checkpoint_sha
        ):
            raise ValueError("Fold-ID corpus mixes foundation extractors")
        tokens[binding.event_id] = token
    if foundation_receipt_sha is None:
        raise RuntimeError("Fold-ID corpus has no replayed token")
    return tokens, foundation_receipt_sha


def _formal_token_event_id_for_timeline_record(
    record: object,
    timeline_event: object,
) -> str:
    """Map one DeepSOZ timeline identity to its native TUSZ token identity.

    The two first-party artifacts intentionally use different event-ID
    namespaces for the same official global annotation row:

    ``<record>__ev0003`` (DeepSOZ signal/timeline) and
    ``<record>__global_ictal_0003`` (TUSZ concept corpus).

    Joining those strings directly makes the real fold-ID probe fail even
    though both artifacts bind the same EDF and official global event.  The
    only admissible bridge is therefore the immutable EDF record basename plus
    the verified official ``global_event_index``.  No onset time, target,
    annotation-coverage mask, or caller-provided crosswalk is used here.
    """

    event_id = str(getattr(record, "event_id", "")).strip()
    local_edf_path = str(getattr(record, "local_edf_path", "")).strip()
    timeline_event_id = str(getattr(timeline_event, "event_id", "")).strip()
    timeline_edf_path = str(
        getattr(timeline_event, "relative_edf_path", "")
    ).strip()
    global_event_index = getattr(timeline_event, "global_event_index", None)
    if (
        not event_id
        or timeline_event_id != event_id
        or not local_edf_path
        or timeline_edf_path != local_edf_path
    ):
        raise ValueError(
            "Fold-ID timeline event/registry EDF identity changed"
        )
    if (
        isinstance(global_event_index, bool)
        or not isinstance(global_event_index, int)
        or global_event_index < 0
    ):
        raise ValueError("Fold-ID timeline global event index is invalid")
    edf = PurePosixPath(local_edf_path)
    if (
        edf.is_absolute()
        or len(edf.parts) != 5
        or edf.parts[0] != "train"
        or edf.suffix != ".edf"
        or any(part in {"", ".", ".."} for part in edf.parts)
    ):
        raise ValueError("Fold-ID timeline EDF path is not canonical source-train")
    record_id = edf.stem
    expected_timeline_id = f"{record_id}__ev{global_event_index:04d}"
    if event_id != expected_timeline_id:
        raise ValueError(
            "Fold-ID timeline event ID disagrees with EDF/global-event identity"
        )
    return f"{record_id}__global_ictal_{global_event_index:04d}"


def _score_one(
    run: LoadedIctalProductionRun,
    token: object,
) -> torch.Tensor:
    if not hasattr(token, "tokens"):
        raise TypeError("Probe token loader returned an invalid token capability")
    head = run.checkpoint.head
    device = _head_device(head)
    head.eval()
    with torch.no_grad():
        logits = head(token.tokens.unsqueeze(0).to(device=device)).detach().cpu()
        scores = IctalInvolvementHead.probabilities(logits).squeeze(0).squeeze(-1)
    value = scores.to(torch.float32).contiguous()
    if tuple(value.shape) != (20, 60) or not torch.isfinite(value).all():
        raise ValueError("Ictal probe producer emitted an invalid score grid")
    return value


def _generate_scale_probe_grid(
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
) -> _ScaleProbeGrid:
    from .formal_reasoner_pipeline import VerifiedGlobalTimelineContext

    if not isinstance(timeline_context, VerifiedGlobalTimelineContext):
        raise TypeError("timeline_context must be a strict formal capability")
    timeline_context.assert_unchanged()
    _require_protocol_run_binding(production_runs, oof_protocol)
    runs = _run_index(production_runs)
    final = runs["final"]
    if (
        final.manifest.get("native_evaluation_corpus_index_sha256")
        != source_dev_corpus.index_sha256
        or final.manifest.get("native_evaluation_manifest_sha256")
        != source_dev_corpus.manifest_receipt_sha256
        or source_dev_corpus.signal_preflight_receipt_sha256
        != timeline_context.signal_preflight_receipt_sha256
    ):
        raise ValueError("Scale probe corpus differs from final source-dev lineage")
    for run in production_runs:
        if run.checkpoint.metadata.get("foundation_feature_receipt_sha256") != (
            source_dev_corpus.foundation_feature_receipt_sha256
        ):
            raise ValueError("Scale probe producer/corpus foundations differ")
    expected_records = tuple(
        record
        for record in timeline_context.event_registry
        if record.model_split == "source_dev"
    )
    if not expected_records or any(
        record.model_split in {"source_train", "source_eval"}
        for record in expected_records
    ):
        raise ValueError("Scale probe requires non-empty source-dev events only")
    token_by_event = _load_native_probe_tokens(source_dev_corpus)
    expected_ids = tuple(record.event_id for record in expected_records)
    if set(token_by_event) != set(expected_ids):
        raise ValueError("Scale token corpus must equal complete source-dev event roster")
    crosswalk = dict(oof_protocol.protocol.receipt.target_public_crosswalk)
    public_by_event = {
        binding.event_id: binding.public_patient_id
        for binding in source_dev_corpus.events
    }
    rows: list[tuple[str, str, str]] = []
    score_rows: list[torch.Tensor] = []
    for record in expected_records:
        if record.patient_id not in crosswalk:
            raise ValueError("Scale-probe target patient lacks a public crosswalk")
        public_id = crosswalk[record.patient_id]
        if public_by_event.get(record.event_id) != public_id:
            raise ValueError("Scale-probe token patient differs from the crosswalk")
        rows.append((record.event_id, record.patient_id, public_id))
        token = token_by_event[record.event_id]
        score_rows.append(
            torch.stack(
                [_score_one(runs[selection], token) for selection in _PROBE_SELECTIONS],
                dim=0,
            )
        )
    scores = torch.stack(score_rows, dim=1).to(torch.float32).contiguous()
    mask = torch.ones(
        (len(rows), 20, 60), dtype=torch.bool, device="cpu"
    )
    return _ScaleProbeGrid(
        scores=scores,
        deployment_mask=mask,
        event_rows=tuple(rows),
    )


def _generate_fold_id_probe_grid(
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
) -> _FoldIdProbeGrid:
    from .formal_reasoner_pipeline import VerifiedGlobalTimelineContext

    if not isinstance(timeline_context, VerifiedGlobalTimelineContext):
        raise TypeError("timeline_context must be a strict formal capability")
    timeline_context.assert_unchanged()
    _require_protocol_run_binding(production_runs, oof_protocol)
    runs = _run_index(production_runs)
    for selection in _PROBE_SELECTIONS[:5]:
        run = runs[selection]
        if (
            run.manifest.get("native_evaluation_corpus_index_sha256")
            != source_train_corpus.index_sha256
            or run.manifest.get("native_evaluation_manifest_sha256")
            != source_train_corpus.master_source_manifest_sha256
        ):
            raise ValueError("Fold-ID corpus differs from a fold native master lineage")
    token_by_event, foundation_receipt_sha = _load_formal_probe_tokens(
        source_train_corpus
    )
    for selection in _PROBE_SELECTIONS[:5]:
        if runs[selection].checkpoint.metadata.get(
            "foundation_feature_receipt_sha256"
        ) != foundation_receipt_sha:
            raise ValueError("Fold-ID producer/corpus foundations differ")
    expected_records = tuple(
        record
        for record in timeline_context.event_registry
        if record.model_split == "source_train"
    )
    if not expected_records:
        raise ValueError("Fold-ID probe has no signal-eligible source-train event")
    token_event_id_by_timeline_event: dict[str, str] = {}
    seen_token_event_ids: set[str] = set()
    for record in expected_records:
        token_event_id = _formal_token_event_id_for_timeline_record(
            record,
            timeline_context.timeline_event(record.event_id),
        )
        if token_event_id in seen_token_event_ids:
            raise ValueError(
                "Fold-ID timeline maps multiple events to one TUSZ token event"
            )
        token_event_id_by_timeline_event[record.event_id] = token_event_id
        seen_token_event_ids.add(token_event_id)
    missing = set(token_event_id_by_timeline_event.values()) - set(token_by_event)
    if missing:
        raise ValueError("Fold-ID corpus omits signal-eligible source-train events")
    protocol = oof_protocol.protocol
    crosswalk = dict(protocol.receipt.target_public_crosswalk)
    fold_by_target = {
        patient_id: fold
        for fold, plan in enumerate(protocol.fold_plans)
        for patient_id in plan.held_out_target_patient_ids
    }
    rows: list[tuple[str, str, str, int]] = []
    scores: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    for record in expected_records:
        patient_id = record.patient_id
        if patient_id not in crosswalk or patient_id not in fold_by_target:
            raise ValueError("Fold-ID event patient lacks OOF lineage")
        fold = fold_by_target[patient_id]
        public_id = crosswalk[patient_id]
        edf = PurePosixPath(record.local_edf_path)
        if edf.parts[1] != public_id:
            raise ValueError(
                "Fold-ID timeline EDF patient differs from the protocol crosswalk"
            )
        if public_id not in set(
            _manifest_roster(
                runs[f"fold{fold}"],
                "held_out_exclusion_public_patient_ids",
            )
        ):
            raise ValueError("Fold-ID event was not held out by its producer")
        rows.append((record.event_id, patient_id, public_id, fold))
        scores.append(
            _score_one(
                runs[f"fold{fold}"],
                token_by_event[token_event_id_by_timeline_event[record.event_id]],
            )
        )
        phases.append(timeline_context.phase_mask(record.event_id).to(torch.bool))
    score_grid = torch.stack(scores, dim=0).to(torch.float32).contiguous()
    deployment = torch.ones_like(score_grid, dtype=torch.bool)
    phase_grid = torch.stack(phases, dim=0).to(torch.bool).contiguous()
    return _FoldIdProbeGrid(
        scores=score_grid,
        deployment_mask=deployment,
        phase_mask=phase_grid,
        event_rows=tuple(rows),
    )


def _verify_scale_probe_grid(
    grid: _ScaleProbeGrid,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
):
    from .ictal_promotion import verify_ictal_scale_alignment_tensors

    return verify_ictal_scale_alignment_tensors(
        production_runs=production_runs,
        oof_protocol=oof_protocol,
        timeline_context=timeline_context,
        shared_source_dev_scores={
            selection: grid.scores[index]
            for index, selection in enumerate(_PROBE_SELECTIONS)
        },
        shared_deployment_mask=grid.deployment_mask,
        source_dev_event_ids=tuple(row[0] for row in grid.event_rows),
    )


def _verify_fold_id_probe_grid(
    grid: _FoldIdProbeGrid,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
):
    from .ictal_promotion import _verify_ictal_fold_identity_score_grid

    return _verify_ictal_fold_identity_score_grid(
        production_runs=production_runs,
        oof_protocol=oof_protocol,
        timeline_context=timeline_context,
        source_train_scores=grid.scores,
        deployment_mask=grid.deployment_mask,
        phase_mask=grid.phase_mask,
        source_train_event_ids=tuple(row[0] for row in grid.event_rows),
    )


def _producer_binding_payload(
    production_runs: Sequence[LoadedIctalProductionRun],
) -> list[dict[str, str]]:
    return [
        {
            "selection": _selection(run.manifest.get("selection")),
            "production_run_manifest_sha256": run.manifest_sha256,
            "checkpoint_manifest_sha256": run.checkpoint.manifest_sha256,
        }
        for run in production_runs
    ]


def _validate_producer_binding_payload(
    value: object,
    production_runs: Sequence[LoadedIctalProductionRun],
) -> None:
    if not isinstance(value, list) or any(
        not isinstance(row, dict) or set(row) != _PRODUCER_BINDING_FIELDS
        for row in value
    ):
        raise ValueError("Probe producer bindings violate their closed schema")
    if value != _producer_binding_payload(production_runs):
        raise ValueError("Probe producer bindings changed")


def _scale_expected_files() -> set[str]:
    return {
        ICTAL_SCALE_PROBE_MANIFEST_FILENAME,
        ICTAL_SCALE_PROBE_RECEIPT_FILENAME,
        _SCALE_SCORE_FILENAME,
        _SCALE_MASK_FILENAME,
    }


def _read_scale_probe_bundle(
    path: str | Path,
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[
    Path,
    str,
    str,
    tuple[LoadedIctalProductionRun, ...],
    object,
]:
    runs = _strict_probe_runs(production_runs)
    expected_grid = _generate_scale_probe_grid(
        runs, oof_protocol, timeline_context, source_dev_corpus
    )
    source = _strict_directory(path, _scale_expected_files())
    manifest_raw = _read_regular_bytes(
        source / ICTAL_SCALE_PROBE_MANIFEST_FILENAME,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    receipt_raw = _read_regular_bytes(
        source / ICTAL_SCALE_PROBE_RECEIPT_FILENAME,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Scale-probe artifact SHA mismatch")
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Scale-probe receipt SHA mismatch")
    manifest = _parse_canonical_json(
        source / ICTAL_SCALE_PROBE_MANIFEST_FILENAME,
        expected_fields=_SCALE_PROBE_MANIFEST_FIELDS,
        raw=manifest_raw,
    )
    receipt = _parse_canonical_json(
        source / ICTAL_SCALE_PROBE_RECEIPT_FILENAME,
        expected_fields=_SCALE_PROBE_RECEIPT_FIELDS,
        raw=receipt_raw,
    )
    if manifest.get("schema_version") != ICTAL_SCALE_PROBE_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported scale-probe artifact schema")
    if receipt.get("schema_version") != ICTAL_SCALE_PROBE_RECEIPT_SCHEMA:
        raise ValueError("Unsupported scale-probe receipt schema")
    _validate_producer_binding_payload(manifest.get("producer_bindings"), runs)
    protocol_receipt = oof_protocol.protocol.receipt.receipt_sha256
    bindings = {
        "oof_protocol_artifact_sha256": oof_protocol.artifact_sha256,
        "oof_protocol_receipt_sha256": protocol_receipt,
        "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
        "signal_preflight_receipt_sha256": timeline_context.signal_preflight_receipt_sha256,
        "event_registry_sha256": timeline_context.event_registry.manifest_sha256,
        "token_corpus_index_sha256": source_dev_corpus.index_sha256,
        "token_corpus_manifest_receipt_sha256": source_dev_corpus.manifest_receipt_sha256,
        "token_corpus_tensor_roster_sha256": source_dev_corpus.tensor_roster_sha256,
        "foundation_feature_receipt_sha256": source_dev_corpus.foundation_feature_receipt_sha256,
    }
    for field, expected in bindings.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Scale-probe changed binding {field}")
    expected_rows = [list(row) for row in expected_grid.event_rows]
    if manifest.get("event_rows") != expected_rows:
        raise ValueError("Scale-probe event/patient roster changed")
    stored_scores = _read_tensor(
        source,
        name="shared_dev_scores",
        record=manifest.get("scores"),
        expected_filename=_SCALE_SCORE_FILENAME,
    )
    stored_mask = _read_tensor(
        source,
        name="scale_deployment_mask",
        record=manifest.get("deployment_mask"),
        expected_filename=_SCALE_MASK_FILENAME,
    )
    if (
        stored_scores.dtype != torch.float32
        or tuple(stored_scores.shape)
        != (6, len(expected_grid.event_rows), 20, 60)
        or stored_mask.dtype != torch.bool
        or tuple(stored_mask.shape) != (len(expected_grid.event_rows), 20, 60)
        or not stored_mask.all()
    ):
        raise ValueError("Scale-probe tensors have invalid shape, dtype, or mask")
    if not torch.equal(stored_scores, expected_grid.scores) or not torch.equal(
        stored_mask, expected_grid.deployment_mask
    ):
        raise ValueError("Scale-probe tensors differ from strict checkpoint/token replay")
    stored_grid = _ScaleProbeGrid(
        scores=stored_scores,
        deployment_mask=stored_mask,
        event_rows=expected_grid.event_rows,
    )
    verification = _verify_scale_probe_grid(
        stored_grid, runs, oof_protocol, timeline_context
    )
    if manifest.get("scale_alignment_receipt_sha256") != verification.receipt_sha256:
        raise ValueError("Scale-probe summary differs from patient-level replay")
    if (
        manifest.get("target_semantics")
        != "target_free_shared_source_dev_probe"
        or manifest.get("score_transform")
        != "identity_sigmoid_of_raw_head_logit"
        or manifest.get("native_or_soz_labels_used") is not False
        or manifest.get("private_labels_used") is not False
        or manifest.get("source_target_mask_used") is not False
        or manifest.get("source_eval_events_used") is not False
    ):
        raise ValueError("Scale-probe artifact violates its target-free boundary")
    expected_receipt = {
        "schema_version": ICTAL_SCALE_PROBE_RECEIPT_SCHEMA,
        "artifact_sha256": artifact_sha,
        "oof_protocol_receipt_sha256": protocol_receipt,
        "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
        "token_corpus_index_sha256": source_dev_corpus.index_sha256,
        "scores_sha256": _tensor_sha256("shared_dev_scores", stored_scores),
        "deployment_mask_sha256": _tensor_sha256(
            "scale_deployment_mask", stored_mask
        ),
        "scale_alignment_receipt_sha256": verification.receipt_sha256,
    }
    if receipt != expected_receipt:
        raise ValueError("Scale-probe receipt does not bind its exact replay")
    return source, artifact_sha, receipt_sha, runs, verification


@dataclass(frozen=True, init=False)
class VerifiedIctalScaleProbeArtifact:
    """Opaque promotion authority for one replayed six-producer scale probe."""

    path: Path
    artifact_sha256: str
    bundle_receipt_sha256: str
    production_runs: tuple[LoadedIctalProductionRun, ...]
    oof_protocol: IctalConceptOOFProtocolArtifact
    timeline_context: object
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact
    verification: object

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        artifact_sha256: str,
        bundle_receipt_sha256: str,
        production_runs: Sequence[LoadedIctalProductionRun],
        oof_protocol: IctalConceptOOFProtocolArtifact,
        timeline_context: object,
        source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
        verification: object,
    ) -> None:
        from .ictal_promotion import VerifiedIctalScaleAlignment

        if _verification_marker is not _SCALE_PROBE_MARKER:
            raise TypeError(
                "VerifiedIctalScaleProbeArtifact can only be issued by the strict loader"
            )
        if not isinstance(verification, VerifiedIctalScaleAlignment):
            raise TypeError("Scale-probe artifact lacks a replayed alignment receipt")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "artifact_sha256",
            _require_sha256(artifact_sha256, field="artifact_sha256"),
        )
        object.__setattr__(
            self,
            "bundle_receipt_sha256",
            _require_sha256(bundle_receipt_sha256, field="bundle_receipt_sha256"),
        )
        object.__setattr__(self, "production_runs", tuple(production_runs))
        object.__setattr__(self, "oof_protocol", oof_protocol)
        object.__setattr__(self, "timeline_context", timeline_context)
        object.__setattr__(self, "source_dev_corpus", source_dev_corpus)
        object.__setattr__(self, "verification", verification)

    @property
    def receipt(self):
        return self.verification.receipt

    @property
    def receipt_sha256(self) -> str:
        return self.bundle_receipt_sha256

    @property
    def scale_alignment_receipt_sha256(self) -> str:
        return self.verification.receipt_sha256

    def assert_unchanged(self) -> None:
        _, artifact_sha, bundle_sha, _, verification = _read_scale_probe_bundle(
            self.path,
            production_runs=self.production_runs,
            oof_protocol=self.oof_protocol,
            timeline_context=self.timeline_context,
            source_dev_corpus=self.source_dev_corpus,
            expected_artifact_sha256=self.artifact_sha256,
            expected_receipt_sha256=self.bundle_receipt_sha256,
        )
        if (
            artifact_sha != self.artifact_sha256
            or bundle_sha != self.bundle_receipt_sha256
            or verification.receipt_sha256
            != self.scale_alignment_receipt_sha256
        ):
            raise ValueError("Verified scale-probe artifact changed after loading")


def load_ictal_scale_probe_artifact(
    path: str | Path,
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedIctalScaleProbeArtifact:
    source, artifact_sha, receipt_sha, runs, verification = (
        _read_scale_probe_bundle(
            path,
            production_runs=production_runs,
            oof_protocol=oof_protocol,
            timeline_context=timeline_context,
            source_dev_corpus=source_dev_corpus,
            expected_artifact_sha256=expected_artifact_sha256,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    )
    return VerifiedIctalScaleProbeArtifact(
        _verification_marker=_SCALE_PROBE_MARKER,
        path=source,
        artifact_sha256=artifact_sha,
        bundle_receipt_sha256=receipt_sha,
        production_runs=runs,
        oof_protocol=oof_protocol,
        timeline_context=timeline_context,
        source_dev_corpus=source_dev_corpus,
        verification=verification,
    )


def materialize_ictal_scale_probe_artifact(
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_dev_corpus: VerifiedIctalNativeEvalTokenCorpusArtifact,
    output_directory: str | Path,
) -> VerifiedIctalScaleProbeArtifact:
    """Publish six source-dev score grids; accepts no caller scores or labels."""

    runs = _strict_probe_runs(production_runs)
    grid = _generate_scale_probe_grid(
        runs, oof_protocol, timeline_context, source_dev_corpus
    )
    verification = _verify_scale_probe_grid(
        grid, runs, oof_protocol, timeline_context
    )
    target = _safe_output(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        score_record = _write_tensor(
            temporary / _SCALE_SCORE_FILENAME,
            "shared_dev_scores",
            grid.scores,
        )
        mask_record = _write_tensor(
            temporary / _SCALE_MASK_FILENAME,
            "scale_deployment_mask",
            grid.deployment_mask,
        )
        manifest_payload = {
            "schema_version": ICTAL_SCALE_PROBE_ARTIFACT_SCHEMA,
            "producer_bindings": _producer_binding_payload(runs),
            "oof_protocol_artifact_sha256": oof_protocol.artifact_sha256,
            "oof_protocol_receipt_sha256": oof_protocol.protocol.receipt.receipt_sha256,
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "signal_preflight_receipt_sha256": timeline_context.signal_preflight_receipt_sha256,
            "event_registry_sha256": timeline_context.event_registry.manifest_sha256,
            "token_corpus_index_sha256": source_dev_corpus.index_sha256,
            "token_corpus_manifest_receipt_sha256": source_dev_corpus.manifest_receipt_sha256,
            "token_corpus_tensor_roster_sha256": source_dev_corpus.tensor_roster_sha256,
            "foundation_feature_receipt_sha256": source_dev_corpus.foundation_feature_receipt_sha256,
            "event_rows": [list(row) for row in grid.event_rows],
            "scores": score_record,
            "deployment_mask": mask_record,
            "scale_alignment_receipt_sha256": verification.receipt_sha256,
            "target_semantics": "target_free_shared_source_dev_probe",
            "score_transform": "identity_sigmoid_of_raw_head_logit",
            "native_or_soz_labels_used": False,
            "private_labels_used": False,
            "source_target_mask_used": False,
            "source_eval_events_used": False,
        }
        manifest_raw = _canonical_json_bytes(manifest_payload)
        artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
        receipt_payload = {
            "schema_version": ICTAL_SCALE_PROBE_RECEIPT_SCHEMA,
            "artifact_sha256": artifact_sha,
            "oof_protocol_receipt_sha256": oof_protocol.protocol.receipt.receipt_sha256,
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "token_corpus_index_sha256": source_dev_corpus.index_sha256,
            "scores_sha256": _tensor_sha256("shared_dev_scores", grid.scores),
            "deployment_mask_sha256": _tensor_sha256(
                "scale_deployment_mask", grid.deployment_mask
            ),
            "scale_alignment_receipt_sha256": verification.receipt_sha256,
        }
        receipt_raw = _canonical_json_bytes(receipt_payload)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        manifest_path = temporary / ICTAL_SCALE_PROBE_MANIFEST_FILENAME
        receipt_path = temporary / ICTAL_SCALE_PROBE_RECEIPT_FILENAME
        manifest_path.write_bytes(manifest_raw)
        receipt_path.write_bytes(receipt_raw)
        _fsync_file(manifest_path)
        _fsync_file(receipt_path)
        _fsync_directory(temporary)
        _, _, _, replayed_runs, replayed_verification = _read_scale_probe_bundle(
            temporary,
            production_runs=runs,
            oof_protocol=oof_protocol,
            timeline_context=timeline_context,
            source_dev_corpus=source_dev_corpus,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
        if os.path.lexists(target):
            raise FileExistsError(f"Prediction artifact output already exists: {target}")
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return VerifiedIctalScaleProbeArtifact(
            _verification_marker=_SCALE_PROBE_MARKER,
            path=target,
            artifact_sha256=artifact_sha,
            bundle_receipt_sha256=receipt_sha,
            production_runs=replayed_runs,
            oof_protocol=oof_protocol,
            timeline_context=timeline_context,
            source_dev_corpus=source_dev_corpus,
            verification=replayed_verification,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _fold_id_expected_files() -> set[str]:
    return {
        ICTAL_FOLD_ID_PROBE_MANIFEST_FILENAME,
        ICTAL_FOLD_ID_PROBE_RECEIPT_FILENAME,
        _FOLD_SCORE_FILENAME,
        _FOLD_MASK_FILENAME,
        _FOLD_PHASE_FILENAME,
    }


def _read_fold_id_probe_bundle(
    path: str | Path,
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[
    Path,
    str,
    str,
    tuple[LoadedIctalProductionRun, ...],
    object,
]:
    runs = _strict_probe_runs(production_runs)
    expected_grid = _generate_fold_id_probe_grid(
        runs, oof_protocol, timeline_context, source_train_corpus
    )
    source = _strict_directory(path, _fold_id_expected_files())
    manifest_raw = _read_regular_bytes(
        source / ICTAL_FOLD_ID_PROBE_MANIFEST_FILENAME,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    receipt_raw = _read_regular_bytes(
        source / ICTAL_FOLD_ID_PROBE_RECEIPT_FILENAME,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if artifact_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Fold-ID probe artifact SHA mismatch")
    if receipt_sha != _require_sha256(
        expected_receipt_sha256, field="expected_receipt_sha256"
    ):
        raise ValueError("Fold-ID probe receipt SHA mismatch")
    manifest = _parse_canonical_json(
        source / ICTAL_FOLD_ID_PROBE_MANIFEST_FILENAME,
        expected_fields=_FOLD_ID_PROBE_MANIFEST_FIELDS,
        raw=manifest_raw,
    )
    receipt = _parse_canonical_json(
        source / ICTAL_FOLD_ID_PROBE_RECEIPT_FILENAME,
        expected_fields=_FOLD_ID_PROBE_RECEIPT_FIELDS,
        raw=receipt_raw,
    )
    if manifest.get("schema_version") != ICTAL_FOLD_ID_PROBE_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported fold-ID probe artifact schema")
    if receipt.get("schema_version") != ICTAL_FOLD_ID_PROBE_RECEIPT_SCHEMA:
        raise ValueError("Unsupported fold-ID probe receipt schema")
    _validate_producer_binding_payload(manifest.get("producer_bindings"), runs)
    protocol_receipt = oof_protocol.protocol.receipt.receipt_sha256
    foundation_receipt_sha = str(
        runs[0].checkpoint.metadata["foundation_feature_receipt_sha256"]
    )
    bindings = {
        "oof_protocol_artifact_sha256": oof_protocol.artifact_sha256,
        "oof_protocol_receipt_sha256": protocol_receipt,
        "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
        "signal_preflight_receipt_sha256": timeline_context.signal_preflight_receipt_sha256,
        "event_registry_sha256": timeline_context.event_registry.manifest_sha256,
        "token_corpus_index_sha256": source_train_corpus.index_sha256,
        "token_corpus_master_source_manifest_sha256": source_train_corpus.master_source_manifest_sha256,
        "token_corpus_tensor_roster_sha256": source_train_corpus.tensor_roster_sha256,
        "foundation_feature_receipt_sha256": foundation_receipt_sha,
    }
    for field, expected in bindings.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Fold-ID probe changed binding {field}")
    if manifest.get("event_rows") != [list(row) for row in expected_grid.event_rows]:
        raise ValueError("Fold-ID probe event/patient/fold roster changed")
    scores = _read_tensor(
        source,
        name="oof_ictal_scores",
        record=manifest.get("scores"),
        expected_filename=_FOLD_SCORE_FILENAME,
    )
    mask = _read_tensor(
        source,
        name="fold_deployment_mask",
        record=manifest.get("deployment_mask"),
        expected_filename=_FOLD_MASK_FILENAME,
    )
    phases = _read_tensor(
        source,
        name="fold_ictal_phase_mask",
        record=manifest.get("ictal_phase_mask"),
        expected_filename=_FOLD_PHASE_FILENAME,
    )
    event_count = len(expected_grid.event_rows)
    if (
        scores.dtype != torch.float32
        or tuple(scores.shape) != (event_count, 20, 60)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != tuple(scores.shape)
        or not mask.all()
        or phases.dtype != torch.bool
        or tuple(phases.shape) != (event_count, 15)
    ):
        raise ValueError("Fold-ID probe tensors have invalid shape, dtype, or mask")
    if (
        not torch.equal(scores, expected_grid.scores)
        or not torch.equal(mask, expected_grid.deployment_mask)
        or not torch.equal(phases, expected_grid.phase_mask)
    ):
        raise ValueError("Fold-ID tensors differ from strict OOF checkpoint/token replay")
    stored_grid = _FoldIdProbeGrid(
        scores=scores,
        deployment_mask=mask,
        phase_mask=phases,
        event_rows=expected_grid.event_rows,
    )
    verification = _verify_fold_id_probe_grid(
        stored_grid, runs, oof_protocol, timeline_context
    )
    from .ictal_promotion import ICTAL_FOLD_IDENTITY_FEATURE_POLICY

    if manifest.get("fold_identity_receipt_sha256") != verification.receipt_sha256:
        raise ValueError("Fold-ID probe statistics differ from strict replay")
    if (
        manifest.get("feature_policy") != ICTAL_FOLD_IDENTITY_FEATURE_POLICY
        or manifest.get("target_semantics")
        != "target_free_signal_eligible_source_train_oof_probe"
        or manifest.get("source_target_mask_used") is not False
        or manifest.get("deepsoz_soz_labels_used") is not False
        or manifest.get("private_labels_used") is not False
        or manifest.get("source_dev_events_used") is not False
        or manifest.get("source_eval_events_used") is not False
        or manifest.get("final_producer_used") is not False
    ):
        raise ValueError("Fold-ID probe artifact violates its target-free OOF boundary")
    expected_receipt = {
        "schema_version": ICTAL_FOLD_ID_PROBE_RECEIPT_SCHEMA,
        "artifact_sha256": artifact_sha,
        "oof_protocol_receipt_sha256": protocol_receipt,
        "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
        "token_corpus_index_sha256": source_train_corpus.index_sha256,
        "scores_sha256": _tensor_sha256("oof_ictal_scores", scores),
        "deployment_mask_sha256": _tensor_sha256("fold_deployment_mask", mask),
        "ictal_phase_mask_sha256": _tensor_sha256(
            "fold_ictal_phase_mask", phases
        ),
        "fold_identity_receipt_sha256": verification.receipt_sha256,
    }
    if receipt != expected_receipt:
        raise ValueError("Fold-ID receipt does not bind its exact replay")
    return source, artifact_sha, receipt_sha, runs, verification


@dataclass(frozen=True, init=False)
class VerifiedIctalFoldIdentityProbeArtifact:
    """Opaque promotion authority for the replayed signal-eligible OOF probe."""

    path: Path
    artifact_sha256: str
    bundle_receipt_sha256: str
    production_runs: tuple[LoadedIctalProductionRun, ...]
    oof_protocol: IctalConceptOOFProtocolArtifact
    timeline_context: object
    source_train_corpus: VerifiedFormalTokenCorpusArtifact
    verification: object

    def __init__(
        self,
        *,
        _verification_marker: object,
        path: Path,
        artifact_sha256: str,
        bundle_receipt_sha256: str,
        production_runs: Sequence[LoadedIctalProductionRun],
        oof_protocol: IctalConceptOOFProtocolArtifact,
        timeline_context: object,
        source_train_corpus: VerifiedFormalTokenCorpusArtifact,
        verification: object,
    ) -> None:
        from .ictal_promotion import VerifiedIctalFoldIdentityProbe

        if _verification_marker is not _FOLD_ID_PROBE_MARKER:
            raise TypeError(
                "VerifiedIctalFoldIdentityProbeArtifact can only be issued by the strict loader"
            )
        if not isinstance(verification, VerifiedIctalFoldIdentityProbe):
            raise TypeError("Fold-ID artifact lacks a replayed probe receipt")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "artifact_sha256",
            _require_sha256(artifact_sha256, field="artifact_sha256"),
        )
        object.__setattr__(
            self,
            "bundle_receipt_sha256",
            _require_sha256(bundle_receipt_sha256, field="bundle_receipt_sha256"),
        )
        object.__setattr__(self, "production_runs", tuple(production_runs))
        object.__setattr__(self, "oof_protocol", oof_protocol)
        object.__setattr__(self, "timeline_context", timeline_context)
        object.__setattr__(self, "source_train_corpus", source_train_corpus)
        object.__setattr__(self, "verification", verification)

    @property
    def receipt(self):
        return self.verification.receipt

    @property
    def receipt_sha256(self) -> str:
        return self.bundle_receipt_sha256

    @property
    def fold_identity_receipt_sha256(self) -> str:
        return self.verification.receipt_sha256

    def assert_unchanged(self) -> None:
        _, artifact_sha, bundle_sha, _, verification = _read_fold_id_probe_bundle(
            self.path,
            production_runs=self.production_runs,
            oof_protocol=self.oof_protocol,
            timeline_context=self.timeline_context,
            source_train_corpus=self.source_train_corpus,
            expected_artifact_sha256=self.artifact_sha256,
            expected_receipt_sha256=self.bundle_receipt_sha256,
        )
        if (
            artifact_sha != self.artifact_sha256
            or bundle_sha != self.bundle_receipt_sha256
            or verification.receipt_sha256
            != self.fold_identity_receipt_sha256
        ):
            raise ValueError("Verified fold-ID artifact changed after loading")


def load_ictal_fold_identity_probe_artifact(
    path: str | Path,
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> VerifiedIctalFoldIdentityProbeArtifact:
    source, artifact_sha, receipt_sha, runs, verification = (
        _read_fold_id_probe_bundle(
            path,
            production_runs=production_runs,
            oof_protocol=oof_protocol,
            timeline_context=timeline_context,
            source_train_corpus=source_train_corpus,
            expected_artifact_sha256=expected_artifact_sha256,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    )
    return VerifiedIctalFoldIdentityProbeArtifact(
        _verification_marker=_FOLD_ID_PROBE_MARKER,
        path=source,
        artifact_sha256=artifact_sha,
        bundle_receipt_sha256=receipt_sha,
        production_runs=runs,
        oof_protocol=oof_protocol,
        timeline_context=timeline_context,
        source_train_corpus=source_train_corpus,
        verification=verification,
    )


def materialize_ictal_fold_identity_probe_artifact(
    *,
    production_runs: Sequence[LoadedIctalProductionRun],
    oof_protocol: IctalConceptOOFProtocolArtifact,
    timeline_context: object,
    source_train_corpus: VerifiedFormalTokenCorpusArtifact,
    output_directory: str | Path,
) -> VerifiedIctalFoldIdentityProbeArtifact:
    """Publish actual OOF scores; accepts no feature matrix, mask, or label."""

    from .ictal_promotion import ICTAL_FOLD_IDENTITY_FEATURE_POLICY

    runs = _strict_probe_runs(production_runs)
    grid = _generate_fold_id_probe_grid(
        runs, oof_protocol, timeline_context, source_train_corpus
    )
    verification = _verify_fold_id_probe_grid(
        grid, runs, oof_protocol, timeline_context
    )
    foundation_receipt_sha = str(
        runs[0].checkpoint.metadata["foundation_feature_receipt_sha256"]
    )
    target = _safe_output(output_directory)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        score_record = _write_tensor(
            temporary / _FOLD_SCORE_FILENAME, "oof_ictal_scores", grid.scores
        )
        mask_record = _write_tensor(
            temporary / _FOLD_MASK_FILENAME,
            "fold_deployment_mask",
            grid.deployment_mask,
        )
        phase_record = _write_tensor(
            temporary / _FOLD_PHASE_FILENAME,
            "fold_ictal_phase_mask",
            grid.phase_mask,
        )
        manifest_payload = {
            "schema_version": ICTAL_FOLD_ID_PROBE_ARTIFACT_SCHEMA,
            "producer_bindings": _producer_binding_payload(runs),
            "oof_protocol_artifact_sha256": oof_protocol.artifact_sha256,
            "oof_protocol_receipt_sha256": oof_protocol.protocol.receipt.receipt_sha256,
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "signal_preflight_receipt_sha256": timeline_context.signal_preflight_receipt_sha256,
            "event_registry_sha256": timeline_context.event_registry.manifest_sha256,
            "token_corpus_index_sha256": source_train_corpus.index_sha256,
            "token_corpus_master_source_manifest_sha256": source_train_corpus.master_source_manifest_sha256,
            "token_corpus_tensor_roster_sha256": source_train_corpus.tensor_roster_sha256,
            "foundation_feature_receipt_sha256": foundation_receipt_sha,
            "event_rows": [list(row) for row in grid.event_rows],
            "scores": score_record,
            "deployment_mask": mask_record,
            "ictal_phase_mask": phase_record,
            "fold_identity_receipt_sha256": verification.receipt_sha256,
            "feature_policy": ICTAL_FOLD_IDENTITY_FEATURE_POLICY,
            "target_semantics": "target_free_signal_eligible_source_train_oof_probe",
            "source_target_mask_used": False,
            "deepsoz_soz_labels_used": False,
            "private_labels_used": False,
            "source_dev_events_used": False,
            "source_eval_events_used": False,
            "final_producer_used": False,
        }
        manifest_raw = _canonical_json_bytes(manifest_payload)
        artifact_sha = hashlib.sha256(manifest_raw).hexdigest()
        receipt_payload = {
            "schema_version": ICTAL_FOLD_ID_PROBE_RECEIPT_SCHEMA,
            "artifact_sha256": artifact_sha,
            "oof_protocol_receipt_sha256": oof_protocol.protocol.receipt.receipt_sha256,
            "timeline_context_receipt_sha256": timeline_context.receipt_sha256,
            "token_corpus_index_sha256": source_train_corpus.index_sha256,
            "scores_sha256": _tensor_sha256("oof_ictal_scores", grid.scores),
            "deployment_mask_sha256": _tensor_sha256(
                "fold_deployment_mask", grid.deployment_mask
            ),
            "ictal_phase_mask_sha256": _tensor_sha256(
                "fold_ictal_phase_mask", grid.phase_mask
            ),
            "fold_identity_receipt_sha256": verification.receipt_sha256,
        }
        receipt_raw = _canonical_json_bytes(receipt_payload)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        manifest_path = temporary / ICTAL_FOLD_ID_PROBE_MANIFEST_FILENAME
        receipt_path = temporary / ICTAL_FOLD_ID_PROBE_RECEIPT_FILENAME
        manifest_path.write_bytes(manifest_raw)
        receipt_path.write_bytes(receipt_raw)
        _fsync_file(manifest_path)
        _fsync_file(receipt_path)
        _fsync_directory(temporary)
        _, _, _, replayed_runs, replayed_verification = _read_fold_id_probe_bundle(
            temporary,
            production_runs=runs,
            oof_protocol=oof_protocol,
            timeline_context=timeline_context,
            source_train_corpus=source_train_corpus,
            expected_artifact_sha256=artifact_sha,
            expected_receipt_sha256=receipt_sha,
        )
        if os.path.lexists(target):
            raise FileExistsError(f"Prediction artifact output already exists: {target}")
        os.rename(temporary, target)
        published = True
        _fsync_directory(target.parent)
        return VerifiedIctalFoldIdentityProbeArtifact(
            _verification_marker=_FOLD_ID_PROBE_MARKER,
            path=target,
            artifact_sha256=artifact_sha,
            bundle_receipt_sha256=receipt_sha,
            production_runs=replayed_runs,
            oof_protocol=oof_protocol,
            timeline_context=timeline_context,
            source_train_corpus=source_train_corpus,
            verification=replayed_verification,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "ICTAL_CONTROL_PREDICTION_ARTIFACT_SCHEMA",
    "ICTAL_CONTROL_PREDICTION_RECEIPT_SCHEMA",
    "ICTAL_MASK_ONLY_CONTROL",
    "ICTAL_MASK_ONLY_CONTROL_ALGORITHM",
    "ICTAL_FOLD_ID_PROBE_ARTIFACT_SCHEMA",
    "ICTAL_FOLD_ID_PROBE_RECEIPT_SCHEMA",
    "ICTAL_NATIVE_PREDICTION_ARTIFACT_SCHEMA",
    "ICTAL_NATIVE_PREDICTION_RECEIPT_SCHEMA",
    "ICTAL_TIME_ONLY_CONTROL",
    "ICTAL_TIME_ONLY_CONTROL_ALGORITHM",
    "VerifiedIctalControlPredictionArtifact",
    "VerifiedIctalFoldIdentityProbeArtifact",
    "VerifiedIctalNativePredictionArtifact",
    "VerifiedIctalScaleProbeArtifact",
    "ICTAL_SCALE_PROBE_ARTIFACT_SCHEMA",
    "ICTAL_SCALE_PROBE_RECEIPT_SCHEMA",
    "load_ictal_control_prediction_artifact",
    "load_ictal_fold_identity_probe_artifact",
    "load_ictal_native_prediction_artifact",
    "load_ictal_scale_probe_artifact",
    "materialize_ictal_fold_identity_probe_artifact",
    "materialize_ictal_mask_only_control_artifact",
    "materialize_ictal_native_prediction_artifact",
    "materialize_ictal_scale_probe_artifact",
    "materialize_ictal_time_only_control_artifact",
    "relocate_verified_ictal_native_prediction_artifact",
    "verified_shortcut_probe_from_artifacts",
]
