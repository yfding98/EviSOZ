"""Split-safe disk input and trainer skeleton for BA-IEG segmental supervision.

The manifest is one content-bound, patient-disjoint inventory spanning only
public ``source_train`` and ``source_dev``.  ``source_eval`` and every private
split are deliberately unrepresentable.  Batch packing uses input token counts
only; target status, interval and censoring fields never influence sampling.

Event inputs are stored as strict JSON metadata plus a non-pickle NPZ tensor
bundle (``allow_pickle=False``).  Boundary targets live in a separate strict
JSON artifact and are joined only after each input has replayed its registered
``BAIEGEventTokens`` receipt.  The optimizer entry point accepts source-train
batches only, while source-dev has a no-gradient calibration-forward route.
Raw annotation-resolution intervals remain unchanged through loading and
bundling; the supervision module projects them onto target-free physical-time
support only after the model has frozen its lattice in ``forward``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
from typing import Any, Final, Iterator, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_SEGMENTAL_CENSOR_REASONS,
    BAIEGPermissionSplitSegmentalStateOutput,
    BAIEGSegmentalBoundaryContext,
    build_ba_ieg_segmental_boundary_context,
)
from .ba_ieg_permission_split_segmental_supervision_v1 import (
    BAIEGPermissionSplitSegmentalLossOutputV1,
    BAIEGSegmentalEventTargetV1,
    BAIEGSegmentalTargetBundleV1,
    BAIEGSegmentalTargetFirewallV1,
    build_ba_ieg_segmental_target_bundle_v1,
    permission_split_segmental_training_loss_v1,
)
from .ba_ieg_training_contract import (
    BAIEGCollatedEventBatch,
    BAIEGEventTokens,
    collate_ba_ieg_events,
)


BA_IEG_SEGMENTAL_DISK_MANIFEST_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_segmental_public_disk_manifest_v1"
BA_IEG_SEGMENTAL_DISK_MANIFEST_METHOD_ID: Final[
    str
] = "patient_disjoint_source_train_dev_target_separated_disk_inventory_v1"
BA_IEG_SEGMENTAL_DISK_INPUT_METADATA_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_segmental_event_input_metadata_v1"
BA_IEG_SEGMENTAL_DISK_TARGET_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_segmental_event_target_disk_v1"

_ALLOWED_SPLITS: Final[frozenset[str]] = frozenset({"source_train", "source_dev"})
_PURPOSE_SPLIT: Final[Mapping[str, str]] = {
    "optimize": "source_train",
    "calibrate": "source_dev",
}
_PUBLIC_DISK_TARGET_AUTHORITIES: Final[Mapping[str, frozenset[str]]] = {
    "source_train": frozenset({"public_seizure_interval"}),
    "source_dev": frozenset(
        {
            "public_seizure_interval",
            "source_development_eeg_expert_atomic_boundary",
        }
    ),
}
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_MAXIMUM_EVENT_TENSOR_BYTES: Final[int] = 2 * 1024 * 1024 * 1024
_TENSOR_FIELDS: Final[tuple[str, ...]] = (
    "physical_xyz",
    "physical_xyz_mask",
    "physical_evidence_mask",
    "view_future_sample_access",
    "view_onset_evidence_authorized",
    "unit_view_index",
    "unit_reference_matrix",
    "unit_evidence_mask",
    "unit_family_mask",
    "token_values",
    "token_feature_mask",
    "token_time_bounds_seconds",
    "token_unit_index",
    "token_view_index",
    "token_scale_index",
    "token_signal_mask",
    "token_family_mask",
    "phase_posterior",
)
_EVENT_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "analysis_interval_seconds",
        "navigation_anchor_seconds",
        "canonical_receipt_sha256",
        "adaptive_window_receipt_sha256",
        "encoder_implementation_id",
        "encoder_lineage",
        "encoder_receipt_sha256",
        "physical_electrode_ids",
        "view_ids",
        "view_roles",
        "view_effective_temporal_roles",
        "view_dependency_policies",
        "view_temporal_evidence_sha256s",
        "view_receipt_sha256s",
        "view_transform_sha256s",
        "reference_families",
        "unit_ids",
        "unit_source_ids",
        "unit_types",
    }
)
_MANIFEST_ROW_INPUT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "input_metadata_relative_path",
        "input_metadata_file_sha256",
        "input_tensors_relative_path",
        "input_tensors_file_sha256",
        "target_relative_path",
        "target_file_sha256",
        "input_event_receipt_sha256",
        "target_receipt_sha256",
        "adaptive_acquisition_receipt_sha256",
        "target_independent_candidate_roster_receipt_sha256",
        "source_reference_receipt_sha256",
        "token_count",
    }
)
_PROVENANCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "source_dataset_id",
        "source_dataset_version",
        "source_corpus_manifest_sha256",
        "source_patient_split_receipt_sha256",
        "input_materializer_code_sha256",
        "target_materializer_code_sha256",
        "target_materialization_policy_sha256",
    }
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{context} must be a positive integer")
    return value


def _relative_artifact_path(value: object, context: str, suffix: str) -> str:
    text = _identifier(value, context)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != text
        or path.suffix != suffix
    ):
        raise ValueError(f"{context} must be a canonical relative {suffix} path")
    return text


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = (root_resolved / relative_path).resolve(strict=True)
    if candidate == root_resolved or root_resolved not in candidate.parents:
        raise ValueError("disk artifact escapes the manifest directory")
    if not candidate.is_file():
        raise ValueError("disk artifact is not a regular file")
    return candidate


def _normalize_provenance(value: object) -> dict[str, str]:
    if type(value) is not dict or set(value) != _PROVENANCE_FIELDS:
        raise ValueError("segmental disk provenance has missing or unknown fields")
    result = deepcopy(value)
    for name in ("source_dataset_id", "source_dataset_version"):
        result[name] = _identifier(result[name], name)
    dataset_id = result["source_dataset_id"].upper()
    if dataset_id != "TUSZ" and not dataset_id.startswith("TUSZ-PUBLIC-"):
        raise ValueError("segmental disk supervision is restricted to public TUSZ")
    for name in _PROVENANCE_FIELDS.difference(
        {"source_dataset_id", "source_dataset_version"}
    ):
        result[name] = _sha256(result[name], name)
    return result


def _normalize_manifest_row(value: object, index: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _MANIFEST_ROW_INPUT_FIELDS:
        raise ValueError(f"segmental disk manifest row {index} has invalid fields")
    row = deepcopy(value)
    for name in ("event_id", "recording_id", "patient_uid"):
        row[name] = _identifier(row[name], f"row {index} {name}")
    if row["model_split"] not in _ALLOWED_SPLITS:
        raise ValueError(
            "segmental disk manifest accepts source_train/source_dev only; "
            "source_eval and private labels are forbidden"
        )
    for name, suffix in (
        ("input_metadata_relative_path", ".json"),
        ("input_tensors_relative_path", ".npz"),
        ("target_relative_path", ".json"),
    ):
        row[name] = _relative_artifact_path(row[name], f"row {index} {name}", suffix)
    for name in (
        "input_metadata_file_sha256",
        "input_tensors_file_sha256",
        "target_file_sha256",
        "input_event_receipt_sha256",
        "target_receipt_sha256",
        "adaptive_acquisition_receipt_sha256",
        "target_independent_candidate_roster_receipt_sha256",
        "source_reference_receipt_sha256",
    ):
        row[name] = _sha256(row[name], f"row {index} {name}")
    row["token_count"] = _positive_integer(row["token_count"], "token_count")
    row["row_receipt_sha256"] = _canonical_sha256(row)
    return row


def _assemble_ba_ieg_segmental_disk_manifest_v1(
    *, rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TypeError("segmental disk manifest rows must be a non-empty sequence")
    normalized = [
        _normalize_manifest_row(dict(row), index) for index, row in enumerate(rows)
    ]
    normalized.sort(
        key=lambda row: (
            row["model_split"],
            row["patient_uid"],
            row["recording_id"],
            row["event_id"],
        )
    )
    event_ids = [row["event_id"] for row in normalized]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("segmental disk event IDs must be globally unique")
    all_paths = [
        row[name]
        for row in normalized
        for name in (
            "input_metadata_relative_path",
            "input_tensors_relative_path",
            "target_relative_path",
        )
    ]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("segmental disk artifact paths must be globally unique")
    patient_split: dict[str, str] = {}
    for row in normalized:
        previous = patient_split.setdefault(row["patient_uid"], row["model_split"])
        if previous != row["model_split"]:
            raise ValueError("one patient crosses source_train/source_dev")
    if {row["model_split"] for row in normalized} != _ALLOWED_SPLITS:
        raise ValueError(
            "segmental disk manifest must freeze both source_train and source_dev"
        )
    candidate_rosters = {
        row["target_independent_candidate_roster_receipt_sha256"] for row in normalized
    }
    if len(candidate_rosters) != 1:
        raise ValueError(
            "one disk manifest must share one target-independent candidate roster"
        )
    split_rosters: dict[str, Any] = {}
    for split in sorted(_ALLOWED_SPLITS):
        selected = [row for row in normalized if row["model_split"] == split]
        patients = sorted({row["patient_uid"] for row in selected})
        events = [row["event_id"] for row in selected]
        split_rosters[split] = {
            "patient_count": len(patients),
            "event_count": len(events),
            "patient_uids": patients,
            "event_ids": events,
            "patient_roster_sha256": _canonical_sha256(patients),
            "event_roster_sha256": _canonical_sha256(events),
        }
    body: dict[str, Any] = {
        "schema_version": BA_IEG_SEGMENTAL_DISK_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "BAIEG-SEGMENTAL-DISK-MANIFEST-PENDING",
        "method_id": BA_IEG_SEGMENTAL_DISK_MANIFEST_METHOD_ID,
        "provenance": _normalize_provenance(dict(provenance)),
        "rows": normalized,
        "row_roster_sha256": _canonical_sha256(normalized),
        "split_rosters": split_rosters,
        "target_independent_candidate_roster_receipt_sha256": next(
            iter(candidate_rosters)
        ),
        "scope_receipt": {
            "public_tusz_source_only": True,
            "source_train_and_source_dev_only": True,
            "source_eval_rows_present": False,
            "private_rows_or_labels_present": False,
            "patient_disjoint_split_verified": True,
            "target_status_or_interval_used_for_batching": False,
            "targets_separate_from_model_input_artifacts": True,
            "source_reference_available_to_model_forward": False,
            "source_dev_gradient_updates_authorized": False,
            "production_or_private_route_authorized": False,
        },
    }
    body["manifest_id"] = "BAIEGSEGDISK-" + _canonical_sha256(body)[:24]
    return body


def build_ba_ieg_segmental_disk_manifest_v1(
    *, rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one public source-train/dev patient-disjoint disk inventory."""

    return validate_ba_ieg_segmental_disk_manifest_v1(
        _assemble_ba_ieg_segmental_disk_manifest_v1(rows=rows, provenance=provenance)
    )


def validate_ba_ieg_segmental_disk_manifest_v1(payload: object) -> dict[str, Any]:
    """Replay manifest schema, provenance, split isolation and content binding."""

    required = {
        "schema_version",
        "manifest_id",
        "method_id",
        "provenance",
        "rows",
        "row_roster_sha256",
        "split_rosters",
        "target_independent_candidate_roster_receipt_sha256",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("segmental disk manifest has missing or unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != BA_IEG_SEGMENTAL_DISK_MANIFEST_SCHEMA_VERSION
        or data["method_id"] != BA_IEG_SEGMENTAL_DISK_MANIFEST_METHOD_ID
    ):
        raise ValueError("segmental disk manifest schema/method drifted")
    provenance = _normalize_provenance(data["provenance"])
    rows = data["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("segmental disk manifest has no rows")
    raw_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != _MANIFEST_ROW_INPUT_FIELDS | {
            "row_receipt_sha256"
        }:
            raise ValueError(f"segmental disk manifest row {index} drifted")
        raw = {name: row[name] for name in _MANIFEST_ROW_INPUT_FIELDS}
        normalized = _normalize_manifest_row(raw, index)
        if normalized != row:
            raise ValueError("segmental disk row receipt or normalization drifted")
        raw_rows.append(raw)
    replayed = _assemble_ba_ieg_segmental_disk_manifest_v1(
        rows=raw_rows, provenance=provenance
    )
    if replayed != data:
        raise ValueError("segmental disk manifest is not canonical or content-bound")
    return data


def ba_ieg_segmental_event_tensor_arrays_v1(
    event: BAIEGEventTokens,
) -> dict[str, np.ndarray]:
    """Project the exact model-input tensors for a non-pickle NPZ artifact."""

    if not isinstance(event, BAIEGEventTokens):
        raise TypeError("segmental disk tensor export requires BAIEGEventTokens")
    event.verify_integrity()
    if event.deterministic_targets is not None:
        raise ValueError(
            "segmental disk input excludes deterministic supervision sidecars"
        )
    return {
        name: getattr(event, name).detach().cpu().contiguous().numpy().copy()
        for name in _TENSOR_FIELDS
    }


def ba_ieg_segmental_event_input_metadata_v1(
    event: BAIEGEventTokens,
    *,
    adaptive_acquisition_receipt_sha256: str,
    quality_gap_intervals_seconds: Sequence[Sequence[float]] = (),
    left_censor_reason_code: str = "none",
    right_censor_reason_code: str = "none",
) -> dict[str, Any]:
    """Build strict target-free metadata accompanying one event NPZ."""

    if not isinstance(event, BAIEGEventTokens):
        raise TypeError("segmental disk metadata export requires BAIEGEventTokens")
    event.verify_integrity()
    if event.model_split not in _ALLOWED_SPLITS:
        raise ValueError("disk metadata export accepts source_train/source_dev only")
    if event.deterministic_targets is not None:
        raise ValueError("disk metadata cannot carry deterministic target sidecars")
    acquisition = _sha256(
        adaptive_acquisition_receipt_sha256,
        "adaptive_acquisition_receipt_sha256",
    )
    gaps: list[list[float]] = []
    previous_stop: float | None = None
    for index, value in enumerate(quality_gap_intervals_seconds):
        if isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(f"quality gap {index} must be a two-item interval")
        start, stop = float(value[0]), float(value[1])
        if (
            not math.isfinite(start)
            or not math.isfinite(stop)
            or stop <= start
            or (previous_stop is not None and start < previous_stop)
        ):
            raise ValueError("quality gaps must be finite, sorted and non-overlapping")
        gaps.append([start, stop])
        previous_stop = stop
    for side, code in (
        ("left", left_censor_reason_code),
        ("right", right_censor_reason_code),
    ):
        if code not in BA_IEG_SEGMENTAL_CENSOR_REASONS:
            raise ValueError(f"unsupported {side} censor reason code")
    event_metadata = {
        "event_id": event.event_id,
        "recording_id": event.recording_id,
        "patient_uid": event.patient_uid,
        "model_split": event.model_split,
        "analysis_interval_seconds": list(event.analysis_interval_seconds),
        "navigation_anchor_seconds": event.navigation_anchor_seconds,
        "canonical_receipt_sha256": event.canonical_receipt_sha256,
        "adaptive_window_receipt_sha256": event.adaptive_window_receipt_sha256,
        "encoder_implementation_id": event.encoder_implementation_id,
        "encoder_lineage": event.encoder_lineage,
        "encoder_receipt_sha256": event.encoder_receipt_sha256,
        "physical_electrode_ids": list(event.physical_electrode_ids),
        "view_ids": list(event.view_ids),
        "view_roles": list(event.view_roles),
        "view_effective_temporal_roles": list(event.view_effective_temporal_roles),
        "view_dependency_policies": list(event.view_dependency_policies),
        "view_temporal_evidence_sha256s": list(event.view_temporal_evidence_sha256s),
        "view_receipt_sha256s": list(event.view_receipt_sha256s),
        "view_transform_sha256s": list(event.view_transform_sha256s),
        "reference_families": list(event.reference_families),
        "unit_ids": list(event.unit_ids),
        "unit_source_ids": list(event.unit_source_ids),
        "unit_types": list(event.unit_types),
    }
    return {
        "schema_version": BA_IEG_SEGMENTAL_DISK_INPUT_METADATA_SCHEMA_VERSION,
        "input_event_receipt_sha256": event.input_receipt_sha256,
        "event_metadata": event_metadata,
        "boundary_context_input": {
            "adaptive_acquisition_receipt_sha256": acquisition,
            "quality_gap_intervals_seconds": gaps,
            "left_censor_reason_code": left_censor_reason_code,
            "right_censor_reason_code": right_censor_reason_code,
        },
        "scope_receipt": {
            "eeg_input_and_acquisition_context_only": True,
            "target_or_reference_interval_embedded": False,
            "private_label_or_clinical_text_embedded": False,
            "pickle_or_executable_payload_required": False,
        },
    }


def ba_ieg_segmental_event_target_payload_v1(
    target: BAIEGSegmentalEventTargetV1,
) -> dict[str, Any]:
    """Serialize one immutable interval/censor target without model inputs."""

    if not isinstance(target, BAIEGSegmentalEventTargetV1):
        raise TypeError("segmental disk target export requires an event target")
    target.verify_integrity()
    if target.model_split not in _ALLOWED_SPLITS:
        raise ValueError("disk target export accepts source_train/source_dev only")
    return {
        "schema_version": BA_IEG_SEGMENTAL_DISK_TARGET_SCHEMA_VERSION,
        "target_receipt_sha256": target.receipt_sha256,
        "target": {
            "event_id": target.event_id,
            "recording_id": target.recording_id,
            "patient_uid": target.patient_uid,
            "model_split": target.model_split,
            "source_event_receipt_sha256": target.source_event_receipt_sha256,
            "adaptive_acquisition_receipt_sha256": (
                target.adaptive_acquisition_receipt_sha256
            ),
            "target_independent_candidate_roster_receipt_sha256": (
                target.target_independent_candidate_roster_receipt_sha256
            ),
            "source_reference_receipt_sha256": (target.source_reference_receipt_sha256),
            "authority": target.authority,
            "event_status": target.event_status,
            "onset_status": target.onset_status,
            "offset_status": target.offset_status,
            "bout_count_status": target.bout_count_status,
            "onset_interval_seconds": (
                list(target.onset_interval_seconds)
                if target.onset_interval_seconds is not None
                else None
            ),
            "offset_interval_seconds": (
                list(target.offset_interval_seconds)
                if target.offset_interval_seconds is not None
                else None
            ),
            "firewall": target.firewall.to_dict(),
        },
        "scope_receipt": {
            "public_source_boundary_target_only": True,
            "available_to_model_forward": False,
            "private_source": False,
        },
    }


def _read_strict_json(path: Path, *, maximum_bytes: int, context: str) -> object:
    size = path.stat().st_size
    if size < 1 or size > maximum_bytes:
        raise ValueError(f"{context} file size is outside the allowed bound")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error


def _load_event_input(
    metadata_path: Path,
    tensor_path: Path,
) -> tuple[BAIEGEventTokens, str, tuple[tuple[float, float], ...], str, str,]:
    payload = _read_strict_json(
        metadata_path, maximum_bytes=4 * 1024 * 1024, context="event input metadata"
    )
    required = {
        "schema_version",
        "input_event_receipt_sha256",
        "event_metadata",
        "boundary_context_input",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("event input metadata has missing or unknown fields")
    if payload["schema_version"] != BA_IEG_SEGMENTAL_DISK_INPUT_METADATA_SCHEMA_VERSION:
        raise ValueError("event input metadata schema drifted")
    if payload["scope_receipt"] != {
        "eeg_input_and_acquisition_context_only": True,
        "target_or_reference_interval_embedded": False,
        "private_label_or_clinical_text_embedded": False,
        "pickle_or_executable_payload_required": False,
    }:
        raise ValueError("event input metadata scope drifted")
    metadata = payload["event_metadata"]
    if type(metadata) is not dict or set(metadata) != _EVENT_METADATA_FIELDS:
        raise ValueError("event metadata fields drifted")
    context = payload["boundary_context_input"]
    context_fields = {
        "adaptive_acquisition_receipt_sha256",
        "quality_gap_intervals_seconds",
        "left_censor_reason_code",
        "right_censor_reason_code",
    }
    if type(context) is not dict or set(context) != context_fields:
        raise ValueError("event boundary-context metadata fields drifted")
    acquisition = _sha256(
        context["adaptive_acquisition_receipt_sha256"],
        "adaptive acquisition receipt",
    )
    left = context["left_censor_reason_code"]
    right = context["right_censor_reason_code"]
    if left not in BA_IEG_SEGMENTAL_CENSOR_REASONS or right not in (
        BA_IEG_SEGMENTAL_CENSOR_REASONS
    ):
        raise ValueError("event boundary-context censor reason drifted")
    gaps_raw = context["quality_gap_intervals_seconds"]
    if not isinstance(gaps_raw, list):
        raise TypeError("quality gaps must be an array")
    gaps: list[tuple[float, float]] = []
    previous_stop: float | None = None
    for index, value in enumerate(gaps_raw):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"quality gap {index} is invalid")
        start, stop = float(value[0]), float(value[1])
        if (
            not math.isfinite(start)
            or not math.isfinite(stop)
            or stop <= start
            or (previous_stop is not None and start < previous_stop)
        ):
            raise ValueError("quality gaps are not canonical")
        gaps.append((start, stop))
        previous_stop = stop
    tensor_file_size = tensor_path.stat().st_size
    if tensor_file_size < 1 or tensor_file_size > _MAXIMUM_EVENT_TENSOR_BYTES:
        raise ValueError("event tensor NPZ size is outside the allowed bound")
    try:
        with zipfile.ZipFile(tensor_path, "r") as container:
            members = container.infolist()
            expected_members = {f"{name}.npy" for name in _TENSOR_FIELDS}
            if (
                {member.filename for member in members} != expected_members
                or any(member.is_dir() for member in members)
                or sum(member.file_size for member in members)
                > _MAXIMUM_EVENT_TENSOR_BYTES
            ):
                raise ValueError("event tensor NPZ archive inventory is invalid")
        with np.load(tensor_path, allow_pickle=False) as archive:
            if set(archive.files) != set(_TENSOR_FIELDS):
                raise ValueError("event tensor NPZ has missing or unknown arrays")
            arrays = {name: np.asarray(archive[name]).copy() for name in _TENSOR_FIELDS}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError) and str(error).startswith("event tensor NPZ"):
            raise
        raise ValueError("event tensor artifact is not a safe NPZ") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("event tensor NPZ cannot contain object arrays")
    tensor = {name: torch.from_numpy(array) for name, array in arrays.items()}
    event = BAIEGEventTokens(
        event_id=metadata["event_id"],
        recording_id=metadata["recording_id"],
        patient_uid=metadata["patient_uid"],
        model_split=metadata["model_split"],
        analysis_interval_seconds=tuple(metadata["analysis_interval_seconds"]),
        navigation_anchor_seconds=metadata["navigation_anchor_seconds"],
        canonical_receipt_sha256=metadata["canonical_receipt_sha256"],
        adaptive_window_receipt_sha256=metadata["adaptive_window_receipt_sha256"],
        encoder_implementation_id=metadata["encoder_implementation_id"],
        encoder_lineage=metadata["encoder_lineage"],
        encoder_receipt_sha256=metadata["encoder_receipt_sha256"],
        physical_electrode_ids=tuple(metadata["physical_electrode_ids"]),
        physical_xyz=tensor["physical_xyz"],
        physical_xyz_mask=tensor["physical_xyz_mask"],
        physical_evidence_mask=tensor["physical_evidence_mask"],
        view_ids=tuple(metadata["view_ids"]),
        view_roles=tuple(metadata["view_roles"]),
        view_effective_temporal_roles=tuple(metadata["view_effective_temporal_roles"]),
        view_dependency_policies=tuple(metadata["view_dependency_policies"]),
        view_future_sample_access=tensor["view_future_sample_access"],
        view_onset_evidence_authorized=tensor["view_onset_evidence_authorized"],
        view_temporal_evidence_sha256s=tuple(
            metadata["view_temporal_evidence_sha256s"]
        ),
        view_receipt_sha256s=tuple(metadata["view_receipt_sha256s"]),
        view_transform_sha256s=tuple(metadata["view_transform_sha256s"]),
        reference_families=tuple(metadata["reference_families"]),
        unit_ids=tuple(metadata["unit_ids"]),
        unit_source_ids=tuple(metadata["unit_source_ids"]),
        unit_types=tuple(metadata["unit_types"]),
        unit_view_index=tensor["unit_view_index"],
        unit_reference_matrix=tensor["unit_reference_matrix"],
        unit_evidence_mask=tensor["unit_evidence_mask"],
        unit_family_mask=tensor["unit_family_mask"],
        token_values=tensor["token_values"],
        token_feature_mask=tensor["token_feature_mask"],
        token_time_bounds_seconds=tensor["token_time_bounds_seconds"],
        token_unit_index=tensor["token_unit_index"],
        token_view_index=tensor["token_view_index"],
        token_scale_index=tensor["token_scale_index"],
        token_signal_mask=tensor["token_signal_mask"],
        token_family_mask=tensor["token_family_mask"],
        phase_posterior=tensor["phase_posterior"],
    )
    if event.input_receipt_sha256 != _sha256(
        payload["input_event_receipt_sha256"], "input event receipt"
    ):
        raise ValueError("disk event tensors/metadata do not replay the input receipt")
    return event, acquisition, tuple(gaps), left, right


def _load_event_target(path: Path) -> BAIEGSegmentalEventTargetV1:
    payload = _read_strict_json(
        path, maximum_bytes=2 * 1024 * 1024, context="event target"
    )
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "target_receipt_sha256",
        "target",
        "scope_receipt",
    }:
        raise ValueError("event target artifact has missing or unknown fields")
    if payload["schema_version"] != BA_IEG_SEGMENTAL_DISK_TARGET_SCHEMA_VERSION:
        raise ValueError("event target disk schema drifted")
    if payload["scope_receipt"] != {
        "public_source_boundary_target_only": True,
        "available_to_model_forward": False,
        "private_source": False,
    }:
        raise ValueError("event target disk scope drifted")
    target = payload["target"]
    required_target = {
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "source_event_receipt_sha256",
        "adaptive_acquisition_receipt_sha256",
        "target_independent_candidate_roster_receipt_sha256",
        "source_reference_receipt_sha256",
        "authority",
        "event_status",
        "onset_status",
        "offset_status",
        "bout_count_status",
        "onset_interval_seconds",
        "offset_interval_seconds",
        "firewall",
    }
    if type(target) is not dict or set(target) != required_target:
        raise ValueError("event target fields drifted")
    firewall = target["firewall"]
    expected_firewall = BAIEGSegmentalTargetFirewallV1().to_dict()
    if firewall != expected_firewall:
        raise ValueError("event target firewall drifted")
    result = BAIEGSegmentalEventTargetV1(
        event_id=target["event_id"],
        recording_id=target["recording_id"],
        patient_uid=target["patient_uid"],
        model_split=target["model_split"],
        source_event_receipt_sha256=target["source_event_receipt_sha256"],
        adaptive_acquisition_receipt_sha256=target[
            "adaptive_acquisition_receipt_sha256"
        ],
        target_independent_candidate_roster_receipt_sha256=target[
            "target_independent_candidate_roster_receipt_sha256"
        ],
        source_reference_receipt_sha256=target["source_reference_receipt_sha256"],
        authority=target["authority"],
        event_status=target["event_status"],
        onset_status=target["onset_status"],
        offset_status=target["offset_status"],
        bout_count_status=target["bout_count_status"],
        onset_interval_seconds=(
            tuple(target["onset_interval_seconds"])
            if target["onset_interval_seconds"] is not None
            else None
        ),
        offset_interval_seconds=(
            tuple(target["offset_interval_seconds"])
            if target["offset_interval_seconds"] is not None
            else None
        ),
        firewall=BAIEGSegmentalTargetFirewallV1(),
    )
    if result.receipt_sha256 != _sha256(
        payload["target_receipt_sha256"], "target receipt"
    ):
        raise ValueError("disk target does not replay its immutable receipt")
    return result


@dataclass(frozen=True)
class BAIEGSegmentalDiskSampleV1:
    event: BAIEGEventTokens
    target: BAIEGSegmentalEventTargetV1
    adaptive_acquisition_receipt_sha256: str
    quality_gap_intervals_seconds: tuple[tuple[float, float], ...]
    left_censor_reason_code: str
    right_censor_reason_code: str
    manifest_row_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.event, BAIEGEventTokens) or not isinstance(
            self.target, BAIEGSegmentalEventTargetV1
        ):
            raise TypeError("disk sample requires registered event and target objects")
        self.event.verify_integrity()
        self.target.verify_integrity()
        _sha256(
            self.adaptive_acquisition_receipt_sha256,
            "adaptive acquisition receipt",
        )
        _sha256(self.manifest_row_receipt_sha256, "manifest row receipt")
        if (
            self.target.event_id != self.event.event_id
            or self.target.recording_id != self.event.recording_id
            or self.target.patient_uid != self.event.patient_uid
            or self.target.model_split != self.event.model_split
            or self.target.source_event_receipt_sha256
            != self.event.input_receipt_sha256
            or self.target.adaptive_acquisition_receipt_sha256
            != self.adaptive_acquisition_receipt_sha256
        ):
            raise ValueError(
                "disk sample input/target identity or receipt binding drifted"
            )
        if (
            self.target.authority
            not in _PUBLIC_DISK_TARGET_AUTHORITIES[self.event.model_split]
        ):
            raise ValueError(
                "disk sample target authority is not authorized for public TUSZ"
            )
        if self.left_censor_reason_code not in BA_IEG_SEGMENTAL_CENSOR_REASONS or (
            self.right_censor_reason_code not in BA_IEG_SEGMENTAL_CENSOR_REASONS
        ):
            raise ValueError("disk sample has unsupported censor reason")


@dataclass(frozen=True)
class BAIEGSegmentalDiskPatientBagV1:
    patient_uid: str
    model_split: str
    purpose: str
    samples: tuple[BAIEGSegmentalDiskSampleV1, ...]
    manifest_id: str
    manifest_file_sha256: str
    candidate_roster_receipt_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.patient_uid, "patient_uid")
        _identifier(self.manifest_id, "manifest_id")
        _sha256(self.manifest_file_sha256, "manifest_file_sha256")
        _sha256(
            self.candidate_roster_receipt_sha256,
            "candidate_roster_receipt_sha256",
        )
        if (
            self.purpose not in _PURPOSE_SPLIT
            or self.model_split != _PURPOSE_SPLIT[self.purpose]
        ):
            raise ValueError("disk patient bag purpose/split drifted")
        if not self.samples:
            raise ValueError("disk patient bag requires at least one event")
        if any(
            sample.event.patient_uid != self.patient_uid
            or sample.event.model_split != self.model_split
            or sample.target.target_independent_candidate_roster_receipt_sha256
            != self.candidate_roster_receipt_sha256
            for sample in self.samples
        ):
            raise ValueError("disk patient bag crosses identity, split or roster")


class BAIEGSegmentalDiskDatasetV1(Sequence[BAIEGSegmentalDiskPatientBagV1]):
    """One index per patient; all that patient's ragged events load together."""

    def __init__(self, manifest_path: str | Path, *, purpose: str) -> None:
        if purpose not in _PURPOSE_SPLIT:
            raise ValueError("disk dataset purpose must be optimize or calibrate")
        path = Path(manifest_path).resolve(strict=True)
        if not path.is_file() or path.suffix != ".json":
            raise ValueError("segmental disk manifest must be an existing JSON file")
        payload = _read_strict_json(
            path,
            maximum_bytes=128 * 1024 * 1024,
            context="segmental disk manifest",
        )
        manifest = validate_ba_ieg_segmental_disk_manifest_v1(payload)
        split = _PURPOSE_SPLIT[purpose]
        rows = [row for row in manifest["rows"] if row["model_split"] == split]
        if not rows:
            raise ValueError(
                "segmental disk manifest has no rows for requested purpose"
            )
        root = path.parent
        # Verify every selected artifact before exposing any target-bearing sample.
        resolved: dict[str, tuple[Path, Path, Path]] = {}
        for row in rows:
            metadata_path = _resolve_artifact(root, row["input_metadata_relative_path"])
            tensor_path = _resolve_artifact(root, row["input_tensors_relative_path"])
            target_path = _resolve_artifact(root, row["target_relative_path"])
            for artifact, expected, context in (
                (
                    metadata_path,
                    row["input_metadata_file_sha256"],
                    "input metadata",
                ),
                (
                    tensor_path,
                    row["input_tensors_file_sha256"],
                    "input tensors",
                ),
                (target_path, row["target_file_sha256"], "target"),
            ):
                if _file_sha256(artifact) != expected:
                    raise ValueError(f"segmental disk {context} file hash mismatch")
            resolved[row["event_id"]] = (
                metadata_path,
                tensor_path,
                target_path,
            )
        by_patient: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_patient.setdefault(row["patient_uid"], []).append(row)
        self._manifest_path = path
        self._manifest = manifest
        self._manifest_file_sha256 = _file_sha256(path)
        self._root = root
        self._purpose = purpose
        self._model_split = split
        self._patient_uids = tuple(sorted(by_patient))
        self._rows_by_patient = {
            patient: tuple(by_patient[patient]) for patient in self._patient_uids
        }
        self._resolved = resolved
        self._candidate_roster_receipt_sha256 = manifest[
            "target_independent_candidate_roster_receipt_sha256"
        ]

    @property
    def purpose(self) -> str:
        return self._purpose

    @property
    def model_split(self) -> str:
        return self._model_split

    @property
    def patient_uids(self) -> tuple[str, ...]:
        return self._patient_uids

    @property
    def manifest_id(self) -> str:
        return self._manifest["manifest_id"]

    @property
    def manifest_file_sha256(self) -> str:
        return self._manifest_file_sha256

    @property
    def candidate_roster_receipt_sha256(self) -> str:
        return self._candidate_roster_receipt_sha256

    @property
    def patient_event_token_counts(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(row["token_count"]) for row in self._rows_by_patient[patient])
            for patient in self._patient_uids
        )

    def __len__(self) -> int:
        return len(self._patient_uids)

    def __getitem__(self, index: int) -> BAIEGSegmentalDiskPatientBagV1:
        patient_uid = self._patient_uids[index]
        samples: list[BAIEGSegmentalDiskSampleV1] = []
        for row in self._rows_by_patient[patient_uid]:
            metadata_path, tensor_path, target_path = self._resolved[row["event_id"]]
            event, acquisition, gaps, left, right = _load_event_input(
                metadata_path, tensor_path
            )
            target = _load_event_target(target_path)
            if (
                event.event_id != row["event_id"]
                or event.recording_id != row["recording_id"]
                or event.patient_uid != row["patient_uid"]
                or event.model_split != row["model_split"]
                or event.input_receipt_sha256 != row["input_event_receipt_sha256"]
                or int(event.token_values.shape[0]) != row["token_count"]
                or target.receipt_sha256 != row["target_receipt_sha256"]
                or acquisition != row["adaptive_acquisition_receipt_sha256"]
                or target.source_reference_receipt_sha256
                != row["source_reference_receipt_sha256"]
                or target.target_independent_candidate_roster_receipt_sha256
                != row["target_independent_candidate_roster_receipt_sha256"]
            ):
                raise ValueError("segmental disk manifest row disagrees with artifacts")
            samples.append(
                BAIEGSegmentalDiskSampleV1(
                    event=event,
                    target=target,
                    adaptive_acquisition_receipt_sha256=acquisition,
                    quality_gap_intervals_seconds=gaps,
                    left_censor_reason_code=left,
                    right_censor_reason_code=right,
                    manifest_row_receipt_sha256=row["row_receipt_sha256"],
                )
            )
        return BAIEGSegmentalDiskPatientBagV1(
            patient_uid=patient_uid,
            model_split=self._model_split,
            purpose=self._purpose,
            samples=tuple(samples),
            manifest_id=self.manifest_id,
            manifest_file_sha256=self.manifest_file_sha256,
            candidate_roster_receipt_sha256=(self.candidate_roster_receipt_sha256),
        )


@dataclass(frozen=True)
class BAIEGSegmentalDiskBatchV1:
    event_batch: BAIEGCollatedEventBatch
    targets: tuple[BAIEGSegmentalEventTargetV1, ...]
    adaptive_acquisition_receipt_sha256s: tuple[str, ...]
    quality_gap_intervals_by_event: tuple[tuple[tuple[float, float], ...], ...]
    left_censor_reason_codes: tuple[str, ...]
    right_censor_reason_codes: tuple[str, ...]
    patient_uids: tuple[str, ...]
    expected_event_counts: tuple[int, ...]
    manifest_id: str
    manifest_file_sha256: str
    manifest_row_receipt_sha256s: tuple[str, ...]
    target_independent_candidate_roster_receipt_sha256: str
    optimization_role: str

    def __post_init__(self) -> None:
        if (
            self.optimization_role not in _PURPOSE_SPLIT
            or self.event_batch.model_split != (_PURPOSE_SPLIT[self.optimization_role])
        ):
            raise ValueError("segmental disk batch role/split drifted")
        events = len(self.event_batch.event_ids)
        aligned = (
            len(self.targets),
            len(self.adaptive_acquisition_receipt_sha256s),
            len(self.quality_gap_intervals_by_event),
            len(self.left_censor_reason_codes),
            len(self.right_censor_reason_codes),
            len(self.manifest_row_receipt_sha256s),
        )
        if any(length != events for length in aligned):
            raise ValueError("segmental disk batch event rows do not align")
        if sum(self.expected_event_counts) != events or len(
            self.expected_event_counts
        ) != len(self.patient_uids):
            raise ValueError("segmental disk batch patient event counts drifted")
        if len(set(self.patient_uids)) != len(self.patient_uids):
            raise ValueError("one optimizer batch cannot duplicate a patient")
        if (
            tuple(target.event_id for target in self.targets)
            != self.event_batch.event_ids
        ):
            raise ValueError("segmental disk target order drifted")
        _sha256(self.manifest_file_sha256, "manifest_file_sha256")
        _sha256(
            self.target_independent_candidate_roster_receipt_sha256,
            "target-independent candidate roster receipt",
        )

    def build_context(self) -> BAIEGSegmentalBoundaryContext:
        return build_ba_ieg_segmental_boundary_context(
            self.event_batch,
            adaptive_acquisition_receipt_sha256s=(
                self.adaptive_acquisition_receipt_sha256s
            ),
            quality_gap_intervals_by_event=self.quality_gap_intervals_by_event,
            left_censor_reason_codes=self.left_censor_reason_codes,
            right_censor_reason_codes=self.right_censor_reason_codes,
        )

    def build_target_bundle(
        self, context: BAIEGSegmentalBoundaryContext
    ) -> BAIEGSegmentalTargetBundleV1:
        return build_ba_ieg_segmental_target_bundle_v1(
            self.event_batch,
            context,
            self.targets,
            optimization_role=self.optimization_role,
            target_independent_candidate_roster_receipt_sha256=(
                self.target_independent_candidate_roster_receipt_sha256
            ),
        )


def collate_ba_ieg_segmental_disk_patient_bags_v1(
    bags: Sequence[BAIEGSegmentalDiskPatientBagV1],
) -> BAIEGSegmentalDiskBatchV1:
    if not bags or not all(
        isinstance(bag, BAIEGSegmentalDiskPatientBagV1) for bag in bags
    ):
        raise TypeError("segmental disk collation requires patient bags")
    if len({bag.patient_uid for bag in bags}) != len(bags):
        raise ValueError("segmental disk collation cannot duplicate a patient")
    common = {
        (
            bag.model_split,
            bag.purpose,
            bag.manifest_id,
            bag.manifest_file_sha256,
            bag.candidate_roster_receipt_sha256,
        )
        for bag in bags
    }
    if len(common) != 1:
        raise ValueError("segmental disk batch cannot mix split/manifest/roster")
    model_split, purpose, manifest_id, manifest_file_sha256, roster = next(iter(common))
    samples = tuple(sample for bag in bags for sample in bag.samples)
    event_batch = collate_ba_ieg_events(tuple(sample.event for sample in samples))
    if event_batch.model_split != model_split:
        raise ValueError("segmental disk collated input split drifted")
    return BAIEGSegmentalDiskBatchV1(
        event_batch=event_batch,
        targets=tuple(sample.target for sample in samples),
        adaptive_acquisition_receipt_sha256s=tuple(
            sample.adaptive_acquisition_receipt_sha256 for sample in samples
        ),
        quality_gap_intervals_by_event=tuple(
            sample.quality_gap_intervals_seconds for sample in samples
        ),
        left_censor_reason_codes=tuple(
            sample.left_censor_reason_code for sample in samples
        ),
        right_censor_reason_codes=tuple(
            sample.right_censor_reason_code for sample in samples
        ),
        patient_uids=tuple(bag.patient_uid for bag in bags),
        expected_event_counts=tuple(len(bag.samples) for bag in bags),
        manifest_id=manifest_id,
        manifest_file_sha256=manifest_file_sha256,
        manifest_row_receipt_sha256s=tuple(
            sample.manifest_row_receipt_sha256 for sample in samples
        ),
        target_independent_candidate_roster_receipt_sha256=roster,
        optimization_role=purpose,
    )


class BAIEGPatientTokenBucketBatchSamplerV1(Sampler[list[int]]):
    """Patient-whole, input-token-only packing for ragged event bags."""

    def __init__(
        self,
        dataset: BAIEGSegmentalDiskDatasetV1,
        *,
        maximum_padded_tokens_per_batch: int,
        maximum_patients_per_batch: int,
        shuffle: bool,
        seed: int = 20260823,
        bucket_size_multiplier: int = 16,
    ) -> None:
        if not isinstance(dataset, BAIEGSegmentalDiskDatasetV1):
            raise TypeError("token bucket sampler requires segmental disk dataset")
        self._dataset = dataset
        self._maximum_tokens = _positive_integer(
            maximum_padded_tokens_per_batch,
            "maximum_padded_tokens_per_batch",
        )
        self._maximum_patients = _positive_integer(
            maximum_patients_per_batch, "maximum_patients_per_batch"
        )
        if type(shuffle) is not bool:
            raise TypeError("shuffle must be boolean")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self._bucket_multiplier = _positive_integer(
            bucket_size_multiplier, "bucket_size_multiplier"
        )
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._shapes = dataset.patient_event_token_counts

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TypeError("epoch must be a non-negative integer")
        self._epoch = epoch

    def _batch_cost(self, indices: Sequence[int]) -> int:
        counts = [count for index in indices for count in self._shapes[index]]
        return len(counts) * max(counts)

    def _ordered_indices(self) -> list[int]:
        ordered = sorted(
            range(len(self._dataset)),
            key=lambda index: (
                max(self._shapes[index]),
                sum(self._shapes[index]),
                self._dataset.patient_uids[index],
            ),
        )
        if not self._shuffle:
            return ordered
        random_state = random.Random(self._seed + self._epoch)
        width = max(1, self._maximum_patients * self._bucket_multiplier)
        buckets = [
            ordered[start : start + width] for start in range(0, len(ordered), width)
        ]
        random_state.shuffle(buckets)
        for bucket in buckets:
            random_state.shuffle(bucket)
        return [index for bucket in buckets for index in bucket]

    def _materialize_batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        for index in self._ordered_indices():
            candidate = [*current, index]
            if current and (
                len(candidate) > self._maximum_patients
                or self._batch_cost(candidate) > self._maximum_tokens
            ):
                batches.append(current)
                current = [index]
            else:
                current = candidate
            # Oversized patients remain visible as a singleton; they are never
            # silently dropped or split across optimizer steps.
            if len(current) == 1 and self._batch_cost(current) > self._maximum_tokens:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._materialize_batches()

    def __len__(self) -> int:
        return len(self._materialize_batches())


def build_ba_ieg_segmental_disk_loader_v1(
    dataset: BAIEGSegmentalDiskDatasetV1,
    *,
    maximum_padded_tokens_per_batch: int,
    maximum_patients_per_batch: int,
    shuffle: bool,
    seed: int = 20260823,
    num_workers: int = 0,
) -> DataLoader[BAIEGSegmentalDiskBatchV1]:
    """Build a patient-whole ragged loader with target-independent packing."""

    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise TypeError("num_workers must be a non-negative integer")
    sampler = BAIEGPatientTokenBucketBatchSamplerV1(
        dataset,
        maximum_padded_tokens_per_batch=maximum_padded_tokens_per_batch,
        maximum_patients_per_batch=maximum_patients_per_batch,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_ba_ieg_segmental_disk_patient_bags_v1,
        num_workers=num_workers,
    )


@dataclass(frozen=True)
class BAIEGSegmentalCalibrationForwardV1:
    context: BAIEGSegmentalBoundaryContext
    output: BAIEGPermissionSplitSegmentalStateOutput
    target_bundle: BAIEGSegmentalTargetBundleV1
    manifest_id: str
    manifest_file_sha256: str


class BAIEGPermissionSplitSegmentalTrainerV1:
    """Minimal optimizer/calibration harness preserving the split firewall."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        *,
        maximum_gradient_norm: float | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("segmental trainer model must be torch.nn.Module")
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("segmental trainer optimizer is invalid")
        if maximum_gradient_norm is not None:
            value = float(maximum_gradient_norm)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("maximum_gradient_norm must be positive and finite")
            maximum_gradient_norm = value
        self.model = model
        self.optimizer = optimizer
        self.maximum_gradient_norm = maximum_gradient_norm

    def optimize_batch(
        self, batch: BAIEGSegmentalDiskBatchV1
    ) -> BAIEGPermissionSplitSegmentalLossOutputV1:
        if not isinstance(batch, BAIEGSegmentalDiskBatchV1):
            raise TypeError("segmental trainer requires a registered disk batch")
        if (
            batch.optimization_role != "optimize"
            or batch.event_batch.model_split != "source_train"
        ):
            raise ValueError(
                "gradient updates are source_train optimize-only; source_dev, "
                "source_eval and private labels are forbidden"
            )
        if self.optimizer is None:
            raise ValueError("segmental optimizer is not configured")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        context = batch.build_context()
        output = self.model(batch.event_batch, context)
        if not isinstance(output, BAIEGPermissionSplitSegmentalStateOutput):
            raise TypeError("segmental model returned an unsupported output")
        bundle = batch.build_target_bundle(context)
        loss = permission_split_segmental_training_loss_v1(
            batch.event_batch, context, output, bundle
        )
        loss.total_loss.backward()
        if self.maximum_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.maximum_gradient_norm
            )
        self.optimizer.step()
        return loss

    def calibration_forward(
        self, batch: BAIEGSegmentalDiskBatchV1
    ) -> BAIEGSegmentalCalibrationForwardV1:
        if not isinstance(batch, BAIEGSegmentalDiskBatchV1):
            raise TypeError("segmental trainer requires a registered disk batch")
        if (
            batch.optimization_role != "calibrate"
            or batch.event_batch.model_split != "source_dev"
        ):
            raise ValueError("calibration forward is source_dev-only")
        self.model.eval()
        with torch.no_grad():
            context = batch.build_context()
            output = self.model(batch.event_batch, context)
            if not isinstance(output, BAIEGPermissionSplitSegmentalStateOutput):
                raise TypeError("segmental model returned an unsupported output")
            output = BAIEGPermissionSplitSegmentalStateOutput(
                **{
                    item.name: (
                        getattr(output, item.name).detach()
                        if isinstance(getattr(output, item.name), torch.Tensor)
                        else getattr(output, item.name)
                    )
                    for item in fields(output)
                }
            )
            bundle = batch.build_target_bundle(context)
        if any(
            tensor.requires_grad
            for tensor in output.__dict__.values()
            if isinstance(tensor, torch.Tensor)
        ):
            raise RuntimeError("source_dev calibration forward retained gradients")
        return BAIEGSegmentalCalibrationForwardV1(
            context=context,
            output=output,
            target_bundle=bundle,
            manifest_id=batch.manifest_id,
            manifest_file_sha256=batch.manifest_file_sha256,
        )


__all__ = [
    "BA_IEG_SEGMENTAL_DISK_INPUT_METADATA_SCHEMA_VERSION",
    "BA_IEG_SEGMENTAL_DISK_MANIFEST_METHOD_ID",
    "BA_IEG_SEGMENTAL_DISK_MANIFEST_SCHEMA_VERSION",
    "BA_IEG_SEGMENTAL_DISK_TARGET_SCHEMA_VERSION",
    "BAIEGPatientTokenBucketBatchSamplerV1",
    "BAIEGPermissionSplitSegmentalTrainerV1",
    "BAIEGSegmentalCalibrationForwardV1",
    "BAIEGSegmentalDiskBatchV1",
    "BAIEGSegmentalDiskDatasetV1",
    "BAIEGSegmentalDiskPatientBagV1",
    "BAIEGSegmentalDiskSampleV1",
    "ba_ieg_segmental_event_input_metadata_v1",
    "ba_ieg_segmental_event_target_payload_v1",
    "ba_ieg_segmental_event_tensor_arrays_v1",
    "build_ba_ieg_segmental_disk_loader_v1",
    "build_ba_ieg_segmental_disk_manifest_v1",
    "collate_ba_ieg_segmental_disk_patient_bags_v1",
    "validate_ba_ieg_segmental_disk_manifest_v1",
]
