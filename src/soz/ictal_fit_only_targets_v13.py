"""Process-isolated fit-only TUSZ target artifacts for the v13 controls.

The legacy formal-v4 target snapshot stores all 1,519 training events in two
monolithic NumPy arrays.  Those arrays include the 12-patient I-gate.  A
normal ``np.load`` (including ``mmap_mode='r'``) therefore makes gate target
bytes reachable before a patient filter is applied.

This module implements a narrower boundary:

* strict-load only the pinned source manifest/receipt and NumPy headers;
* resolve an exact k31 fit allow-list before reading any array data;
* use ``os.pread`` for the C-order byte range of each permitted row only;
* verify every selected target/mask row against its frozen TUSZ event hash;
* atomically publish a physical artifact that contains fit rows only.

The control trainer consumes only that fit-only artifact.  It never receives
the legacy monolithic arrays as an argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import torch

from .ictal_fit_primitives_v13 import (
    IctalTrainingConfig,
    LABRAM_K31_OOF_RUN_SCHEMA_V1_2,
    LABRAM_K31_TARGET_SEMANTICS,
    VerifiedFitOnlyIctalTargetSnapshotV13,
    canonical_json_bytes as _canonical_json_bytes,
    file_sha256 as _file_sha256,
    patient_roster as _patient_roster,
    patient_roster_sha256,
    require_sha256 as _require_sha256,
    safe_new_output as _safe_new_output,
    selection as _selection,
)

if TYPE_CHECKING:
    from .data.tusz_training import TUSZIctalTrainingManifest


ICTAL_TARGET_SNAPSHOT_SEMANTICS = LABRAM_K31_TARGET_SEMANTICS


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
SOURCE_TARGET_FILES = frozenset(
    {
        "manifest.json",
        "receipt.json",
        "full_native_logits.npy",
        "native_targets.npy",
        "native_target_mask.npy",
        "training_targets.npy",
        "training_target_mask.npy",
    }
)
SOURCE_TENSOR_NAMES = frozenset(
    {
        "full_native_logits",
        "native_targets",
        "native_target_mask",
        "training_targets",
        "training_target_mask",
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
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _laplace_logit(*, positive_count: int, observed_count: int) -> float:
    """Return the frozen Beta(1,1) posterior-mean logit."""

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
    targets: torch.Tensor,
    target_mask: torch.Tensor,
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
        or mask.dtype is not torch.bool
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
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    """Compute the three frozen deterministic controls from fit rows only.

    Every count explicitly intersects the observation mask.  Values outside
    that mask are never interpreted as negatives (or as positives).
    """

    values, mask = _shortcut_tensor_inputs(targets, target_mask)
    positives = mask & (values == 1)
    event_count, edge_count, second_count = (int(item) for item in values.shape)

    time_rows: list[dict[str, object]] = []
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
        positive_count=global_positive,
        observed_count=global_observed,
    )
    prevalence_only = _hashed_parameter_section(
        {
            "algorithm_id": PREVALENCE_ONLY_ALGORITHM_V13,
            "observed_cell_count": global_observed,
            "positive_cell_count": global_positive,
            "logit": global_logit,
        }
    )

    flattened_mask = mask.reshape(event_count, edge_count * second_count)
    densities = flattened_mask.to(torch.float64).mean(dim=1)
    boundaries = torch.tensor(MASK_DENSITY_BOUNDARIES_V13, dtype=torch.float64)
    assignments = torch.bucketize(densities, boundaries, right=False)
    bin_rows: list[dict[str, object]] = []
    for bin_index in range(len(MASK_DENSITY_BOUNDARIES_V13) + 1):
        selected = assignments == bin_index
        selected_event_count = int(selected.sum().item())
        if selected_event_count:
            selected_mask = mask[selected]
            observed_count = int(selected_mask.sum().item())
            positive_count = int(positives[selected].sum().item())
            fallback = False
            logit = _laplace_logit(
                positive_count=positive_count,
                observed_count=observed_count,
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

    core: dict[str, object] = {
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
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_json_numbers(item, field=f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json_numbers(item, field=f"{field}[{index}]")


def validate_fit_only_shortcut_parameters_v13(
    value: object,
    *,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    """Strictly validate hashes and exactly recompute all fit-only controls."""

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
        declared_sha = _require_sha256(
            record.pop("parameters_sha256"),
            field=f"fit_only_shortcut_parameters.{name}.parameters_sha256",
        )
        if _canonical_sha256(record) != declared_sha:
            raise ValueError(f"fit-only shortcut section SHA mismatch: {name}")
    bundle_record = dict(payload)
    declared_bundle_sha = _require_sha256(
        bundle_record.pop("bundle_sha256"),
        field="fit_only_shortcut_parameters.bundle_sha256",
    )
    if _canonical_sha256(bundle_record) != declared_bundle_sha:
        raise ValueError("fit-only shortcut bundle SHA mismatch")
    if _canonical_json_bytes(payload) != _canonical_json_bytes(expected):
        raise ValueError("fit-only shortcut parameters failed exact fit-row recomputation")
    return expected


def _strict_json(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], str]:
    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError(f"Strict JSON must be a regular absolute file: {source.name}")
    before = source.stat()
    if before.st_size < 1 or before.st_size > _MAX_JSON_BYTES:
        raise ValueError(f"Strict JSON has an invalid size: {source.name}")
    raw = source.read_bytes()
    after = source.stat()
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id:
        raise RuntimeError(f"Strict JSON changed while read: {source.name}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _require_sha256(
        expected_sha256, field=f"expected_{source.name}_sha256"
    ):
        raise ValueError(f"Strict JSON SHA mismatch: {source.name}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Strict JSON is invalid: {source.name}") from exc
    canonical = _canonical_json_bytes(payload)
    if not isinstance(payload, dict) or raw not in {canonical, canonical + b"\n"}:
        raise ValueError(f"Strict JSON is not canonical: {source.name}")
    return payload, digest


def _event_rows(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("training_event_rows must be a non-empty array")
    rows: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("training_event_rows entries must be [event_id, patient_id]")
        event_id, patient_id = (str(part).strip() for part in item)
        if not event_id or not patient_id:
            raise ValueError("training_event_rows contains an empty identity")
        rows.append((event_id, patient_id))
    if len({event_id for event_id, _ in rows}) != len(rows):
        raise ValueError("training_event_rows contains duplicate event IDs")
    return tuple(rows)


def _source_tensor_record(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"source tensor record must be a mapping: {name}")
    record = dict(value)
    expected = {
        "dtype",
        "file_sha256",
        "file_size_bytes",
        "filename",
        "shape",
        "tensor_sha256",
    }
    if set(record) != expected:
        raise ValueError(f"source tensor record violates its closed schema: {name}")
    _require_sha256(record["file_sha256"], field=f"{name}.file_sha256")
    _require_sha256(record["tensor_sha256"], field=f"{name}.tensor_sha256")
    if (
        not isinstance(record["filename"], str)
        or not record["filename"].endswith(".npy")
        or "/" in record["filename"]
    ):
        raise ValueError(f"source tensor filename is invalid: {name}")
    if (
        isinstance(record["file_size_bytes"], bool)
        or not isinstance(record["file_size_bytes"], int)
        or record["file_size_bytes"] < 1
    ):
        raise ValueError(f"source tensor file size is invalid: {name}")
    if not isinstance(record["shape"], list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in record["shape"]
    ):
        raise ValueError(f"source tensor shape is invalid: {name}")
    if record["dtype"] not in {"torch.float32", "torch.bool"}:
        raise ValueError(f"source tensor dtype is invalid: {name}")
    return record


@dataclass(frozen=True)
class SourceNpyLayout:
    path: Path
    declared_file_sha256: str
    declared_tensor_sha256: str
    shape: tuple[int, ...]
    torch_dtype: str
    numpy_dtype: str
    data_offset_bytes: int
    row_size_bytes: int
    file_size_bytes: int


def _read_npy_header(
    path: Path,
    *,
    record: Mapping[str, object],
    expected_torch_dtype: str,
) -> SourceNpyLayout:
    """Read a NumPy header without decoding, mapping, or hashing array data."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError(f"Source NumPy file must be regular: {source.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Source NumPy file must be regular: {source.name}")
        # ``buffering=0`` is a firewall requirement.  A default
        # ``BufferedReader`` may prefetch array payload bytes while NumPy asks
        # only for the header.
        with os.fdopen(os.dup(descriptor), "rb", buffering=0) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    handle
                )
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    handle
                )
            else:
                raise ValueError(f"Unsupported NumPy format version: {version}")
            data_offset = handle.tell()
        expected_dtype = (
            np.dtype(np.float32)
            if expected_torch_dtype == "torch.float32"
            else np.dtype(np.bool_)
        )
        expected_shape = tuple(int(value) for value in record["shape"])
        if fortran_order or tuple(shape) != expected_shape or dtype != expected_dtype:
            raise ValueError(f"Source NumPy header changed: {source.name}")
        if record["dtype"] != expected_torch_dtype:
            raise ValueError(f"Source NumPy declared dtype changed: {source.name}")
        row_size = int(np.prod(expected_shape[1:], dtype=np.int64)) * dtype.itemsize
        expected_size = data_offset + expected_shape[0] * row_size
        if (
            metadata.st_size != expected_size
            or metadata.st_size != int(record["file_size_bytes"])
        ):
            raise ValueError(f"Source NumPy size/header mismatch: {source.name}")
        return SourceNpyLayout(
            path=source,
            declared_file_sha256=_require_sha256(
                record["file_sha256"], field=f"{source.name}.declared_file_sha256"
            ),
            declared_tensor_sha256=_require_sha256(
                record["tensor_sha256"], field=f"{source.name}.declared_tensor_sha256"
            ),
            shape=expected_shape,
            torch_dtype=expected_torch_dtype,
            numpy_dtype=dtype.str,
            data_offset_bytes=data_offset,
            row_size_bytes=row_size,
            file_size_bytes=metadata.st_size,
        )
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class SourceTargetIndexV13:
    path: Path
    manifest_sha256: str
    receipt_sha256: str
    training_manifest_sha256: str
    training_corpus_index_sha256: str
    native_manifest_sha256: str
    native_corpus_index_sha256: str
    event_rows: tuple[tuple[str, str], ...]
    target_layout: SourceNpyLayout
    mask_layout: SourceNpyLayout
    source_full_tensor_hashes_computed: bool = False
    source_full_arrays_loaded: bool = False
    source_full_arrays_mapped: bool = False


def load_source_target_index_v13(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> SourceTargetIndexV13:
    """Strict-load target identity and headers, never full tensor bytes."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("Source target snapshot must be a regular absolute directory")
    if {entry.name for entry in source.iterdir()} != SOURCE_TARGET_FILES:
        raise ValueError("Source target snapshot has missing or unknown files")
    manifest, manifest_sha = _strict_json(
        source / "manifest.json", expected_sha256=expected_manifest_sha256
    )
    receipt, receipt_sha = _strict_json(
        source / "receipt.json", expected_sha256=expected_receipt_sha256
    )
    if manifest.get("schema_version") != "soz_ictal_native_prediction_artifact_v1":
        raise ValueError("Source target snapshot schema changed")
    if receipt.get("schema_version") != "soz_ictal_native_prediction_bundle_receipt_v1":
        raise ValueError("Source target receipt schema changed")
    if receipt.get("artifact_sha256") != manifest_sha:
        raise ValueError("Source target receipt does not bind its manifest")
    fixed = {
        "selection": "final",
        "target_semantics": ICTAL_TARGET_SNAPSHOT_SEMANTICS,
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "missing_tusz_bins_imputed_as_negative": False,
    }
    if any(manifest.get(field) != value for field, value in fixed.items()):
        raise ValueError("Source target snapshot changed a frozen boundary")
    tensor_files = manifest.get("tensor_files")
    if not isinstance(tensor_files, Mapping) or set(tensor_files) != SOURCE_TENSOR_NAMES:
        raise ValueError("Source target tensor roster changed")
    records = {
        name: _source_tensor_record(tensor_files[name], name=name)
        for name in SOURCE_TENSOR_NAMES
    }
    if records["training_targets"]["filename"] != "training_targets.npy":
        raise ValueError("Source training target filename changed")
    if records["training_target_mask"]["filename"] != "training_target_mask.npy":
        raise ValueError("Source training mask filename changed")
    rows = _event_rows(manifest.get("training_event_rows"))
    if tuple(records["training_targets"]["shape"]) != (len(rows), 20, 60):
        raise ValueError("Source training target shape changed")
    if tuple(records["training_target_mask"]["shape"]) != (len(rows), 20, 60):
        raise ValueError("Source training mask shape changed")
    targets = _read_npy_header(
        source / "training_targets.npy",
        record=records["training_targets"],
        expected_torch_dtype="torch.float32",
    )
    mask = _read_npy_header(
        source / "training_target_mask.npy",
        record=records["training_target_mask"],
        expected_torch_dtype="torch.bool",
    )
    return SourceTargetIndexV13(
        path=source,
        manifest_sha256=manifest_sha,
        receipt_sha256=receipt_sha,
        training_manifest_sha256=_require_sha256(
            manifest.get("training_manifest_sha256"),
            field="source.training_manifest_sha256",
        ),
        training_corpus_index_sha256=_require_sha256(
            manifest.get("training_corpus_index_sha256"),
            field="source.training_corpus_index_sha256",
        ),
        native_manifest_sha256=_require_sha256(
            manifest.get("native_evaluation_manifest_sha256"),
            field="source.native_manifest_sha256",
        ),
        native_corpus_index_sha256=_require_sha256(
            manifest.get("native_evaluation_corpus_index_sha256"),
            field="source.native_corpus_index_sha256",
        ),
        event_rows=rows,
        target_layout=targets,
        mask_layout=mask,
    )


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


def selected_fit_source_rows(
    source: SourceTargetIndexV13,
    manifest: "TUSZIctalTrainingManifest",
    *,
    fit_patient_ids: Sequence[object],
    forbidden_i_gate_patient_ids: Sequence[object],
) -> tuple[tuple[int, object], ...]:
    """Resolve and validate the allow-list before any source data read."""

    from .data.tusz_training import TUSZIctalTrainingManifest

    if not isinstance(source, SourceTargetIndexV13):
        raise TypeError("source must be SourceTargetIndexV13")
    if not isinstance(manifest, TUSZIctalTrainingManifest):
        raise TypeError("manifest must be TUSZIctalTrainingManifest")
    fit = _patient_roster(fit_patient_ids, field="fit_patient_ids", allow_empty=False)
    gate = _patient_roster(
        forbidden_i_gate_patient_ids,
        field="forbidden_i_gate_patient_ids",
        allow_empty=False,
    )
    if set(fit) & set(gate):
        raise ValueError("Fit target allow-list overlaps I-gate")
    if set(manifest.patient_ids) != set(fit) | set(gate):
        raise ValueError("Training manifest is not exactly fit plus I-gate")
    source_index = {
        event_id: (index, patient_id)
        for index, (event_id, patient_id) in enumerate(source.event_rows)
    }
    selected: list[tuple[int, object]] = []
    for patient_id in fit:
        for event in manifest.events_for_patient(patient_id):
            source_row = source_index.get(event.event_id)
            if source_row is None or source_row[1] != patient_id:
                raise ValueError("Source target snapshot is missing a fit event")
            selected.append((source_row[0], event))
    selected.sort(key=lambda item: item[0])
    selected_indices = {index for index, _ in selected}
    gate_event_ids = {
        event.event_id
        for patient_id in gate
        for event in manifest.events_for_patient(patient_id)
    }
    gate_indices = {
        index
        for index, (event_id, patient_id) in enumerate(source.event_rows)
        if patient_id in set(gate) and event_id in gate_event_ids
    }
    if not gate_indices or selected_indices & gate_indices:
        raise ValueError("Fit/I-gate source-row firewall failed")
    if len(selected_indices) != len(selected):
        raise ValueError("Fit target selection contains duplicate source rows")
    return tuple(selected)


def _pread_rows(
    layout: SourceNpyLayout,
    *,
    row_indices: Sequence[int],
    forbidden_row_indices: Sequence[int],
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    indices = tuple(int(value) for value in row_indices)
    forbidden = {int(value) for value in forbidden_row_indices}
    if (
        not indices
        or indices != tuple(sorted(indices))
        or len(set(indices)) != len(indices)
        or any(index < 0 or index >= layout.shape[0] for index in indices)
        or set(indices) & forbidden
    ):
        raise ValueError("Selected source row allow-list is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(layout.path, flags)
    chunks: list[bytes] = []
    ranges: list[tuple[int, int, int]] = []
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != layout.file_size_bytes:
            raise ValueError("Source array changed after header validation")
        for index in indices:
            start = layout.data_offset_bytes + index * layout.row_size_bytes
            stop = start + layout.row_size_bytes
            if index in forbidden:
                raise RuntimeError("Attempted to read a forbidden I-gate source row")
            raw = os.pread(descriptor, layout.row_size_bytes, start)
            if len(raw) != layout.row_size_bytes:
                raise ValueError("Short source-array row read")
            chunks.append(raw)
            ranges.append((index, start, stop))
    finally:
        os.close(descriptor)
    dtype = np.dtype(layout.numpy_dtype)
    array = np.frombuffer(b"".join(chunks), dtype=dtype).copy()
    return array.reshape((len(indices), *layout.shape[1:])), tuple(ranges)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


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
            raise FileExistsError(f"fit-only target output already exists: {target}")
        raise OSError(error, os.strerror(error), str(target))


@dataclass(frozen=True)
class LoadedFitOnlyTargetArtifactV13:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    receipt_sha256: str
    events: tuple["FitOnlyTargetEventV13", ...]
    snapshot: VerifiedFitOnlyIctalTargetSnapshotV13
    shortcut_parameters: Mapping[str, object]


@dataclass(frozen=True)
class FitOnlyTargetEventV13:
    event_id: str
    patient_id: str
    source_row_index: int
    event_record_sha256: str
    preprocess_receipt_sha256: str
    target_sha256: str
    target_mask_sha256: str


def materialize_fit_only_target_artifact_v13(
    output_directory: str | Path,
    *,
    source: SourceTargetIndexV13,
    training_manifest: "TUSZIctalTrainingManifest",
    training_bundle_manifest_sha256: str,
    training_corpus_index_sha256: str,
    k31_manifest: Mapping[str, object],
    k31_manifest_sha256: str,
) -> LoadedFitOnlyTargetArtifactV13:
    """Extract and atomically seal exactly one k31 selection's fit rows."""

    if not isinstance(k31_manifest, Mapping):
        raise TypeError("k31_manifest must be a mapping")
    k31 = dict(k31_manifest)
    if k31.get("schema_version") != LABRAM_K31_OOF_RUN_SCHEMA_V1_2:
        raise ValueError("fit-only extraction requires k31 v1.2")
    selection, fold = _selection(k31.get("selection"))
    fit = _patient_roster(
        k31.get("training_public_patient_ids"),
        field="k31.training_public_patient_ids",
        allow_empty=False,
    )
    gate = _patient_roster(
        k31.get("i_gate_patient_ids_excluded_unopened"),
        field="k31.i_gate_patient_ids_excluded_unopened",
        allow_empty=False,
    )
    if len(gate) != 12 or set(fit) & set(gate):
        raise ValueError("k31 fit/I-gate roster is invalid")
    expected_lineage = {
        "training_manifest_sha256": training_manifest.manifest_sha256,
        "training_corpus_index_sha256": _require_sha256(
            training_corpus_index_sha256, field="training_corpus_index_sha256"
        ),
        "target_snapshot_manifest_sha256": source.manifest_sha256,
        "target_snapshot_receipt_sha256": source.receipt_sha256,
    }
    for field, value in expected_lineage.items():
        if k31.get(field) != value:
            raise ValueError(f"fit-only extraction differs from k31: {field}")
    # The legacy source snapshot is master-bound while each fold uses a
    # derived training manifest.  k31 binds the source snapshot by its exact
    # manifest/receipt hashes and the selected training manifest by its exact
    # source hash; its v1.2 closed schema did not persist the training bundle
    # manifest or a separate master-manifest field.
    _require_sha256(
        training_bundle_manifest_sha256,
        field="training_bundle_manifest_sha256",
    )
    selected = selected_fit_source_rows(
        source,
        training_manifest,
        fit_patient_ids=fit,
        forbidden_i_gate_patient_ids=gate,
    )
    selected_indices = tuple(index for index, _ in selected)
    gate_indices = tuple(
        index
        for index, (_, patient_id) in enumerate(source.event_rows)
        if patient_id in set(gate)
    )
    targets_array, target_ranges = _pread_rows(
        source.target_layout,
        row_indices=selected_indices,
        forbidden_row_indices=gate_indices,
    )
    mask_array, mask_ranges = _pread_rows(
        source.mask_layout,
        row_indices=selected_indices,
        forbidden_row_indices=gate_indices,
    )
    if tuple(item[0] for item in target_ranges) != selected_indices or tuple(
        item[0] for item in mask_ranges
    ) != selected_indices:
        raise RuntimeError("Source pread receipt changed selected row order")
    targets = torch.from_numpy(np.ascontiguousarray(targets_array)).to(torch.float32)
    mask = torch.from_numpy(np.ascontiguousarray(mask_array)).to(torch.bool)
    if tuple(targets.shape) != (len(selected), 20, 60) or mask.shape != targets.shape:
        raise ValueError("Fit-only extracted arrays have the wrong shape")
    shortcut_parameters = compute_fit_only_shortcut_parameters_v13(targets, mask)
    event_rows: list[list[object]] = []
    for row_index, ((source_index, event), target, target_mask) in enumerate(
        zip(selected, targets, mask, strict=True)
    ):
        target_sha = _event_tensor_sha256(target)
        mask_sha = _event_tensor_sha256(target_mask)
        if target_sha != event.target_sha256 or mask_sha != event.target_mask_sha256:
            raise ValueError(f"Selected source row failed event hash: {event.event_id}")
        if event.signal_preflight_receipt_sha256 is None:
            raise ValueError("Fit-only events require a frozen signal preflight receipt")
        event_rows.append(
            [
                event.event_id,
                event.patient_id,
                source_index,
                event.event_record_sha256,
                event.signal_preflight_receipt_sha256,
                target_sha,
                mask_sha,
            ]
        )
    target = _safe_new_output(output_directory)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.fit-only-", dir=target.parent))
    published = False
    try:
        target_file = staging / FIT_ONLY_TARGETS
        mask_file = staging / FIT_ONLY_TARGET_MASK
        with target_file.open("wb") as handle:
            np.save(handle, targets.numpy(), allow_pickle=False)
        with mask_file.open("wb") as handle:
            np.save(handle, mask.numpy(), allow_pickle=False)
        tensor_files = {
            "fit_targets": {
                "filename": FIT_ONLY_TARGETS,
                "shape": list(targets.shape),
                "dtype": str(targets.dtype),
                "file_size_bytes": target_file.stat().st_size,
                "file_sha256": _file_sha256(target_file),
                "tensor_sha256": _artifact_tensor_sha256("fit_targets", targets),
            },
            "fit_target_mask": {
                "filename": FIT_ONLY_TARGET_MASK,
                "shape": list(mask.shape),
                "dtype": str(mask.dtype),
                "file_size_bytes": mask_file.stat().st_size,
                "file_sha256": _file_sha256(mask_file),
                "tensor_sha256": _artifact_tensor_sha256("fit_target_mask", mask),
            },
        }
        payload = {
            "schema_version": FIT_ONLY_TARGET_SCHEMA_V13,
            "selection": selection,
            "oof_fold": fold,
            "target_semantics": ICTAL_TARGET_SNAPSHOT_SEMANTICS,
            "source_access_policy": "npy_header_then_explicit_fit_c_order_pread_rows_v1",
            "matched_k31_manifest_sha256": _require_sha256(
                k31_manifest_sha256, field="k31_manifest_sha256"
            ),
            "matched_k31_checkpoint_sha256": _require_sha256(
                k31.get("checkpoint_sha256"), field="k31.checkpoint_sha256"
            ),
            "matched_training_config": dict(k31["training_config"]),
            "training_manifest_bundle_sha256": _require_sha256(
                training_bundle_manifest_sha256,
                field="training_manifest_bundle_sha256",
            ),
            "training_manifest_sha256": training_manifest.manifest_sha256,
            "training_corpus_index_sha256": _require_sha256(
                training_corpus_index_sha256,
                field="training_corpus_index_sha256",
            ),
            "source_target_snapshot_manifest_sha256": source.manifest_sha256,
            "source_target_snapshot_receipt_sha256": source.receipt_sha256,
            "source_native_evaluation_manifest_sha256": source.native_manifest_sha256,
            "source_native_evaluation_corpus_index_sha256": source.native_corpus_index_sha256,
            "source_training_targets_declared_file_sha256": source.target_layout.declared_file_sha256,
            "source_training_target_mask_declared_file_sha256": source.mask_layout.declared_file_sha256,
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
            "fit_patient_ids": list(fit),
            "fit_patient_roster_sha256": patient_roster_sha256(fit),
            "i_gate_patient_ids_excluded_unopened": list(gate),
            "i_gate_patient_roster_sha256": patient_roster_sha256(gate),
            "fit_event_count": len(event_rows),
            "fit_event_rows": event_rows,
            "selected_source_row_indices_sha256": _canonical_sha256(selected_indices),
            "selected_target_byte_ranges_sha256": _canonical_sha256(target_ranges),
            "selected_mask_byte_ranges_sha256": _canonical_sha256(mask_ranges),
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
            "fit_only_shortcut_parameters": shortcut_parameters,
            "tensor_files": tensor_files,
        }
        raw = _canonical_json_bytes(payload)
        (staging / FIT_ONLY_TARGET_MANIFEST).write_bytes(raw)
        artifact_sha = hashlib.sha256(raw).hexdigest()
        receipt_payload = {
            "schema_version": FIT_ONLY_TARGET_RECEIPT_SCHEMA_V13,
            "artifact_sha256": artifact_sha,
            "artifact_size_bytes": len(raw),
        }
        receipt_raw = _canonical_json_bytes(receipt_payload)
        (staging / FIT_ONLY_TARGET_RECEIPT).write_bytes(receipt_raw)
        for file in staging.iterdir():
            _fsync_file(file)
        _fsync_directory(staging)
        _rename_noreplace(staging, target)
        _fsync_directory(target.parent)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return load_fit_only_target_artifact_v13(
        target,
        expected_manifest_sha256=_file_sha256(target / FIT_ONLY_TARGET_MANIFEST),
        expected_receipt_sha256=_file_sha256(target / FIT_ONLY_TARGET_RECEIPT),
    )


def _load_artifact_array(
    path: Path,
    *,
    record: object,
    name: str,
    filename: str,
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
    if len(raw) != payload["file_size_bytes"] or hashlib.sha256(raw).hexdigest() != _require_sha256(
        payload["file_sha256"], field=f"{name}.file_sha256"
    ):
        raise ValueError(f"fit-only tensor file receipt mismatch: {name}")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"fit-only tensor is not safe NumPy: {name}") from exc
    tensor = torch.from_numpy(np.ascontiguousarray(array)).contiguous()
    if list(tensor.shape) != payload["shape"] or str(tensor.dtype) != payload["dtype"]:
        raise ValueError(f"fit-only tensor shape/dtype changed: {name}")
    if _artifact_tensor_sha256(name, tensor) != _require_sha256(
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
        "target_semantics": ICTAL_TARGET_SNAPSHOT_SEMANTICS,
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
    selection, fold = _selection(manifest.get("selection"))
    if manifest.get("oof_fold") != fold:
        raise ValueError("fit-only selection/fold mismatch")
    fit = _patient_roster(
        manifest.get("fit_patient_ids"), field="fit_patient_ids", allow_empty=False
    )
    gate = _patient_roster(
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
    for field in (
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
    ):
        _require_sha256(manifest.get(field), field=field)
    if manifest.get("matched_training_config") != asdict(IctalTrainingConfig()):
        raise ValueError("fit-only target matched training config changed")
    event_rows = manifest.get("fit_event_rows")
    if not isinstance(event_rows, list) or len(event_rows) != manifest.get(
        "fit_event_count"
    ):
        raise ValueError("fit-only event roster changed")
    normalized: list[FitOnlyTargetEventV13] = []
    for row in event_rows:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("fit-only event row schema changed")
        (
            event_id,
            patient_id,
            source_index,
            event_record_sha,
            preprocess_receipt_sha,
            target_sha,
            mask_sha,
        ) = row
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
                event_record_sha256=_require_sha256(
                    event_record_sha, field="event.event_record_sha256"
                ),
                preprocess_receipt_sha256=_require_sha256(
                    preprocess_receipt_sha,
                    field="event.preprocess_receipt_sha256",
                ),
                target_sha256=_require_sha256(
                    target_sha, field="event.target_sha256"
                ),
                target_mask_sha256=_require_sha256(
                    mask_sha, field="event.mask_sha256"
                ),
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
    shortcut_parameters = validate_fit_only_shortcut_parameters_v13(
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
        training_event_rows=tuple(
            (row.event_id, row.patient_id) for row in normalized
        ),
        native_event_rows=tuple(),
        training_targets=targets.contiguous(),
        training_target_mask=mask.contiguous(),
        native_targets=empty_targets,
        native_target_mask=empty_mask,
    )
    return LoadedFitOnlyTargetArtifactV13(
        path=source,
        manifest={**manifest, "selection": selection},
        manifest_sha256=manifest_sha,
        receipt_sha256=receipt_sha,
        events=tuple(normalized),
        snapshot=snapshot,
        shortcut_parameters=shortcut_parameters,
    )


__all__ = (
    "FIT_ONLY_TARGET_MANIFEST",
    "FIT_ONLY_TARGET_MASK",
    "FIT_ONLY_TARGET_RECEIPT",
    "FIT_ONLY_TARGET_SCHEMA_V13",
    "FIT_ONLY_TARGETS",
    "FIT_ONLY_SHORTCUT_SCHEMA_V13",
    "MASK_DENSITY_BOUNDARIES_V13",
    "MASK_ONLY_ALGORITHM_V13",
    "PREVALENCE_ONLY_ALGORITHM_V13",
    "TIME_ONLY_ALGORITHM_V13",
    "UNKNOWN_TARGET_CELL_POLICY_V13",
    "FitOnlyTargetEventV13",
    "LoadedFitOnlyTargetArtifactV13",
    "SourceNpyLayout",
    "SourceTargetIndexV13",
    "load_fit_only_target_artifact_v13",
    "load_source_target_index_v13",
    "materialize_fit_only_target_artifact_v13",
    "compute_fit_only_shortcut_parameters_v13",
    "selected_fit_source_rows",
    "validate_fit_only_shortcut_parameters_v13",
)
