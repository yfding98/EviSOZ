"""Executable TUSZ-only native18 SeizureTransformer clean-room runner.

This module is deliberately independent from the frozen common17/ST16
runner.  It keeps the public SeizureTransformer 18-lead signal path and model
semantics, while changing only the data exposure and selection authority:

* labelled TUSZ ``source_train`` is the only training exposure;
* one deterministic upstream-category selection is cached and reused for all
  100 epochs;
* training is scratch RAdam plus unweighted dense BCE, with no gradient clip;
* ``source_dev`` inference consumes a target-free physical identity roster;
* posterior and released-decoder events are frozen before TERM references can
  be opened for scoring; and
* ``source_eval`` has no accepted path or CLI option.

The public paper code used Siena plus TUSZ and selected the best epoch on the
official development set.  This TUSZ-only runner is therefore an
architecture/native-preprocessing reproduction, not a paper-checkpoint or
paper-training-exposure reproduction.  Its primary checkpoint is exactly the
completed epoch-100 checkpoint; partial and earlier epoch states are resume
artifacts only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .canonical_edf_materialization import (
    load_canonical_edf_physical_source_identity,
)
from .continuous_detection_benchmark import (
    aggregate_continuous_detection_metrics,
)
from .eventnet_common17_streaming_v1 import load_common17_manifest
from .onset_collar_scoring_v1 import ordered_onset_collar_matching
from .st16_common17_cleanroom_formal_entry_v1 import (
    provider_target_sample_count,
    validate_source_train_manifest,
)
from . import seizuretransformer_tusz_native18_v1 as native18
from .tusz_canonical_physical_signal_audit_v1 import (
    validate_tusz_canonical_physical_analysis_projection_v1,
    validate_tusz_canonical_physical_duplicate_audit_v1,
)
from src.lookaroundnet_native18.unified_score import (
    score_normalized_source_dev_rows,
)


SCHEMA_VERSION: Final[str] = "seizuretransformer_tusz_native18_runner_v1"
PLAN_SCHEMA_VERSION: Final[str] = "native18_tusz_only_selected_tile_plan_v1"
CACHE_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "native18_tusz_only_selected_tile_cache_contract_v1"
)
CACHE_TILE_SCHEMA_VERSION: Final[str] = (
    "native18_tusz_only_selected_tile_cache_row_v1"
)
CACHE_INVENTORY_SCHEMA_VERSION: Final[str] = (
    "native18_tusz_only_selected_tile_cache_inventory_v1"
)
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "native18_tusz_only_fixed100_checkpoint_v1"
)
MONITORING_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "native18_tusz_only_coarse_monitoring_weights_v1"
)
ROSTER_SCHEMA_VERSION: Final[str] = (
    "native18_target_free_source_dev_roster_v1"
)
PREDICTION_ROW_SCHEMA_VERSION: Final[str] = (
    "native18_source_dev_frozen_prediction_row_v1"
)
PREDICTION_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "native18_source_dev_frozen_prediction_inventory_v1"
)
EXTERNAL19_PREDICTION_ROW_SCHEMA_VERSION: Final[str] = (
    "external_native19_source_dev_diagnostic_prediction_row_v1"
)
EXTERNAL19_PREDICTION_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "seizuretransformer-external19-prediction-manifest-v1"
)
FREEZE_GATE_SCHEMA_VERSION: Final[str] = (
    "native18_source_dev_prediction_freeze_gate_v1"
)
EVALUATION_SCHEMA_VERSION: Final[str] = (
    "native18_source_dev_postfreeze_evaluation_v1"
)
ISOLATED_SOURCE_TRAIN_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_detector_source_train_labeled_manifest_v1"
)
SOURCE_DEV_REFERENCE_SCHEMA_VERSION: Final[str] = (
    "native18_source_dev_reference_manifest_v1"
)
PROVIDER_ID: Final[str] = native18.TUSZ_ONLY_PROFILE_ID
FIXED_EPOCH_COUNT: Final[int] = 100
DEFAULT_SELECTION_SEED: Final[int] = 20260826
DEFAULT_MODEL_SEED: Final[int] = 20260826
CACHE_WORKERS_MAXIMUM: Final[int] = 8
TIMESCORING_COMMIT: Final[str] = "426f8d2b77974641dc9db71884e0812b249ba93b"
TIMESCORING_VERSION: Final[str] = "0.0.7"
_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
_FORBIDDEN_TARGET_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "seizure_events",
        "reference_events",
        "reference_csv_bi_sha256",
        "annotation",
        "doctor_text",
        "excel",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    value: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    supplied = result.get("receipt_sha256")
    if (
        not isinstance(supplied, str)
        or len(supplied) != 64
        or any(character not in "0123456789abcdef" for character in supplied)
    ):
        raise ValueError(f"{context} lacks a lowercase SHA-256 content address")
    result["receipt_sha256"] = _PENDING
    if _canonical_sha256(result) != supplied:
        raise ValueError(f"{context} content address failed replay")
    result["receipt_sha256"] = supplied
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(
    path: Path, value: object, *, replace: bool
) -> None:
    target = path.resolve(strict=False)
    if target.is_symlink() or (not replace and target.exists()):
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_numpy_atomic(
    path: Path, value: np.ndarray, *, replace: bool
) -> None:
    target = path.resolve(strict=False)
    if target.is_symlink() or (not replace and target.exists()):
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".npy.tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, value: object) -> None:
    target = path.resolve(strict=False)
    if target.is_symlink():
        raise PermissionError("native18 checkpoint path may not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pt.tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_payloads_equal(left: object, right: object) -> bool:
    """Semantic equality for weights-only checkpoint payload replay."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            return False
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _torch_payloads_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _torch_payloads_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _install_or_replay_completed_primary(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a completed last checkpoint as primary, or validate its replay."""

    if path.is_symlink():
        raise PermissionError("native18 primary checkpoint may not be a symlink")
    if path.exists():
        observed = torch.load(path, map_location="cpu", weights_only=True)
        if not _torch_payloads_equal(observed, payload):
            raise PermissionError("existing epoch100 primary differs from complete last.pt")
        return
    _atomic_torch_save(path, payload)


def _safe_edf(root: Path, relative_path: str, *, split: str) -> Path:
    expected_prefix = {"source_train": "train", "source_dev": "dev"}.get(split)
    if expected_prefix is None:
        raise PermissionError("source_eval and unknown EDF splits are forbidden")
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != expected_prefix
        or relative.suffix.lower() != ".edf"
    ):
        raise PermissionError("EDF path crosses the authorized train/dev root")
    source = (root / relative).resolve(strict=True)
    source.relative_to(root)
    if source.is_symlink() or not source.is_file():
        raise ValueError("EDF must be a regular non-symlink file")
    return source


def _upstream_event_spans(
    record: Mapping[str, Any], *, sample_count: int | None = None
) -> tuple[tuple[int, int], ...]:
    """Replay released CSV mask indexing: ``int(seconds * 256)``.

    The manifest has already reduced labels to global TERM ``seiz`` intervals.
    Mask union is made explicit so overlapping/adjacent rows cannot double
    count positive samples during category selection.
    """

    support = (
        int(record["target_sample_count_256hz"])
        if sample_count is None
        else int(sample_count)
    )
    if support < 0:
        raise ValueError("target sample count may not be negative")
    events = record.get("seizure_events")
    if not isinstance(events, list):
        raise TypeError("source-train record lacks seizure_events")
    raw: list[tuple[int, int]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("source-train seizure event must be an object")
        start = int(float(event["start_seconds"]) * native18.TARGET_FS_HZ)
        stop = int(float(event["stop_seconds"]) * native18.TARGET_FS_HZ)
        start = max(0, min(support, start))
        stop = max(start, min(support, stop))
        if stop > start:
            raw.append((start, stop))
    raw.sort()
    merged: list[list[int]] = []
    for start, stop in raw:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return tuple((start, stop) for start, stop in merged)


def _positive_samples_in_window(
    spans: Sequence[tuple[int, int]], *, start: int, stop: int
) -> int:
    return sum(
        max(0, min(stop, event_stop) - max(start, event_start))
        for event_start, event_stop in spans
    )


def _tile_id(identity: str, start_sample: int) -> str:
    digest = hashlib.sha256(
        (
            f"{PROVIDER_ID}|{identity}|{start_sample}|"
            f"{native18.TILE_SAMPLES}|selected-cache-v1"
        ).encode("utf-8")
    ).hexdigest()
    return f"N18TILE-{digest[:40]}"


def build_selected_tile_plan_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_receipt_sha256: str,
    seed: int = DEFAULT_SELECTION_SEED,
    complete_source_train_roster: bool,
) -> dict[str, Any]:
    """Enumerate every upstream window, then select category quotas once."""

    if not records:
        raise ValueError("source-train selection roster is empty")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("selection seed must be a nonnegative integer")
    if not isinstance(manifest_receipt_sha256, str) or len(
        manifest_receipt_sha256
    ) != 64:
        raise ValueError("manifest receipt must be a SHA-256")

    enumerated: list[tuple[int, int, str, int]] = []
    category_counts: Counter[str] = Counter()
    excluded_short: list[str] = []
    omitted_terminal_category_counts: Counter[str] = Counter()
    omitted_terminal_positive_samples = 0
    ordered_records = sorted(
        (dict(row) for row in records),
        key=lambda row: str(row["analysis_identity_id"]),
    )
    for record_index, record in enumerate(ordered_records):
        if record.get("model_split") != "source_train" or record.get(
            "official_split"
        ) != "train":
            raise PermissionError("tile selection accepts source_train only")
        identity = str(record["analysis_identity_id"])
        sample_count = int(record["target_sample_count_256hz"])
        spans = _upstream_event_spans(record, sample_count=sample_count)
        starts = native18.upstream_training_tile_starts(sample_count)
        if not starts:
            excluded_short.append(identity)
        whole_second_samples = (
            sample_count // native18.TARGET_FS_HZ
        ) * native18.TARGET_FS_HZ
        if whole_second_samples >= native18.TILE_SAMPLES:
            omitted_start = whole_second_samples - native18.TILE_SAMPLES
            omitted_positive = _positive_samples_in_window(
                spans,
                start=omitted_start,
                stop=omitted_start + native18.TILE_SAMPLES,
            )
            omitted_category = (
                "no_seizure"
                if omitted_positive == 0
                else (
                    "full_seizure"
                    if omitted_positive == native18.TILE_SAMPLES
                    else "partial_seizure"
                )
            )
            omitted_terminal_category_counts[omitted_category] += 1
            omitted_terminal_positive_samples += omitted_positive
        for start in starts:
            positive = _positive_samples_in_window(
                spans, start=start, stop=start + native18.TILE_SAMPLES
            )
            category = (
                "no_seizure"
                if positive == 0
                else (
                    "full_seizure"
                    if positive == native18.TILE_SAMPLES
                    else "partial_seizure"
                )
            )
            category_counts[category] += 1
            enumerated.append((record_index, start, category, positive))

    categories = [row[2] for row in enumerated]
    selected_indices = native18.select_upstream_training_window_indices(
        categories, seed=seed, alpha=0.7, beta=2.0
    )
    if not selected_indices:
        raise ValueError("source-train roster has no partial-seizure training tile")
    selected_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for selection_index, enumerated_index in enumerate(selected_indices):
        record_index, start, category, positive = enumerated[enumerated_index]
        record = ordered_records[record_index]
        identity = str(record["analysis_identity_id"])
        selected_counts[category] += 1
        selected_rows.append(
            {
                "selection_index": selection_index,
                "tile_id": _tile_id(identity, start),
                "analysis_identity_id": identity,
                "patient_id": str(record["patient_id"]),
                "edf_relative_path": str(record["edf_relative_path"]),
                "canonical_source_tensor_sha256": str(
                    record["canonical_source_tensor_sha256"]
                ),
                "target_sample_count_256hz": int(
                    record["target_sample_count_256hz"]
                ),
                "start_sample": int(start),
                "stop_sample_exclusive": int(start + native18.TILE_SAMPLES),
                "category": category,
                "positive_sample_count": int(positive),
            }
        )
    if len({row["tile_id"] for row in selected_rows}) != len(selected_rows):
        raise RuntimeError("selected native18 tile identifiers are not unique")
    selected_positive_samples = sum(
        int(row["positive_sample_count"]) for row in selected_rows
    )
    selected_dense_samples = len(selected_rows) * native18.TILE_SAMPLES
    selected_positive_by_category: Counter[str] = Counter()
    for row in selected_rows:
        selected_positive_by_category[str(row["category"])] += int(
            row["positive_sample_count"]
        )
    signal_npy_bytes_per_tile = 128 + 18 * native18.TILE_SAMPLES * 4
    target_npy_bytes_per_tile = 128 + native18.TILE_SAMPLES
    payload_bytes_per_tile = signal_npy_bytes_per_tile + target_npy_bytes_per_tile

    plan = _content_address(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "claim_status": "tusz_only_source_train_selected_cache_plan",
            "manifest_receipt_sha256": manifest_receipt_sha256,
            "source_train_record_count": len(ordered_records),
            "source_train_identity_roster_sha256": _canonical_sha256(
                [row["analysis_identity_id"] for row in ordered_records]
            ),
            "complete_source_train_roster": bool(complete_source_train_roster),
            "selection_seed": seed,
            "selection_determinism_repair": (
                "explicit_numpy_generator_replaces_upstream_unseeded_sklearn_shuffle"
            ),
            "window_seconds": native18.TILE_SECONDS,
            "window_samples": native18.TILE_SAMPLES,
            "hop_seconds": native18.TRAIN_HOP_SAMPLES / native18.TARGET_FS_HZ,
            "hop_samples": native18.TRAIN_HOP_SAMPLES,
            "upstream_exclusive_terminal_window_bug_retained": True,
            "upstream_exact_terminal_full_window_omission": {
                "omitted_window_count": sum(
                    omitted_terminal_category_counts.values()
                ),
                "omitted_category_counts": dict(
                    sorted(omitted_terminal_category_counts.items())
                ),
                "omitted_positive_sample_count": omitted_terminal_positive_samples,
                "one_terminal_full_window_omitted_for_each_record_with_at_least_60_whole_seconds": True,
            },
            "label_projection": "int(float(seconds)*256)_half_open_mask_union",
            "enumerated_window_count": len(enumerated),
            "enumerated_category_counts": dict(sorted(category_counts.items())),
            "selection_quotas": {
                "partial_seizure": "all",
                "full_seizure": "min(available,floor(0.7*partial))",
                "no_seizure": "min(available,floor(2.0*partial))",
            },
            "selected_window_count": len(selected_rows),
            "selected_category_counts": dict(sorted(selected_counts.items())),
            "selected_positive_sample_count": selected_positive_samples,
            "selected_dense_sample_count": selected_dense_samples,
            "selected_positive_sample_fraction": (
                selected_positive_samples / selected_dense_samples
            ),
            "selected_positive_sample_count_by_tile_category": dict(
                sorted(selected_positive_by_category.items())
            ),
            "selected_record_count": len(
                {row["analysis_identity_id"] for row in selected_rows}
            ),
            "excluded_no_upstream_window_record_count": len(excluded_short),
            "excluded_no_upstream_window_identity_roster_sha256": (
                _canonical_sha256(sorted(excluded_short))
            ),
            "selected_tiles": selected_rows,
            "cache_storage_estimate": {
                "signal_npy_bytes_per_tile": signal_npy_bytes_per_tile,
                "target_npy_bytes_per_tile": target_npy_bytes_per_tile,
                "payload_bytes_per_tile": payload_bytes_per_tile,
                "selected_payload_bytes": payload_bytes_per_tile
                * len(selected_rows),
                "selected_payload_GiB": (
                    payload_bytes_per_tile * len(selected_rows) / 1073741824.0
                ),
                "whole_record_signal_cache_persisted": False,
                "atomic_write_peak_extra_payload_bytes": max(
                    signal_npy_bytes_per_tile, target_npy_bytes_per_tile
                ),
                "sidecar_and_inventory_JSON_bytes_not_known_until_materialization": True,
            },
            "permissions": {
                "source_train_TERM_targets_used": True,
                "source_dev_opened": False,
                "source_eval_opened": False,
                "EDF_annotations_opened": False,
            },
            "receipt_sha256": _PENDING,
        }
    )
    return plan


def build_selected_tile_plan(
    manifest_path: str | Path,
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    maximum_source_records: int | None = None,
    parent_manifest_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve(strict=True)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("schema_version") == ISOLATED_SOURCE_TRAIN_SCHEMA_VERSION:
        manifest = validate_source_train_manifest(raw)
        if parent_manifest_receipt_sha256 is None:
            raise ValueError(
                "isolated source-train planning requires the parent common17 manifest receipt"
            )
        rows = _adapt_isolated_source_train_rows(manifest["records"])
        manifest_receipt = parent_manifest_receipt_sha256
    else:
        manifest = load_common17_manifest(source, require_complete=True)
        rows = [
            dict(row)
            for row in manifest["records"]
            if row["model_split"] == "source_train"
        ]
        manifest_receipt = str(manifest["receipt_sha256"])
    rows.sort(key=lambda row: str(row["analysis_identity_id"]))
    complete = maximum_source_records is None
    if maximum_source_records is not None:
        if maximum_source_records < 1:
            raise ValueError("maximum_source_records must be positive")
        rows = rows[:maximum_source_records]
    return build_selected_tile_plan_from_records(
        rows,
        manifest_receipt_sha256=manifest_receipt,
        seed=seed,
        complete_source_train_roster=complete,
    )


def _adapt_isolated_source_train_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt the validated train-only firewall artifact to the frozen planner."""

    adapted: list[dict[str, Any]] = []
    for row in rows:
        adapted.append(
            {
                "analysis_identity_id": str(row["analysis_identity_id"]),
                "patient_id": str(row["local_patient_id"]),
                "edf_relative_path": str(row["local_edf_path"]),
                "model_split": "source_train",
                "official_split": "train",
                "canonical_source_tensor_sha256": str(
                    row["canonical_physical_source_tensor_sha256"]
                ),
                "target_sample_count_256hz": provider_target_sample_count(row),
                "seizure_events": [
                    {
                        "start_seconds": float(event["start_seconds_decimal"]),
                        "stop_seconds": float(event["stop_seconds_decimal"]),
                    }
                    for event in row["global_TERM_seiz_intervals_seconds"]
                ],
            }
        )
    adapted.sort(key=lambda row: str(row["analysis_identity_id"]))
    return adapted


def _tile_paths(cache_root: Path, tile_id: str) -> tuple[Path, Path, Path]:
    if not tile_id.startswith("N18TILE-") or any(
        character not in "0123456789abcdef" for character in tile_id[8:]
    ):
        raise ValueError("native18 tile identifier is malformed")
    directory = cache_root / "tiles" / tile_id[8:10] / tile_id
    return directory / "signal.npy", directory / "target.npy", directory / "receipt.json"


def _cache_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _content_address(
        {
            "schema_version": CACHE_CONTRACT_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "plan_receipt_sha256": str(plan["receipt_sha256"]),
            "signal_shape": [18, native18.TILE_SAMPLES],
            "signal_dtype": "float32",
            "target_shape": [native18.TILE_SAMPLES],
            "target_dtype": "uint8",
            "tile_transform": (
                "native18_whole_record_montage_population_zscore_FFT_resample_"
                "then_tile_local_causal_filters"
            ),
            "selected_tiles_reused_across_all_100_epochs": True,
            "source_eval_opened": False,
            "receipt_sha256": _PENDING,
        }
    )


def _load_json_content_address(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return _validate_content_address(value, context=context)


def _install_or_replay_json(
    path: Path, expected: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        observed = _load_json_content_address(path, context=context)
        if observed != dict(expected):
            raise PermissionError(f"existing {context} differs from requested content")
        return observed
    _write_json_atomic(path, expected, replace=False)
    return dict(expected)


def _source_rate(record: Any) -> float:
    rows = record.source_header_receipt["channel_signal_headers"]
    rates = {
        (
            int(row["sampling_rate_numerator"]),
            int(row["sampling_rate_denominator"]),
        )
        for row in rows
    }
    if len(rates) != 1:
        raise ValueError("canonical native18 source has multiple clocks")
    numerator, denominator = next(iter(rates))
    return float(Fraction(numerator, denominator))


def _transform_native18_edf(
    edf_path: Path,
    *,
    expected_source_tensor_sha256: str,
    expected_target_sample_count: int,
) -> native18.Native18Record:
    physical = load_canonical_edf_physical_source_identity(edf_path)
    if (
        physical.source_header_receipt["source_tensor_sha256"]
        != expected_source_tensor_sha256
    ):
        raise PermissionError("canonical EDF physical tensor hash drifted")
    observed = np.asarray(
        physical.observed_signal_volts.detach().cpu().numpy(), dtype=np.float32
    )
    carrier = native18.materialize_upstream_literal_referential19(
        observed, physical.observed_channel_ids
    )
    transformed = native18.transform_upstream_native18_record(
        carrier, source_sampling_rate_hz=_source_rate(physical)
    )
    if transformed.signal.shape != (18, expected_target_sample_count):
        raise RuntimeError("native18 transformed support differs from frozen inventory")
    return transformed


def _dense_target(record: Mapping[str, Any], *, sample_count: int) -> np.ndarray:
    target = np.zeros(sample_count, dtype=np.uint8)
    for start, stop in _upstream_event_spans(record, sample_count=sample_count):
        target[start:stop] = 1
    return target


def _cache_tile_sidecar(
    *,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    transformed: native18.Native18Record,
    signal_path: Path,
    target_path: Path,
) -> dict[str, Any]:
    return _content_address(
        {
            "schema_version": CACHE_TILE_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "plan_receipt_sha256": str(plan["receipt_sha256"]),
            "cache_contract_receipt_sha256": str(contract["receipt_sha256"]),
            "tile_id": str(row["tile_id"]),
            "analysis_identity_id": str(row["analysis_identity_id"]),
            "start_sample": int(row["start_sample"]),
            "category": str(row["category"]),
            "positive_sample_count": int(row["positive_sample_count"]),
            "native18_transform_receipt_sha256": transformed.receipt[
                "receipt_sha256"
            ],
            "referential_carrier_receipt_sha256": transformed.receipt[
                "referential_carrier_receipt_sha256"
            ],
            "signal_relative_path": str(signal_path.relative_to(signal_path.parents[3])),
            "target_relative_path": str(target_path.relative_to(target_path.parents[3])),
            "signal_npy_sha256": _file_sha256(signal_path),
            "target_npy_sha256": _file_sha256(target_path),
            "signal_size_bytes": signal_path.stat().st_size,
            "target_size_bytes": target_path.stat().st_size,
            "source_eval_opened": False,
            "receipt_sha256": _PENDING,
        }
    )


def _validate_cached_tile(
    row: Mapping[str, Any],
    *,
    cache_root: Path,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    verify_payload_hashes: bool,
) -> dict[str, Any] | None:
    validated = _validate_cached_tile_with_filesystem_snapshot(
        row,
        cache_root=cache_root,
        plan=plan,
        contract=contract,
        verify_payload_hashes=verify_payload_hashes,
    )
    return None if validated is None else validated[0]


def _regular_file_identity(path: Path) -> tuple[int, int, int, int, int, int] | None:
    """Return a cheap mutation seal for one non-symlink regular file.

    ``ctime_ns`` prevents an in-place writer from hiding a change by restoring
    ``mtime``; inode/device prevent an atomic replacement from looking like the
    previously hashed payload.  This is deliberately a mutation detector, not
    a substitute for the initial content hash.
    """

    try:
        observed = path.lstat()
    except OSError:
        return None
    if (observed.st_mode & 0o170000) != 0o100000:
        return None
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
        int(observed.st_nlink),
    )


def _cache_tile_filesystem_snapshot(
    signal_path: Path, target_path: Path, receipt_path: Path
) -> tuple[tuple[int, int, int, int, int, int], ...] | None:
    identities = tuple(
        _regular_file_identity(path)
        for path in (signal_path, target_path, receipt_path)
    )
    if any(identity is None for identity in identities):
        return None
    return tuple(identity for identity in identities if identity is not None)


def _validate_cached_tile_with_filesystem_snapshot(
    row: Mapping[str, Any],
    *,
    cache_root: Path,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    verify_payload_hashes: bool,
) -> tuple[
    dict[str, Any], tuple[tuple[int, int, int, int, int, int], ...]
] | None:
    """Validate a tile and bind that validation to the exact file versions.

    The before/after identity comparison closes the path-replacement window
    around payload hashing.  A caller may later compare the returned snapshot
    while holding the cache invocation lock and safely reuse this invocation's
    strong hash result if every file is unchanged.
    """

    signal_path, target_path, receipt_path = _tile_paths(
        cache_root, str(row["tile_id"])
    )
    before = _cache_tile_filesystem_snapshot(
        signal_path, target_path, receipt_path
    )
    if before is None:
        return None
    try:
        receipt = _load_json_content_address(
            receipt_path, context=f"cache tile {row['tile_id']}"
        )
        required = {
            "schema_version": CACHE_TILE_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "plan_receipt_sha256": plan["receipt_sha256"],
            "cache_contract_receipt_sha256": contract["receipt_sha256"],
            "tile_id": row["tile_id"],
            "analysis_identity_id": row["analysis_identity_id"],
            "start_sample": row["start_sample"],
            "category": row["category"],
            "positive_sample_count": row["positive_sample_count"],
            "source_eval_opened": False,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            return None
        signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        target = np.load(target_path, mmap_mode="r", allow_pickle=False)
        geometry_ok = (
            signal.shape == (18, native18.TILE_SAMPLES)
            and signal.dtype == np.dtype("float32")
            and target.shape == (native18.TILE_SAMPLES,)
            and target.dtype == np.dtype("uint8")
            and int(np.sum(target, dtype=np.int64))
            == int(row["positive_sample_count"])
            and signal_path.stat().st_size
            == int(receipt["signal_size_bytes"])
            and target_path.stat().st_size
            == int(receipt["target_size_bytes"])
        )
        del signal, target
        if not geometry_ok:
            return None
        if verify_payload_hashes and (
            _file_sha256(signal_path) != receipt["signal_npy_sha256"]
            or _file_sha256(target_path) != receipt["target_npy_sha256"]
        ):
            return None
        after = _cache_tile_filesystem_snapshot(
            signal_path, target_path, receipt_path
        )
        if after is None or after != before:
            return None
        return receipt, after
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _cache_inventory_tile_row(
    row: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "tile_id": row["tile_id"],
        "tile_receipt_sha256": receipt["receipt_sha256"],
        "signal_npy_sha256": receipt["signal_npy_sha256"],
        "target_npy_sha256": receipt["target_npy_sha256"],
        "signal_size_bytes": receipt["signal_size_bytes"],
        "target_size_bytes": receipt["target_size_bytes"],
    }


def _validate_cache_workers(workers: int) -> None:
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= CACHE_WORKERS_MAXIMUM
    ):
        raise ValueError(
            f"native18 cache workers must be an integer in [1,{CACHE_WORKERS_MAXIMUM}]"
        )


def _cache_invocation_lock_path(cache_root: Path) -> Path:
    return cache_root / "cache_invocation.lock"


@contextmanager
def _exclusive_cache_invocation_lock(cache_root: Path) -> Iterator[Path]:
    """Serialize one complete cache audit/materialize/freeze invocation.

    The lock covers the initial strong payload audit, every record worker, and
    final inventory publication.  It is advisory but all writer entry points
    in this runner pass through it; kernel ``flock`` also makes it crash
    releasable without deleting a possibly live ownership file.
    """

    root_argument = Path(cache_root)
    if root_argument.is_symlink():
        raise PermissionError("native18 cache root may not be a symlink")
    root = root_argument.resolve(strict=True)
    if not root.is_dir():
        raise PermissionError("native18 cache root must be a regular directory")
    lock_path = _cache_invocation_lock_path(root)
    if lock_path.parent.is_symlink():
        raise PermissionError("native18 cache invocation lock parent may not be a symlink")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        observed = os.fstat(descriptor)
        if (
            (observed.st_mode & 0o170000) != 0o100000
            or int(observed.st_nlink) != 1
        ):
            raise PermissionError(
                "native18 cache invocation lock must be a singly linked regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "native18 cache is already owned by another materialization invocation"
            ) from error
        owner = _canonical_bytes(
            {
                "cache_root": str(root),
                "pid": os.getpid(),
                "lock_semantics": "whole_invocation_kernel_flock_crash_released",
            }
        )
        os.ftruncate(descriptor, 0)
        os.write(descriptor, owner + b"\n")
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _record_cache_lock_path(cache_root: Path, identity: str) -> Path:
    if not isinstance(identity, str) or not identity:
        raise ValueError("cache record lock identity must be non-empty")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return cache_root / "record_locks" / f"{digest}.lock"


@contextmanager
def _exclusive_record_cache_lock(
    cache_root: Path, identity: str
) -> Iterator[Path]:
    """Hold an automatically crash-released advisory lock for one record.

    Lock files persist as tiny ownership slots, but ``flock`` itself is tied to
    the open file descriptor and is released by the kernel if a worker dies.
    A resumed invocation therefore never needs unsafe stale-lock deletion.
    """

    lock_path = _record_cache_lock_path(cache_root, identity)
    if lock_path.parent.is_symlink():
        raise PermissionError("native18 record lock directory may not be a symlink")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        stat = os.fstat(descriptor)
        if not (stat.st_mode & 0o170000) == 0o100000:
            raise PermissionError("native18 record lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"native18 cache record is already owned by another worker: {identity}"
            ) from error
        owner = _canonical_bytes(
            {
                "analysis_identity_id": identity,
                "pid": os.getpid(),
                "lock_semantics": "kernel_flock_crash_released",
            }
        )
        os.ftruncate(descriptor, 0)
        os.write(descriptor, owner + b"\n")
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _plan_cache_record_tasks(
    *,
    by_record: Mapping[str, Sequence[Mapping[str, Any]]],
    source_rows: Mapping[str, Mapping[str, Any]],
    tusz_root: Path,
    cache_root: Path,
    plan_receipt_sha256: str,
    contract_receipt_sha256: str,
    maximum_new_tiles: int | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Freeze an ordered, record-disjoint work roster before any process starts."""

    if maximum_new_tiles is not None and maximum_new_tiles < 1:
        raise ValueError("maximum_new_tiles must be positive")
    total_missing_tiles = sum(len(rows) for rows in by_record.values())
    remaining = total_missing_tiles if maximum_new_tiles is None else maximum_new_tiles
    tasks: list[dict[str, Any]] = []
    scheduled_tile_ids: set[str] = set()
    for identity in sorted(by_record):
        if remaining <= 0:
            break
        source_row = source_rows.get(identity)
        if source_row is None:
            raise PermissionError("selected tile identity left source_train manifest")
        if source_row.get("model_split") != "source_train":
            raise PermissionError("cache worker task left source_train")
        rows = sorted(
            (dict(row) for row in by_record[identity]),
            key=lambda row: int(row["start_sample"]),
        )
        rows = rows[:remaining]
        if not rows:
            continue
        if any(str(row["analysis_identity_id"]) != identity for row in rows):
            raise PermissionError("cache worker task mixes record identities")
        tile_ids = {str(row["tile_id"]) for row in rows}
        if len(tile_ids) != len(rows) or scheduled_tile_ids.intersection(tile_ids):
            raise RuntimeError("cache worker tasks do not own disjoint tile paths")
        scheduled_tile_ids.update(tile_ids)
        tasks.append(
            {
                "task_index": len(tasks),
                "analysis_identity_id": identity,
                "tusz_root": str(tusz_root),
                "cache_root": str(cache_root),
                "source_row": dict(source_row),
                "selected_rows": rows,
                "plan_receipt_sha256": plan_receipt_sha256,
                "contract_receipt_sha256": contract_receipt_sha256,
            }
        )
        remaining -= len(rows)
    scheduled_count = sum(len(task["selected_rows"]) for task in tasks)
    expected_scheduled = (
        total_missing_tiles
        if maximum_new_tiles is None
        else min(total_missing_tiles, maximum_new_tiles)
    )
    if scheduled_count != expected_scheduled:
        raise RuntimeError("cache task partition lost one or more requested tiles")
    return tasks, scheduled_count < total_missing_tiles


def _materialize_selected_record_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Transform one source-train EDF and publish only its disjoint tiles."""

    began = time.perf_counter()
    task = dict(payload)
    identity = str(task["analysis_identity_id"])
    source_row = dict(task["source_row"])
    rows = [dict(row) for row in task["selected_rows"]]
    if (
        source_row.get("model_split") != "source_train"
        or str(source_row.get("analysis_identity_id")) != identity
        or not rows
        or any(str(row.get("analysis_identity_id")) != identity for row in rows)
    ):
        raise PermissionError("native18 cache worker payload crossed source-train lineage")
    if len({str(row["tile_id"]) for row in rows}) != len(rows):
        raise RuntimeError("native18 record worker repeats a tile path")
    root = Path(str(task["tusz_root"])).resolve(strict=True)
    cache = Path(str(task["cache_root"])).resolve(strict=True)
    if cache.is_symlink() or not cache.is_dir():
        raise PermissionError("native18 worker cache root drifted")
    # Cache preprocessing is CPU-only.  One Torch thread per spawned worker
    # prevents workers=6 from multiplying the host thread pool.
    if task.get("limit_host_threads") is True:
        torch.set_num_threads(1)
    with _exclusive_record_cache_lock(cache, identity):
        edf_path = _safe_edf(
            root, str(source_row["edf_relative_path"]), split="source_train"
        )
        transformed = _transform_native18_edf(
            edf_path,
            expected_source_tensor_sha256=str(
                source_row["canonical_source_tensor_sha256"]
            ),
            expected_target_sample_count=int(
                source_row["target_sample_count_256hz"]
            ),
        )
        dense = _dense_target(source_row, sample_count=transformed.signal.shape[1])
        outputs: list[dict[str, Any]] = []
        for row in rows:
            signal, target = native18.materialize_upstream_training_tile(
                transformed, dense, start_sample=int(row["start_sample"])
            )
            target_u8 = np.ascontiguousarray(target, dtype=np.uint8)
            if (
                native18.upstream_training_window_category(target_u8)
                != row["category"]
                or int(target_u8.sum(dtype=np.int64))
                != int(row["positive_sample_count"])
            ):
                raise RuntimeError("materialized native18 target category drifted")
            signal_path, target_path, receipt_path = _tile_paths(
                cache, str(row["tile_id"])
            )
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            _save_numpy_atomic(signal_path, signal, replace=True)
            _save_numpy_atomic(target_path, target_u8, replace=True)
            sidecar = _cache_tile_sidecar(
                row=row,
                plan={"receipt_sha256": task["plan_receipt_sha256"]},
                contract={"receipt_sha256": task["contract_receipt_sha256"]},
                transformed=transformed,
                signal_path=signal_path,
                target_path=target_path,
            )
            _write_json_atomic(receipt_path, sidecar, replace=True)
            outputs.append(
                {
                    "tile_id": row["tile_id"],
                    "tile_receipt_sha256": sidecar["receipt_sha256"],
                }
            )
    return {
        "task_index": int(task["task_index"]),
        "analysis_identity_id": identity,
        "new_tile_count": len(outputs),
        "tile_outputs": outputs,
        "worker_pid": os.getpid(),
        "worker_CUDA_initialized": torch.cuda.is_initialized(),
        "wall_seconds": time.perf_counter() - began,
    }


def _ordered_bounded_cache_workers(
    tasks: Sequence[Mapping[str, Any]], *, workers: int
) -> Iterator[dict[str, Any]]:
    """Yield process results in task order with at most 2N live futures."""

    maximum_inflight = min(len(tasks), max(workers, 2 * workers))
    executor = ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    )
    futures: dict[int, Future[dict[str, Any]]] = {}
    next_submit = 0
    next_yield = 0
    try:
        while next_yield < len(tasks):
            while next_submit < len(tasks) and len(futures) < maximum_inflight:
                futures[next_submit] = executor.submit(
                    _materialize_selected_record_worker, dict(tasks[next_submit])
                )
                next_submit += 1
            result = futures.pop(next_yield).result()
            if int(result.get("task_index", -1)) != next_yield:
                raise RuntimeError("native18 cache worker result order drifted")
            yield result
            next_yield += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def materialize_selected_tile_cache(
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    seed: int = DEFAULT_SELECTION_SEED,
    maximum_source_records: int | None = None,
    maximum_new_tiles: int | None = None,
    progress_every_records: int = 25,
    workers: int = 1,
    tusz_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build or resume the cache under one whole-invocation writer lock."""

    cache_argument = Path(cache_dir)
    if cache_argument.is_symlink():
        raise PermissionError("native18 cache root may not be a symlink")
    cache = cache_argument.resolve(strict=False)
    cache.mkdir(parents=True, exist_ok=True)
    with _exclusive_cache_invocation_lock(cache):
        return _materialize_selected_tile_cache_locked(
            manifest_path=manifest_path,
            cache_dir=cache,
            seed=seed,
            maximum_source_records=maximum_source_records,
            maximum_new_tiles=maximum_new_tiles,
            progress_every_records=progress_every_records,
            workers=workers,
            tusz_root=tusz_root,
        )


def _materialize_selected_tile_cache_locked(
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    seed: int = DEFAULT_SELECTION_SEED,
    maximum_source_records: int | None = None,
    maximum_new_tiles: int | None = None,
    progress_every_records: int = 25,
    workers: int = 1,
    tusz_root: str | Path | None = None,
) -> dict[str, Any]:
    """Implementation entered only by ``materialize_selected_tile_cache``."""

    if maximum_new_tiles is not None and maximum_new_tiles < 1:
        raise ValueError("maximum_new_tiles must be positive")
    if progress_every_records < 1:
        raise ValueError("progress_every_records must be positive")
    _validate_cache_workers(workers)
    manifest_source = Path(manifest_path).resolve(strict=True)
    cache = Path(cache_dir).resolve(strict=False)
    if cache.is_symlink():
        raise PermissionError("native18 cache root may not be a symlink")
    cache.mkdir(parents=True, exist_ok=True)
    manifest_raw = json.loads(manifest_source.read_text(encoding="utf-8"))
    isolated_source_train = (
        manifest_raw.get("schema_version") == ISOLATED_SOURCE_TRAIN_SCHEMA_VERSION
    )
    if isolated_source_train:
        manifest = validate_source_train_manifest(manifest_raw)
        existing_plan_path = cache / "selected_tile_plan.json"
        if existing_plan_path.is_symlink() or not existing_plan_path.is_file():
            raise FileNotFoundError(
                "isolated source-train cache replay requires the existing parent-bound plan"
            )
        existing_plan = _load_json_content_address(
            existing_plan_path, context="native18 existing selected tile plan"
        )
        parent_receipt = str(existing_plan["manifest_receipt_sha256"])
        source_record_rows = _adapt_isolated_source_train_rows(manifest["records"])
        plan = build_selected_tile_plan_from_records(
            source_record_rows,
            manifest_receipt_sha256=parent_receipt,
            seed=seed,
            complete_source_train_roster=maximum_source_records is None,
        )
        if maximum_source_records is not None:
            raise PermissionError(
                "isolated source-train replay is admitted only for the complete roster"
            )
        if plan != existing_plan:
            raise PermissionError(
                "isolated source-train rows do not replay the parent-bound selected plan"
            )
        manifest_receipt_sha256 = parent_receipt
        manifest_content_sha256 = str(manifest["content_sha256"])
        manifest_schema_version = ISOLATED_SOURCE_TRAIN_SCHEMA_VERSION
        bound_tusz_root = None if tusz_root is None else Path(tusz_root).resolve(strict=True)
    else:
        manifest = load_common17_manifest(manifest_source, require_complete=True)
        source_record_rows = [
            dict(row)
            for row in manifest["records"]
            if row["model_split"] == "source_train"
        ]
        plan = build_selected_tile_plan_from_records(
            source_record_rows,
            manifest_receipt_sha256=str(manifest["receipt_sha256"]),
            seed=seed,
            complete_source_train_roster=maximum_source_records is None,
        )
        if maximum_source_records is not None:
            source_record_rows = sorted(
                source_record_rows, key=lambda row: str(row["analysis_identity_id"])
            )[:maximum_source_records]
            plan = build_selected_tile_plan_from_records(
                source_record_rows,
                manifest_receipt_sha256=str(manifest["receipt_sha256"]),
                seed=seed,
                complete_source_train_roster=False,
            )
        manifest_receipt_sha256 = str(manifest["receipt_sha256"])
        manifest_content_sha256 = None
        manifest_schema_version = str(manifest["schema_version"])
        legacy_root = Path(manifest["source_bindings"]["tusz_root"]).resolve(strict=True)
        bound_tusz_root = (
            legacy_root
            if tusz_root is None
            else Path(tusz_root).resolve(strict=True)
        )
        if bound_tusz_root != legacy_root:
            raise PermissionError("explicit TUSZ root differs from legacy manifest binding")
    plan = _install_or_replay_json(
        cache / "selected_tile_plan.json", plan, context="native18 selected tile plan"
    )
    contract = _install_or_replay_json(
        cache / "cache_contract.json",
        _cache_contract(plan),
        context="native18 cache contract",
    )
    source_rows = {
        str(row["analysis_identity_id"]): dict(row)
        for row in source_record_rows
    }
    selected = [dict(row) for row in plan["selected_tiles"]]
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reused: dict[str, dict[str, Any]] = {}
    reused_filesystem_snapshots: dict[
        str, tuple[tuple[int, int, int, int, int, int], ...]
    ] = {}
    invalid_existing_count = 0
    for row in selected:
        validated = _validate_cached_tile_with_filesystem_snapshot(
            row,
            cache_root=cache,
            plan=plan,
            contract=contract,
            verify_payload_hashes=True,
        )
        if validated is None:
            paths = _tile_paths(cache, str(row["tile_id"]))
            if any(path.exists() or path.is_symlink() for path in paths):
                invalid_existing_count += 1
            by_record[str(row["analysis_identity_id"])].append(row)
        else:
            receipt, filesystem_snapshot = validated
            tile_id = str(row["tile_id"])
            reused[tile_id] = receipt
            reused_filesystem_snapshots[tile_id] = filesystem_snapshot
    began = time.perf_counter()
    new_tile_count = 0
    processed_record_count = 0
    worker_pids: set[int] = set()
    tasks, stopped_at_limit = _plan_cache_record_tasks(
        by_record=by_record,
        source_rows=source_rows,
        tusz_root=(
            bound_tusz_root
            if bound_tusz_root is not None
            else Path("/isolated-source-train-replay-without-new-IO")
        ),
        cache_root=cache,
        plan_receipt_sha256=str(plan["receipt_sha256"]),
        contract_receipt_sha256=str(contract["receipt_sha256"]),
        maximum_new_tiles=maximum_new_tiles,
    )
    if tasks and bound_tusz_root is None:
        raise PermissionError(
            "isolated source-train cache has missing tiles and requires --tusz-root"
        )
    for task in tasks:
        task["limit_host_threads"] = workers > 1
    if workers == 1:
        results: Iterator[dict[str, Any]] = (
            _materialize_selected_record_worker(task) for task in tasks
        )
        execution_mode = "serial_record_worker_in_parent"
    else:
        results = _ordered_bounded_cache_workers(tasks, workers=workers)
        execution_mode = "spawn_record_parallel_parent_ordered_bounded"
    for result in results:
        if result.get("worker_CUDA_initialized") is not False:
            raise PermissionError("native18 cache worker initialized CUDA")
        worker_pids.add(int(result["worker_pid"]))
        new_tile_count += int(result["new_tile_count"])
        processed_record_count += 1
        if processed_record_count % progress_every_records == 0:
            print(
                json.dumps(
                    {
                        "stage": "native18_cache_progress",
                        "processed_record_count": processed_record_count,
                        "scheduled_record_count": len(tasks),
                        "new_tile_count": new_tile_count,
                        "reused_tile_count": len(reused),
                        "workers": workers,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if new_tile_count != sum(len(task["selected_rows"]) for task in tasks):
        raise RuntimeError("native18 cache workers did not publish every scheduled tile")

    fast_path_candidate = (
        invalid_existing_count == 0
        and not by_record
        and not tasks
        and new_tile_count == 0
        and processed_record_count == 0
        and len(reused) == len(selected)
        and len(reused_filesystem_snapshots) == len(selected)
    )
    post_initial_audit_filesystem_drift_detected = False
    if fast_path_candidate:
        for row in selected:
            tile_id = str(row["tile_id"])
            current_snapshot = _cache_tile_filesystem_snapshot(
                *_tile_paths(cache, tile_id)
            )
            if current_snapshot != reused_filesystem_snapshots[tile_id]:
                post_initial_audit_filesystem_drift_detected = True
                break

    same_invocation_verified_receipt_fast_path = (
        fast_path_candidate
        and not post_initial_audit_filesystem_drift_detected
    )
    valid_rows: list[dict[str, Any]] = []
    if same_invocation_verified_receipt_fast_path:
        valid_rows = [
            _cache_inventory_tile_row(row, reused[str(row["tile_id"])])
            for row in selected
        ]
        final_inventory_validation_mode = (
            "same_invocation_strong_hash_receipts_reused_under_exclusive_lock_"
            "after_unchanged_filesystem_identity_seal"
        )
    else:
        for row in selected:
            receipt = _validate_cached_tile(
                row,
                cache_root=cache,
                plan=plan,
                contract=contract,
                verify_payload_hashes=True,
            )
            if receipt is not None:
                valid_rows.append(_cache_inventory_tile_row(row, receipt))
        final_inventory_validation_mode = "full_payload_hash_revalidation"
    complete = len(valid_rows) == len(selected)
    inventory = _content_address(
        {
            "schema_version": CACHE_INVENTORY_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "manifest_path": str(manifest_source),
            "manifest_file_sha256": _file_sha256(manifest_source),
            "manifest_schema_version": manifest_schema_version,
            "manifest_content_sha256": manifest_content_sha256,
            "manifest_receipt_sha256": manifest_receipt_sha256,
            "isolated_source_train_manifest_used": isolated_source_train,
            "plan_receipt_sha256": plan["receipt_sha256"],
            "cache_contract_receipt_sha256": contract["receipt_sha256"],
            "complete_source_train_roster": plan["complete_source_train_roster"],
            "selected_tile_count": len(selected),
            "valid_cached_tile_count": len(valid_rows),
            "cache_complete": complete,
            "reused_content_verified_tile_count": len(reused),
            "newly_materialized_tile_count": new_tile_count,
            "invalid_existing_tile_count": invalid_existing_count,
            "maximum_new_tiles": maximum_new_tiles,
            "stopped_at_invocation_limit": stopped_at_limit,
            "cache_worker_execution": {
                "execution_mode": execution_mode,
                "requested_workers": workers,
                "maximum_workers": CACHE_WORKERS_MAXIMUM,
                "scheduled_record_task_count": len(tasks),
                "processed_record_task_count": processed_record_count,
                "worker_pids": sorted(worker_pids),
                "record_tasks_sorted_by_analysis_identity": True,
                "record_task_results_consumed_in_sorted_order": True,
                "one_task_per_record": True,
                "tile_paths_disjoint_across_tasks": True,
                "whole_invocation_kernel_flock_held": True,
                "per_record_kernel_flock_used": True,
                "maximum_inflight_future_bound": (
                    1 if workers == 1 else 2 * workers
                ),
                "final_inventory_validation_parallelized": False,
                "final_inventory_validation_mode": final_inventory_validation_mode,
                "same_invocation_verified_receipt_fast_path": (
                    same_invocation_verified_receipt_fast_path
                ),
                "payload_hash_reverification_skipped_tile_count": (
                    len(valid_rows)
                    if same_invocation_verified_receipt_fast_path
                    else 0
                ),
                "filesystem_identity_seal_fields": [
                    "st_dev",
                    "st_ino",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                    "st_nlink",
                ],
                "post_initial_audit_filesystem_drift_detected": (
                    post_initial_audit_filesystem_drift_detected
                ),
                "worker_CUDA_initialized": False,
            },
            "tile_inventory": valid_rows,
            "total_npy_bytes": sum(
                int(row["signal_size_bytes"]) + int(row["target_size_bytes"])
                for row in valid_rows
            ),
            "selected_positive_sample_count": plan[
                "selected_positive_sample_count"
            ],
            "selected_dense_sample_count": plan["selected_dense_sample_count"],
            "selected_positive_sample_fraction": plan[
                "selected_positive_sample_fraction"
            ],
            "upstream_exact_terminal_full_window_omission": plan[
                "upstream_exact_terminal_full_window_omission"
            ],
            "storage_contract": {
                **plan["cache_storage_estimate"],
                "persisted_payload_is_selected_tiles_only": True,
                "whole_record_native18_arrays_are_process_memory_only_and_released_after_each_record": True,
                "full_record_plus_selected_tile_duplicate_disk_cache": False,
                "atomic_publication_uses_same_directory_temporary_file": True,
                "atomic_peak_extra_upper_bound_bytes": 64 * 1024 * 1024,
                "metadata_upper_bound_bytes": 512 * 1024 * 1024,
                "estimated_complete_cache_peak_upper_bound_bytes": (
                    int(plan["cache_storage_estimate"]["selected_payload_bytes"])
                    + 576 * 1024 * 1024
                ),
                "estimated_complete_cache_peak_upper_bound_GiB": (
                    int(plan["cache_storage_estimate"]["selected_payload_bytes"])
                    + 576 * 1024 * 1024
                )
                / 1073741824.0,
            },
            "wall_seconds": time.perf_counter() - began,
            "source_dev_opened": not isolated_source_train,
            "source_eval_opened": False,
            "receipt_sha256": _PENDING,
        }
    )
    _write_json_atomic(cache / "cache_inventory.json", inventory, replace=True)
    return inventory


def _load_complete_cache(
    cache_dir: str | Path,
    *,
    verify_payload_hashes: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, tuple[Path, Path]]]:
    cache = Path(cache_dir).resolve(strict=True)
    if cache.is_symlink() or not cache.is_dir():
        raise PermissionError("native18 training cache must be a regular directory")
    plan = _load_json_content_address(
        cache / "selected_tile_plan.json", context="native18 selected tile plan"
    )
    contract = _load_json_content_address(
        cache / "cache_contract.json", context="native18 cache contract"
    )
    inventory = _load_json_content_address(
        cache / "cache_inventory.json", context="native18 cache inventory"
    )
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("provider_id") != PROVIDER_ID
        or plan.get("complete_source_train_roster") is not True
        or contract.get("schema_version") != CACHE_CONTRACT_SCHEMA_VERSION
        or contract.get("plan_receipt_sha256") != plan["receipt_sha256"]
        or inventory.get("schema_version") != CACHE_INVENTORY_SCHEMA_VERSION
        or inventory.get("provider_id") != PROVIDER_ID
        or inventory.get("plan_receipt_sha256") != plan["receipt_sha256"]
        or inventory.get("cache_contract_receipt_sha256")
        != contract["receipt_sha256"]
        or inventory.get("cache_complete") is not True
        or inventory.get("complete_source_train_roster") is not True
        or inventory.get("valid_cached_tile_count")
        != inventory.get("selected_tile_count")
        or inventory.get("selected_tile_count") != len(plan["selected_tiles"])
        or inventory.get("source_eval_opened") is not False
    ):
        raise PermissionError("native18 cache is incomplete or lineage-ineligible")
    inventory_by_id = {
        str(row["tile_id"]): dict(row) for row in inventory["tile_inventory"]
    }
    if len(inventory_by_id) != len(plan["selected_tiles"]):
        raise ValueError("native18 cache inventory tile IDs are incomplete")
    paths: dict[str, tuple[Path, Path]] = {}
    for row in plan["selected_tiles"]:
        tile_id = str(row["tile_id"])
        if tile_id not in inventory_by_id:
            raise ValueError("native18 selected tile is absent from cache inventory")
        receipt = _validate_cached_tile(
            row,
            cache_root=cache,
            plan=plan,
            contract=contract,
            verify_payload_hashes=verify_payload_hashes,
        )
        expected = inventory_by_id[tile_id]
        if receipt is None or (
            receipt.get("receipt_sha256")
            != expected.get("tile_receipt_sha256")
            or receipt.get("signal_npy_sha256")
            != expected.get("signal_npy_sha256")
            or receipt.get("target_npy_sha256")
            != expected.get("target_npy_sha256")
        ):
            raise ValueError(f"native18 cached tile {tile_id} failed replay")
        signal_path, target_path, _ = _tile_paths(cache, tile_id)
        paths[tile_id] = (signal_path, target_path)
    return cache, plan, contract, inventory, paths


def _training_config(
    *,
    plan: Mapping[str, Any],
    inventory: Mapping[str, Any],
    microbatch_size: int,
    device: torch.device,
    precision: str,
    model_seed: int,
    checkpoint_every_batches: int,
    monitoring_epoch_numbers: Sequence[int],
) -> dict[str, Any]:
    return {
        "profile_id": PROVIDER_ID,
        "fixed_epoch_count": FIXED_EPOCH_COUNT,
        "plan_receipt_sha256": plan["receipt_sha256"],
        "cache_inventory_receipt_sha256": inventory["receipt_sha256"],
        "manifest_receipt_sha256": plan["manifest_receipt_sha256"],
        "selected_tile_count": plan["selected_window_count"],
        "model_seed": model_seed,
        "official_logical_batch_size": 86,
        "microbatch_size": microbatch_size,
        "gradient_accumulation_used": microbatch_size < 86,
        "microbatch_BatchNorm1d_statistics_equal_official_batch86": (
            microbatch_size == 86
        ),
        "device": str(device),
        "precision": precision,
        "optimizer": "torch.optim.RAdam",
        "learning_rate": 1e-4,
        "weight_decay": 2e-5,
        "loss": "unweighted_dense_binary_cross_entropy",
        "gradient_clipping": None,
        "epoch_permutation": (
            "numpy_PCG64_seeded_by_sha256(model_seed,plan_sha,epoch_index)"
        ),
        "checkpoint_every_batches": checkpoint_every_batches,
        "coarse_monitoring_epoch_numbers": list(monitoring_epoch_numbers),
        "source_dev_epoch_selection": False,
        "source_eval_opened": False,
    }


def _epoch_permutation(
    tile_count: int, *, seed: int, plan_receipt_sha256: str, epoch_index: int
) -> np.ndarray:
    material = (
        f"{seed}|{plan_receipt_sha256}|fixed-cache-epoch|{epoch_index}"
    ).encode("utf-8")
    epoch_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = np.random.default_rng(epoch_seed)
    permutation = np.arange(tile_count, dtype=np.int64)
    generator.shuffle(permutation)
    return permutation


def _validate_training_arguments(
    *,
    microbatch_size: int,
    device: torch.device,
    precision: str,
    model_seed: int,
    maximum_steps: int | None,
    checkpoint_every_batches: int,
    monitoring_epoch_numbers: Sequence[int],
) -> None:
    if (
        isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, int)
        or microbatch_size < 1
        or microbatch_size > 86
    ):
        raise ValueError("native18 microbatch_size must be an explicit integer in [1,86]")
    if not isinstance(model_seed, int) or isinstance(model_seed, bool) or model_seed < 0:
        raise ValueError("model seed must be a nonnegative integer")
    if maximum_steps is not None and maximum_steps < 1:
        raise ValueError("maximum_steps must be positive")
    if checkpoint_every_batches < 1:
        raise ValueError("checkpoint_every_batches must be positive")
    normalized_monitoring = tuple(int(value) for value in monitoring_epoch_numbers)
    if (
        not normalized_monitoring
        or normalized_monitoring != tuple(sorted(set(normalized_monitoring)))
        or any(not 1 <= value <= FIXED_EPOCH_COUNT for value in normalized_monitoring)
        or FIXED_EPOCH_COUNT not in normalized_monitoring
    ):
        raise ValueError(
            "monitoring epochs must be sorted unique values in [1,100] including 100"
        )
    if precision not in {"fp32", "cuda_bf16"}:
        raise ValueError("precision must be fp32 or cuda_bf16")
    if precision == "cuda_bf16" and device.type != "cuda":
        raise ValueError("cuda_bf16 precision requires a CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested native18 CUDA device is unavailable")
        if precision == "cuda_bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("requested native18 CUDA BF16 is unsupported")


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def train_tusz_only_native18(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    microbatch_size: int,
    device_name: str,
    precision: str,
    model_seed: int = DEFAULT_MODEL_SEED,
    maximum_steps: int | None = None,
    resume: bool = False,
    checkpoint_every_batches: int = 25,
    verify_cache_payload_hashes: bool = False,
    monitoring_epoch_numbers: Sequence[int] = (20, 40, 60, 80, 100),
) -> dict[str, Any]:
    """Train scratch native18 for a fixed 100 epochs with exact batch resume."""

    device = torch.device(device_name)
    _validate_training_arguments(
        microbatch_size=microbatch_size,
        device=device,
        precision=precision,
        model_seed=model_seed,
        maximum_steps=maximum_steps,
        checkpoint_every_batches=checkpoint_every_batches,
        monitoring_epoch_numbers=monitoring_epoch_numbers,
    )
    cache, plan, _, inventory, tile_paths = _load_complete_cache(
        cache_dir, verify_payload_hashes=verify_cache_payload_hashes
    )
    output = Path(output_dir).resolve(strict=False)
    if output.is_symlink():
        raise PermissionError("native18 training output may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    primary_path = output / "epoch_0100_primary.pt"

    random.seed(model_seed)
    np.random.seed(model_seed % (2**32))
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        free_before, total_memory = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        free_before = total_memory = None
    model = native18.build_upstream_native18_model(seed=model_seed, device=str(device))
    optimizer = native18.build_upstream_native18_optimizer(model)
    config = _training_config(
        plan=plan,
        inventory=inventory,
        microbatch_size=microbatch_size,
        device=device,
        precision=precision,
        model_seed=model_seed,
        checkpoint_every_batches=checkpoint_every_batches,
        monitoring_epoch_numbers=monitoring_epoch_numbers,
    )
    selected = [dict(row) for row in plan["selected_tiles"]]
    tile_count = len(selected)
    official_logical_batch_size = 86
    batch_count = math.ceil(tile_count / official_logical_batch_size)
    next_epoch = 0
    next_batch = 0
    global_step = 0
    completed_history: list[dict[str, Any]] = []
    current_loss_sum = 0.0
    current_loss_count = 0
    finalize_only_resume = False
    completed_last_payload: Mapping[str, Any] | None = None

    if (last_path.exists() or last_path.is_symlink()) and not resume:
        raise FileExistsError("existing native18 last.pt requires resume=True")
    if resume:
        if last_path.is_symlink() or not last_path.is_file():
            raise FileNotFoundError("resume=True requires a regular last.pt")
        loaded = torch.load(last_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(loaded, Mapping)
            or loaded.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or loaded.get("provider_id") != PROVIDER_ID
            or loaded.get("training_config") != config
            or loaded.get("cache_inventory_receipt_sha256")
            != inventory["receipt_sha256"]
            or loaded.get("source_dev_opened") is not False
            or loaded.get("source_eval_opened") is not False
        ):
            raise PermissionError("native18 resume checkpoint lineage drifted")
        next_epoch = int(loaded["next_epoch"])
        next_batch = int(loaded["next_batch"])
        global_step = int(loaded["global_step"])
        completed_history = [dict(row) for row in loaded["completed_epoch_history"]]
        current_loss_sum = float(loaded["current_epoch_loss_sum"])
        current_loss_count = int(loaded["current_epoch_loss_count"])
        if loaded.get("training_complete") is True:
            if (
                loaded.get("checkpoint_role")
                != "epoch100_primary_inference_eligible"
                or loaded.get("inference_eligible") is not True
                or loaded.get("fixed_epoch_count") != FIXED_EPOCH_COUNT
                or loaded.get("completed_epoch_count") != FIXED_EPOCH_COUNT
                or next_epoch != FIXED_EPOCH_COUNT
                or next_batch != 0
                or len(completed_history) != FIXED_EPOCH_COUNT
                or current_loss_sum != 0.0
                or current_loss_count != 0
            ):
                raise PermissionError("completed native18 last.pt finalize cursor drifted")
            finalize_only_resume = True
            completed_last_payload = loaded
        else:
            if (
                not 0 <= next_epoch < FIXED_EPOCH_COUNT
                or not 0 <= next_batch < batch_count
            ):
                raise ValueError(
                    "native18 resume cursor is outside the fixed training plan"
                )
            if current_loss_count != next_batch:
                raise PermissionError("native18 checkpoint loss accumulator drifted")
            model.load_state_dict(loaded["model_state"], strict=True)
            optimizer.load_state_dict(loaded["optimizer_state"])
            _move_optimizer_state(optimizer, device)
            torch.set_rng_state(loaded["torch_cpu_rng_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(loaded["torch_cuda_rng_state_all"])
        print(
            json.dumps(
                {
                    "stage": (
                        "native18_complete_checkpoint_finalize_admitted"
                        if finalize_only_resume
                        else "native18_resume_admitted"
                    ),
                    "next_epoch": next_epoch,
                    "next_batch": next_batch,
                    "global_step": global_step,
                    "checkpoint_sha256": _file_sha256(last_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def checkpoint_payload(*, training_complete: bool) -> dict[str, Any]:
        role = (
            "epoch100_primary_inference_eligible"
            if training_complete
            else "resume_only_not_inference_eligible"
        )
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "checkpoint_role": role,
            "training_complete": training_complete,
            "inference_eligible": training_complete,
            "fixed_epoch_count": FIXED_EPOCH_COUNT,
            "completed_epoch_count": len(completed_history),
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "current_epoch_loss_sum": current_loss_sum,
            "current_epoch_loss_count": current_loss_count,
            "completed_epoch_history": completed_history,
            "training_config": config,
            "manifest_receipt_sha256": plan["manifest_receipt_sha256"],
            "selected_tile_plan_receipt_sha256": plan["receipt_sha256"],
            "cache_inventory_receipt_sha256": inventory["receipt_sha256"],
            "cache_directory": str(cache),
            "model_axis_order": list(native18.UPSTREAM_BIPOLAR_DBANANA_18),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            ),
            "epoch_permutation_rng_is_content_derived": True,
            "source_dev_opened": False,
            "source_eval_opened": False,
            "pickle_container_requires_weights_only_true": True,
        }

    def monitoring_payload(*, completed_epoch_number: int) -> dict[str, Any]:
        return {
            "schema_version": MONITORING_CHECKPOINT_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "checkpoint_role": "coarse_monitoring_weights_not_primary",
            "completed_epoch_count": completed_epoch_number,
            "fixed_primary_epoch_count": FIXED_EPOCH_COUNT,
            "may_replace_epoch100_primary": False,
            "source_dev_metric_at_save_time": None,
            "source_dev_coarse_sensitivity_review_requires_separate_posthoc_run": True,
            "selected_tile_plan_receipt_sha256": plan["receipt_sha256"],
            "cache_inventory_receipt_sha256": inventory["receipt_sha256"],
            "model_axis_order": list(native18.UPSTREAM_BIPOLAR_DBANANA_18),
            "model_state": model.state_dict(),
            "optimizer_state_persisted": False,
            "RNG_state_persisted": False,
            "source_dev_opened": False,
            "source_eval_opened": False,
            "pickle_container_requires_weights_only_true": True,
        }

    invocation_steps = 0
    began = time.perf_counter()
    stopped_early = False
    model.train()
    for epoch_index in range(next_epoch, FIXED_EPOCH_COUNT):
        permutation = _epoch_permutation(
            tile_count,
            seed=model_seed,
            plan_receipt_sha256=str(plan["receipt_sha256"]),
            epoch_index=epoch_index,
        )
        epoch_batch_start = next_batch if epoch_index == next_epoch else 0
        epoch_began = time.perf_counter()
        for batch_index in range(epoch_batch_start, batch_count):
            indices = permutation[
                batch_index * official_logical_batch_size : min(
                    (batch_index + 1) * official_logical_batch_size, tile_count
                )
            ]
            logical_batch_size = int(indices.size)
            optimizer.zero_grad()
            logical_loss_value = 0.0
            for micro_start in range(0, logical_batch_size, microbatch_size):
                micro_indices = indices[
                    micro_start : min(
                        micro_start + microbatch_size, logical_batch_size
                    )
                ]
                signals: list[np.ndarray] = []
                targets: list[np.ndarray] = []
                for index in micro_indices.tolist():
                    row = selected[int(index)]
                    signal_path, target_path = tile_paths[str(row["tile_id"])]
                    signals.append(
                        np.asarray(
                            np.load(signal_path, mmap_mode="r", allow_pickle=False),
                            dtype=np.float32,
                        )
                    )
                    targets.append(
                        np.asarray(
                            np.load(target_path, mmap_mode="r", allow_pickle=False),
                            dtype=np.float32,
                        )
                    )
                values = torch.from_numpy(np.stack(signals)).to(device=device)
                target = torch.from_numpy(np.stack(targets)).to(device=device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "cuda_bf16",
                ):
                    probability = model(values)
                probability = probability.float()
                micro_loss = torch.nn.functional.binary_cross_entropy(
                    probability, target
                )
                if not bool(torch.isfinite(micro_loss)):
                    raise native18.Native18NumericalFailure(
                        "native18 fixed100 BCE became non-finite"
                    )
                micro_weight = len(micro_indices) / logical_batch_size
                (micro_loss * micro_weight).backward()
                logical_loss_value += float(micro_loss.detach().cpu()) * micro_weight
            # The released trainer has no gradient clipping.
            optimizer.step()
            loss_value = logical_loss_value
            global_step += 1
            invocation_steps += 1
            current_loss_sum += loss_value
            current_loss_count += 1
            next_epoch = epoch_index
            next_batch = batch_index + 1
            batch_finished_epoch = next_batch == batch_count
            if batch_finished_epoch:
                completed_history.append(
                    {
                        "epoch_index": epoch_index,
                        "epoch_number_one_based": epoch_index + 1,
                        "batch_count": batch_count,
                        "tile_count": tile_count,
                        "partial_final_batch_size": (
                            tile_count % official_logical_batch_size
                        ),
                        "mean_unweighted_dense_BCE": (
                            current_loss_sum / current_loss_count
                        ),
                        "source_dev_metric_or_epoch_selection": None,
                        "elapsed_seconds": time.perf_counter() - epoch_began,
                    }
                )
                next_epoch = epoch_index + 1
                next_batch = 0
                current_loss_sum = 0.0
                current_loss_count = 0
                completed_epoch_number = epoch_index + 1
                if (
                    completed_epoch_number in set(monitoring_epoch_numbers)
                    and completed_epoch_number != FIXED_EPOCH_COUNT
                ):
                    monitoring_path = output / (
                        f"epoch_{completed_epoch_number:04d}_monitoring_weights.pt"
                    )
                    _atomic_torch_save(
                        monitoring_path,
                        monitoring_payload(
                            completed_epoch_number=completed_epoch_number
                        ),
                    )
            should_save = (
                global_step % checkpoint_every_batches == 0
                or batch_finished_epoch
                or (
                    maximum_steps is not None
                    and invocation_steps >= maximum_steps
                )
            )
            training_complete = next_epoch == FIXED_EPOCH_COUNT and next_batch == 0
            if should_save:
                _atomic_torch_save(
                    last_path, checkpoint_payload(training_complete=training_complete)
                )
                print(
                    json.dumps(
                        {
                            "stage": "native18_train_checkpoint",
                            "epoch_number_one_based": min(next_epoch + 1, FIXED_EPOCH_COUNT),
                            "next_batch": next_batch,
                            "global_step": global_step,
                            "last_loss": loss_value,
                            "training_complete": training_complete,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if maximum_steps is not None and invocation_steps >= maximum_steps:
                stopped_early = not training_complete
                break
        if maximum_steps is not None and invocation_steps >= maximum_steps:
            break

    training_complete = next_epoch == FIXED_EPOCH_COUNT and next_batch == 0
    if training_complete:
        if finalize_only_resume:
            if completed_last_payload is None:
                raise AssertionError("completed native18 finalize payload is absent")
            _install_or_replay_completed_primary(primary_path, completed_last_payload)
        else:
            primary = checkpoint_payload(training_complete=True)
            _atomic_torch_save(primary_path, primary)
            _atomic_torch_save(last_path, primary)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        resource = {
            "device": str(device),
            "precision": precision,
            "free_memory_before_MiB": free_before / 1048576.0,
            "free_memory_after_MiB": free_after / 1048576.0,
            "total_memory_MiB": total_memory / 1048576.0,
            "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1048576.0,
            "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 1048576.0,
        }
    else:
        resource = {
            "device": str(device),
            "precision": "fp32",
            "free_memory_before_MiB": None,
            "free_memory_after_MiB": None,
            "total_memory_MiB": None,
            "peak_allocated_MiB": None,
            "peak_reserved_MiB": None,
        }
    receipt_path = output / "training_receipt.json"
    if finalize_only_resume and (receipt_path.exists() or receipt_path.is_symlink()):
        observed_receipt = _load_json_content_address(
            receipt_path, context="native18 completed training receipt"
        )
        if (
            observed_receipt.get("schema_version") != SCHEMA_VERSION
            or observed_receipt.get("provider_id") != PROVIDER_ID
            or observed_receipt.get("status") != "completed_epoch100_primary"
            or observed_receipt.get("training_complete") is not True
            or observed_receipt.get("next_epoch") != FIXED_EPOCH_COUNT
            or observed_receipt.get("next_batch") != 0
            or observed_receipt.get("last_checkpoint_path") != str(last_path)
            or observed_receipt.get("last_checkpoint_sha256")
            != _file_sha256(last_path)
            or observed_receipt.get("primary_checkpoint_path") != str(primary_path)
            or observed_receipt.get("primary_checkpoint_sha256")
            != _file_sha256(primary_path)
            or observed_receipt.get("source_dev_opened") is not False
            or observed_receipt.get("source_eval_opened") is not False
        ):
            raise PermissionError("completed native18 training receipt drifted")
        return observed_receipt
    receipt = _content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "fixed100_training_invocation",
            "provider_id": PROVIDER_ID,
            "status": (
                "completed_epoch100_primary"
                if training_complete
                else "step_limited_resume_checkpoint"
            ),
            "fixed_epoch_count": FIXED_EPOCH_COUNT,
            "training_complete": training_complete,
            "stopped_early": stopped_early,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "invocation_steps": invocation_steps,
            "maximum_steps": maximum_steps,
            "last_checkpoint_path": str(last_path),
            "last_checkpoint_sha256": _file_sha256(last_path),
            "primary_checkpoint_path": str(primary_path) if training_complete else None,
            "primary_checkpoint_sha256": (
                _file_sha256(primary_path) if training_complete else None
            ),
            "only_epoch100_checkpoint_is_primary": True,
            "coarse_monitoring_checkpoints": [
                {
                    "epoch_number": epoch_number,
                    "path": str(
                        primary_path
                        if epoch_number == FIXED_EPOCH_COUNT
                        else output
                        / f"epoch_{epoch_number:04d}_monitoring_weights.pt"
                    ),
                    "exists": (
                        primary_path.is_file()
                        if epoch_number == FIXED_EPOCH_COUNT
                        else (
                            output
                            / f"epoch_{epoch_number:04d}_monitoring_weights.pt"
                        ).is_file()
                    ),
                    "role": (
                        "epoch100_primary"
                        if epoch_number == FIXED_EPOCH_COUNT
                        else "posthoc_coarse_sensitivity_only_not_primary"
                    ),
                }
                for epoch_number in monitoring_epoch_numbers
            ],
            "monitoring_weights_saved_without_optimizer_or_RNG": True,
            "source_dev_used_to_select_epoch": False,
            "source_dev_opened": False,
            "source_eval_opened": False,
            "official_contract": {
                "batch_size": 86,
                "precision": "float32",
                "epochs": 100,
            },
            "executed_resource_adaptation": {
                "logical_batch_size": 86,
                "microbatch_size": microbatch_size,
                "gradient_accumulation_steps_for_full_logical_batch": (
                    math.ceil(86 / microbatch_size)
                ),
                "precision": precision,
                "optimizer_update_batch_cardinality_differs_from_official": False,
                "microbatch_forward_cardinality_differs_from_official": (
                    microbatch_size != 86
                ),
                "model_contains_BatchNorm1d": True,
                "microbatch_BatchNorm_running_statistics_are_not_equivalent_to_batch86": (
                    microbatch_size != 86
                ),
                "dropout_RNG_trajectory_may_differ_from_single_batch86_forward": (
                    microbatch_size != 86
                ),
                "precision_differs_from_official": precision != "fp32",
                "vLLM_service_stopped_by_runner": False,
            },
            "resource_receipt": resource,
            "wall_seconds": time.perf_counter() - began,
            "clinical_use_authorized": False,
            "receipt_sha256": _PENDING,
        }
    )
    _write_json_atomic(receipt_path, receipt, replace=True)
    return receipt


def load_epoch100_native18_checkpoint(
    checkpoint_path: str | Path, *, device_name: str
) -> tuple[nn.Module, dict[str, Any], str]:
    source = Path(checkpoint_path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError("native18 checkpoint must be a regular file")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("provider_id") != PROVIDER_ID
        or checkpoint.get("checkpoint_role")
        != "epoch100_primary_inference_eligible"
        or checkpoint.get("training_complete") is not True
        or checkpoint.get("inference_eligible") is not True
        or checkpoint.get("fixed_epoch_count") != FIXED_EPOCH_COUNT
        or checkpoint.get("completed_epoch_count") != FIXED_EPOCH_COUNT
        or checkpoint.get("next_epoch") != FIXED_EPOCH_COUNT
        or checkpoint.get("next_batch") != 0
        or checkpoint.get("source_dev_opened") is not False
        or checkpoint.get("source_eval_opened") is not False
        or checkpoint.get("model_axis_order")
        != list(native18.UPSTREAM_BIPOLAR_DBANANA_18)
    ):
        raise PermissionError("checkpoint is not the completed epoch100 primary")
    device = torch.device(device_name)
    model = native18.build_upstream_native18_model(
        seed=int(checkpoint["training_config"]["model_seed"]), device=str(device)
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, dict(checkpoint), _file_sha256(source)


def build_target_free_source_dev_roster(
    *,
    analysis_projection_path: str | Path,
    physical_audit_path: str | Path,
) -> dict[str, Any]:
    """Bind the canonical 1,821-row source-dev roster without TERM targets."""

    projection_source = Path(analysis_projection_path).resolve(strict=True)
    audit_source = Path(physical_audit_path).resolve(strict=True)
    projection_raw = json.loads(projection_source.read_text(encoding="utf-8"))
    audit_raw = json.loads(audit_source.read_text(encoding="utf-8"))
    audit = validate_tusz_canonical_physical_duplicate_audit_v1(audit_raw)
    # Both artifacts are self-validating here.  The validator's deeper
    # ``audit=`` branch also requires the historical complete roster and
    # pre-physical projection, which are not accepted by this target-free
    # runner.  Their direct receipt/ID binding is replayed immediately below.
    projection = validate_tusz_canonical_physical_analysis_projection_v1(
        projection_raw
    )
    if (
        projection["source_binding"][
            "source_canonical_physical_audit_receipt_sha256"
        ]
        != audit["receipt_sha256"]
        or projection["source_binding"]["source_canonical_physical_audit_id"]
        != audit["audit_id"]
        or projection["reference_access_receipt"]["reference_files_opened"] != 0
        or projection["reference_access_receipt"][
            "seizure_interval_or_label_values_read"
        ]
        is not False
    ):
        raise PermissionError("target-free physical projection authority drifted")
    outcome_by_identity = {
        str(row["analysis_identity_id"]): row for row in audit["outcomes"]
    }
    rows: list[dict[str, Any]] = []
    for projection_row in projection["records"]:
        if projection_row["model_split"] != "source_dev":
            continue
        if _FORBIDDEN_TARGET_FIELDS.intersection(projection_row):
            raise PermissionError("source-dev identity projection contains a target")
        identity = str(projection_row["analysis_identity_id"])
        outcome = outcome_by_identity.get(identity)
        if (
            outcome is None
            or outcome.get("terminal_status") != "success"
            or outcome.get("failure") is not None
            or outcome.get("model_split") != "source_dev"
            or outcome.get("local_edf_path") != projection_row["local_edf_path"]
            or outcome.get("local_patient_id")
            != projection_row["local_patient_id"]
        ):
            raise PermissionError("source-dev physical audit join drifted")
        physical = outcome["physical_signal"]
        duration_fraction = physical["duration_seconds_fraction"]
        rate_fraction = physical["sampling_rate_fraction"]
        if (
            not isinstance(duration_fraction, list)
            or len(duration_fraction) != 2
            or not isinstance(rate_fraction, list)
            or len(rate_fraction) != 2
        ):
            raise ValueError("source-dev physical clock is malformed")
        duration = Fraction(int(duration_fraction[0]), int(duration_fraction[1]))
        rate = Fraction(int(rate_fraction[0]), int(rate_fraction[1]))
        sample_count = int(physical["sample_count"])
        if duration <= 0 or rate <= 0 or Fraction(sample_count, 1) / rate != duration:
            raise ValueError("source-dev physical duration does not replay")
        target_sample_count = int(sample_count / float(rate) * native18.TARGET_FS_HZ)
        if (
            physical["canonical_source_tensor_sha256"]
            != projection_row["canonical_physical_source_tensor_sha256"]
        ):
            raise PermissionError("source-dev physical tensor binding drifted")
        rows.append(
            {
                "analysis_identity_id": identity,
                "local_edf_path": str(projection_row["local_edf_path"]),
                "local_patient_id": str(projection_row["local_patient_id"]),
                "model_split": "source_dev",
                "official_split": "dev",
                "duration_seconds_fraction": [
                    duration.numerator,
                    duration.denominator,
                ],
                "source_sampling_rate_fraction": [rate.numerator, rate.denominator],
                "source_sample_count": sample_count,
                "target_sample_count_256hz": target_sample_count,
                "canonical_source_tensor_sha256": str(
                    physical["canonical_source_tensor_sha256"]
                ),
                "canonical_physical_equivalence_id": str(
                    projection_row["canonical_physical_equivalence_id"]
                ),
            }
        )
    rows.sort(key=lambda row: row["analysis_identity_id"])
    if len(rows) != 1821:
        raise ValueError("target-free canonical source-dev roster must contain 1,821 records")
    if len({row["local_patient_id"] for row in rows}) != 53:
        raise ValueError("target-free canonical source-dev roster must contain 53 patients")
    if len({row["analysis_identity_id"] for row in rows}) != len(rows):
        raise ValueError("target-free source-dev identities are not unique")
    return _content_address(
        {
            "schema_version": ROSTER_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "split": "source_dev",
            "analysis_projection_path": str(projection_source),
            "analysis_projection_file_sha256": _file_sha256(projection_source),
            "analysis_projection_receipt_sha256": projection["receipt_sha256"],
            "physical_audit_path": str(audit_source),
            "physical_audit_file_sha256": _file_sha256(audit_source),
            "physical_audit_receipt_sha256": audit["receipt_sha256"],
            "record_count": len(rows),
            "patient_count": len({row["local_patient_id"] for row in rows}),
            "identity_roster_sha256": _canonical_sha256(
                [row["analysis_identity_id"] for row in rows]
            ),
            "duration_source": "target_free_canonical_physical_signal_audit",
            "reference_annotation_or_TERM_target_opened": False,
            "source_eval_opened": False,
            "records": rows,
            "receipt_sha256": _PENDING,
        }
    )


def _load_prediction_row_for_reuse(
    receipt_path: Path,
    *,
    roster_receipt_sha256: str,
    checkpoint_sha256: str,
    inference_batch_size: int,
) -> dict[str, Any] | None:
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None
    try:
        row = _load_json_content_address(
            receipt_path, context="native18 source-dev prediction row"
        )
        if (
            row.get("schema_version") != PREDICTION_ROW_SCHEMA_VERSION
            or row.get("provider_id") != PROVIDER_ID
            or row.get("target_free_roster_receipt_sha256")
            != roster_receipt_sha256
            or row.get("checkpoint_sha256") != checkpoint_sha256
            or row.get("inference_batch_size") != inference_batch_size
            or row.get("source_eval_opened") is not False
            or row.get("reference_annotation_or_target_opened") is not False
            or row.get("status")
            not in {"prediction_complete", "upstream_skip_below_60s"}
        ):
            return None
        if row["status"] == "prediction_complete":
            posterior = Path(str(row["posterior_path"])).resolve(strict=True)
            if (
                posterior.is_symlink()
                or _file_sha256(posterior) != row["posterior_npy_sha256"]
            ):
                return None
            values = np.load(posterior, mmap_mode="r", allow_pickle=False)
            valid = (
                values.shape == (int(row["target_sample_count_256hz"]),)
                and values.dtype == np.dtype("float32")
            )
            del values
            if not valid:
                return None
        return row
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _event_rows_from_decoded(
    decoded: native18.Native18DecodedEvents,
) -> list[dict[str, Any]]:
    return [
        {
            "start_sample": int(start),
            "stop_sample_exclusive": int(stop),
            "start_seconds": start / native18.TARGET_FS_HZ,
            "stop_seconds": stop / native18.TARGET_FS_HZ,
        }
        for start, stop in decoded.event_sample_spans
    ]


def infer_target_free_source_dev(
    *,
    checkpoint_path: str | Path,
    analysis_projection_path: str | Path,
    physical_audit_path: str | Path,
    tusz_root: str | Path,
    output_dir: str | Path,
    device_name: str,
    inference_batch_size: int,
    maximum_records: int | None = None,
) -> dict[str, Any]:
    """Infer source-dev without labels and freeze posterior plus native events."""

    if (
        isinstance(inference_batch_size, bool)
        or not isinstance(inference_batch_size, int)
        or inference_batch_size < 1
    ):
        raise ValueError("inference_batch_size must be a positive integer")
    if maximum_records is not None and maximum_records < 1:
        raise ValueError("maximum_records must be positive")
    output = Path(output_dir).resolve(strict=False)
    replay = _preflight_existing_complete_prediction_manifest(
        output=output,
        manifest_name="prediction_manifest.json",
        expected_schema_version=PREDICTION_MANIFEST_SCHEMA_VERSION,
        checkpoint_path=checkpoint_path,
        analysis_projection_path=analysis_projection_path,
        physical_audit_path=physical_audit_path,
        inference_batch_size=inference_batch_size,
        maximum_records=maximum_records,
        validator=_validate_frozen_prediction_manifest,
        context="native18",
    )
    if replay is not None:
        return replay
    roster = build_target_free_source_dev_roster(
        analysis_projection_path=analysis_projection_path,
        physical_audit_path=physical_audit_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    roster = _install_or_replay_json(
        output / "target_free_source_dev_roster.json",
        roster,
        context="native18 target-free source-dev roster",
    )
    model, checkpoint, checkpoint_sha = load_epoch100_native18_checkpoint(
        checkpoint_path, device_name=device_name
    )
    root = Path(tusz_root).resolve(strict=True)
    all_rows = [dict(row) for row in roster["records"]]
    rows = all_rows if maximum_records is None else all_rows[:maximum_records]
    result_rows: list[dict[str, Any]] = []
    began = time.perf_counter()
    for index, identity_row in enumerate(rows):
        identity = str(identity_row["analysis_identity_id"])
        record_dir = output / "records" / identity
        if record_dir.is_symlink():
            raise PermissionError("native18 prediction record directory is a symlink")
        record_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = record_dir / "receipt.json"
        reused = _load_prediction_row_for_reuse(
            receipt_path,
            roster_receipt_sha256=str(roster["receipt_sha256"]),
            checkpoint_sha256=checkpoint_sha,
            inference_batch_size=inference_batch_size,
        )
        if reused is not None:
            result_rows.append(reused)
            continue
        base = {
            "schema_version": PREDICTION_ROW_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "analysis_identity_id": identity,
            "patient_id": identity_row["local_patient_id"],
            "edf_relative_path": identity_row["local_edf_path"],
            "model_split": "source_dev",
            "prediction_roster_index": index,
            "duration_seconds_fraction": identity_row["duration_seconds_fraction"],
            "target_sample_count_256hz": identity_row[
                "target_sample_count_256hz"
            ],
            "target_free_roster_receipt_sha256": roster["receipt_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "inference_batch_size": inference_batch_size,
            "released_threshold": native18.RELEASED_THRESHOLD,
            "reference_annotation_or_target_opened": False,
            "source_eval_opened": False,
        }
        record_began = time.perf_counter()
        if int(identity_row["target_sample_count_256hz"]) < native18.TILE_SAMPLES:
            row = {
                **base,
                "status": "upstream_skip_below_60s",
                "upstream_literal_primary_lane": True,
                "minimum_analyzable_samples": native18.TILE_SAMPLES,
                "EDF_opened": False,
                "posterior_state": "absent_by_released_lt60s_skip",
                "posterior_path": None,
                "posterior_npy_sha256": None,
                "predicted_events": [],
                "predicted_event_count": 0,
                "coverage": {
                    "intention_to_evaluate_record_retained": True,
                    "model_inference_observed_sample_count": 0,
                    "tail_padding_sample_count": 0,
                    "zero_alarm_assigned_for_scoring": True,
                    "coverage_padding_sensitivity_lane": False,
                },
                "wall_seconds": time.perf_counter() - record_began,
            }
        else:
            edf_path = _safe_edf(
                root,
                str(identity_row["local_edf_path"]),
                split="source_dev",
            )
            transformed = _transform_native18_edf(
                edf_path,
                expected_source_tensor_sha256=str(
                    identity_row["canonical_source_tensor_sha256"]
                ),
                expected_target_sample_count=int(
                    identity_row["target_sample_count_256hz"]
                ),
            )
            inference_began = time.perf_counter()
            result = native18.infer_upstream_native18_full_record(
                model,
                transformed,
                device=device_name,
                batch_size=inference_batch_size,
                threshold=native18.RELEASED_THRESHOLD,
            )
            inference_seconds = time.perf_counter() - inference_began
            posterior_path = record_dir / "dense_posterior.npy"
            _save_numpy_atomic(posterior_path, result.posterior, replace=True)
            event_rows = _event_rows_from_decoded(result.decoded)
            tile_ledger = result.receipt["tile_ledger"]
            row = {
                **base,
                "status": "prediction_complete",
                "upstream_literal_primary_lane": True,
                "EDF_opened": True,
                "posterior_state": "complete_dense_256hz_prethreshold",
                "posterior_path": str(posterior_path),
                "posterior_npy_sha256": _file_sha256(posterior_path),
                "posterior_dtype": "float32",
                "posterior_sample_count": len(result.posterior),
                "posterior_minimum": float(np.min(result.posterior)),
                "posterior_maximum": float(np.max(result.posterior)),
                "predicted_events": event_rows,
                "predicted_event_count": len(event_rows),
                "decoder_receipt": result.decoded.receipt,
                "inference_receipt": result.receipt,
                "native18_transform_receipt": transformed.receipt,
                "coverage": {
                    "intention_to_evaluate_record_retained": True,
                    "model_inference_observed_sample_count": len(result.posterior),
                    "complete_observed_support_coverage": True,
                    "inference_tile_count": len(tile_ledger),
                    "tail_padding_sample_count": sum(
                        int(tile["right_padding_sample_count"])
                        for tile in tile_ledger
                    ),
                    "tail_padding_applied_only_after_record_ge60s": True,
                    "tail_padding_trimmed_before_posterior_freeze": True,
                    "coverage_padding_sensitivity_lane": False,
                },
                "inference_seconds": inference_seconds,
                "wall_seconds": time.perf_counter() - record_began,
            }
        row = _content_address(row)
        _write_json_atomic(receipt_path, row, replace=True)
        result_rows.append(row)
        if (index + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "stage": "native18_source_dev_inference_progress",
                        "completed_record_count": index + 1,
                        "requested_record_count": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    status_counts = Counter(str(row["status"]) for row in result_rows)
    inventory_complete = maximum_records is None and len(result_rows) == len(all_rows)
    manifest = _content_address(
        {
            "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "claim_status": "source_dev_native18_post_epoch100",
            "split": "source_dev",
            "checkpoint_path": str(Path(checkpoint_path).resolve(strict=True)),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_completed_epoch_count": checkpoint["completed_epoch_count"],
            "target_free_roster_path": str(
                output / "target_free_source_dev_roster.json"
            ),
            "target_free_roster_receipt_sha256": roster["receipt_sha256"],
            "full_expected_record_count": len(all_rows),
            "materialized_record_count": len(result_rows),
            "maximum_records_smoke_limit": maximum_records,
            "complete_prediction_inventory": inventory_complete,
            "prediction_frozen_for_scoring": inventory_complete,
            "status_counts": dict(sorted(status_counts.items())),
            "intention_to_evaluate_denominator_preserved": len(result_rows)
            == len(rows),
            "below_60s_policy": "released_upstream_skip_zero_alarm_primary_lane",
            "coverage_padding_sensitivity_lane_executed": False,
            "inference_batch_size": inference_batch_size,
            "inference_precision": "float32",
            "posterior_and_released_events_persisted_before_reference_access": True,
            "reference_annotation_or_target_opened": False,
            "source_eval_opened": False,
            "prediction_rows": result_rows,
            "wall_seconds": time.perf_counter() - began,
            "clinical_use_authorized": False,
            "receipt_sha256": _PENDING,
        }
    )
    if inventory_complete:
        _prevalidate_native18_complete_manifest(output, manifest)
    return _write_native18_prediction_manifest_frozen(
        output / "prediction_manifest.json", manifest
    )


def _write_native18_prediction_manifest_frozen(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Allow partial progress until one complete native18 inventory is frozen."""

    if path.is_symlink():
        raise PermissionError("native18 prediction manifest may not be a symlink")
    candidate = _validate_content_address(
        manifest, context="native18 prediction manifest candidate"
    )
    if path.exists():
        observed = _load_json_content_address(
            path, context="native18 frozen prediction manifest"
        )
        if observed.get("complete_prediction_inventory") is True:
            if observed != candidate:
                raise PermissionError(
                    "a different or partial native18 inventory cannot replace the frozen inventory"
                )
            return observed
    _write_json_atomic(path, candidate, replace=True)
    return candidate


def _prevalidate_native18_complete_manifest(
    output: Path, manifest: Mapping[str, Any]
) -> None:
    if manifest.get("complete_prediction_inventory") is not True:
        raise ValueError("native18 pre-freeze validation requires full coverage")
    staging = output / ".prediction_manifest.prefreeze.json"
    if staging.is_symlink():
        raise PermissionError("native18 pre-freeze manifest may not be a symlink")
    _write_json_atomic(staging, manifest, replace=True)
    try:
        _validate_frozen_prediction_manifest(staging)
    finally:
        if staging.exists():
            staging.unlink()


def _transform_external19_edf(
    edf_path: Path,
    *,
    expected_source_tensor_sha256: str,
    expected_target_sample_count: int,
) -> native18.ExternalNative19Record:
    physical = load_canonical_edf_physical_source_identity(edf_path)
    if (
        physical.source_header_receipt["source_tensor_sha256"]
        != expected_source_tensor_sha256
    ):
        raise PermissionError("external19 canonical EDF tensor hash drifted")
    carrier = native18.materialize_upstream_literal_referential19(
        np.asarray(
            physical.observed_signal_volts.detach().cpu().numpy(),
            dtype=np.float32,
        ),
        physical.observed_channel_ids,
    )
    transformed = native18.transform_external_upstream_native19_record(
        carrier, source_sampling_rate_hz=_source_rate(physical)
    )
    if transformed.signal.shape != (19, expected_target_sample_count):
        raise RuntimeError("external19 transformed support differs from roster")
    return transformed


def _is_expected_external19_missing_axis_failure(error: BaseException) -> bool:
    """Whitelist only the literal 19-axis missing-electrode rejection.

    Resource exhaustion, I/O faults, numerical failures, hash drift, and code
    defects must terminate inference.  Converting those failures to zero-alarm
    rows could otherwise make an invalid inventory look complete before the
    post-freeze validator gets a chance to reject it.
    """

    return (
        isinstance(error, ValueError)
        and str(error) == "upstream get_data native19 requires all 19 electrodes"
    )


def _reuse_external19_row(
    receipt_path: Path,
    *,
    identity_row: Mapping[str, Any],
    prediction_roster_index: int,
    roster_receipt_sha256: str,
    checkpoint_sha256: str,
    inference_batch_size: int,
) -> dict[str, Any] | None:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        return None
    try:
        row = _load_json_content_address(
            receipt_path, context="external19 diagnostic prediction row"
        )
        expected_identity = str(identity_row["analysis_identity_id"])
        expected_sample_count = int(identity_row["target_sample_count_256hz"])
        if (
            row.get("schema_version") != EXTERNAL19_PREDICTION_ROW_SCHEMA_VERSION
            or row.get("profile_id") != native18.EXTERNAL_NATIVE19_PROFILE_ID
            or row.get("analysis_identity_id") != expected_identity
            or row.get("patient_id") != identity_row["local_patient_id"]
            or row.get("edf_relative_path") != identity_row["local_edf_path"]
            or row.get("model_split") != "source_dev"
            or row.get("prediction_roster_index") != prediction_roster_index
            or row.get("duration_seconds_fraction")
            != identity_row["duration_seconds_fraction"]
            or row.get("target_sample_count_256hz") != expected_sample_count
            or row.get("target_free_roster_receipt_sha256")
            != roster_receipt_sha256
            or row.get("checkpoint_sha256") != checkpoint_sha256
            or row.get("inference_batch_size") != inference_batch_size
            or row.get("released_threshold") != native18.RELEASED_THRESHOLD
            or row.get("paper_native18_reproduction") is not False
            or row.get("external_artifact_provenance_and_training_exposure_known")
            is not False
            or row.get("source_eval_opened") is not False
            or row.get("reference_annotation_or_target_opened") is not False
            or row.get("status")
            not in {
                "external19_prediction_complete",
                "external19_upstream_skip_below_60s",
                "external19_typed_technical_failure",
            }
        ):
            return None
        if row["status"] == "external19_prediction_complete":
            posterior = Path(str(row["posterior_path"])).resolve(strict=True)
            expected_posterior = (receipt_path.parent / "dense_posterior.npy").resolve(
                strict=True
            )
            if (
                expected_sample_count < native18.TILE_SAMPLES
                or posterior != expected_posterior
                or posterior.is_symlink()
                or _file_sha256(posterior) != row["posterior_npy_sha256"]
            ):
                return None
            posterior_array = np.load(posterior, mmap_mode="r", allow_pickle=False)
            geometry_matches = (
                posterior_array.shape == (expected_sample_count,)
                and posterior_array.dtype == np.dtype("float32")
                and row.get("posterior_sample_count") == expected_sample_count
            )
            del posterior_array
            if not geometry_matches:
                return None
        elif row["status"] == "external19_upstream_skip_below_60s":
            if (
                expected_sample_count >= native18.TILE_SAMPLES
                or row.get("EDF_opened") is not False
                or row.get("posterior_path") is not None
                or row.get("posterior_npy_sha256") is not None
                or row.get("predicted_events") != []
                or row.get("predicted_event_count") != 0
                or row.get("zero_alarm_assigned") is not True
            ):
                return None
        else:
            if (
                expected_sample_count < native18.TILE_SAMPLES
                or row.get("EDF_opened") is not True
                or row.get("failure_type") != "ValueError"
                or row.get("failure_message")
                != "upstream get_data native19 requires all 19 electrodes"
                or row.get("posterior_path") is not None
                or row.get("posterior_npy_sha256") is not None
                or row.get("predicted_events") != []
                or row.get("predicted_event_count") != 0
                or row.get("zero_alarm_assigned") is not True
            ):
                return None
        return row
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _write_external19_prediction_manifest_frozen(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Install a partial manifest or immutably replay a complete inventory.

    Partial smoke/progress manifests remain replaceable so the same output
    directory can be resumed to full coverage.  Once a complete inventory is
    present, however, any byte-semantic difference is rejected rather than
    silently replacing predictions that may already have entered scoring.
    """

    if path.is_symlink():
        raise PermissionError("external19 prediction manifest may not be a symlink")
    candidate = _validate_content_address(
        manifest, context="external19 diagnostic prediction manifest candidate"
    )
    if path.exists():
        observed = _load_json_content_address(
            path, context="external19 frozen diagnostic prediction manifest"
        )
        if observed.get("complete_prediction_inventory") is True:
            if observed != candidate:
                raise PermissionError(
                    "a different complete external19 inventory is already frozen"
                )
            return observed
    _write_json_atomic(path, candidate, replace=True)
    return candidate


def _prevalidate_external19_complete_manifest(
    output: Path, manifest: Mapping[str, Any]
) -> None:
    """Replay the complete target-free inventory before immutable install."""

    if manifest.get("complete_prediction_inventory") is not True:
        raise ValueError("external19 pre-freeze validation requires full coverage")
    staging = output / ".external19_prediction_manifest.prefreeze.json"
    if staging.is_symlink():
        raise PermissionError("external19 pre-freeze manifest may not be a symlink")
    _write_json_atomic(staging, manifest, replace=True)
    try:
        _validate_external19_frozen_prediction_manifest(staging)
    finally:
        if staging.exists():
            staging.unlink()


def _preflight_existing_complete_prediction_manifest(
    *,
    output: Path,
    manifest_name: str,
    expected_schema_version: str,
    checkpoint_path: str | Path,
    analysis_projection_path: str | Path,
    physical_audit_path: str | Path,
    inference_batch_size: int,
    maximum_records: int | None,
    validator: Any,
    context: str,
) -> dict[str, Any] | None:
    """Replay a frozen inventory before any inference-side mutation.

    A late immutable-manifest check is insufficient: a changed batch size or
    checkpoint can replace row receipts and posterior arrays before the final
    manifest write is rejected.  This gate therefore runs before roster
    installation, model/device loading, record-directory creation, or row
    reuse.  Exact request lineage gets a full validator replay and an
    idempotent return; incompatible lineage fails without touching output.
    """

    if output.is_symlink():
        raise PermissionError(f"{context} inference output may not be a symlink")
    manifest_path = output / manifest_name
    if manifest_path.is_symlink():
        raise PermissionError(f"{context} prediction manifest may not be a symlink")
    if not manifest_path.exists():
        return None
    probe = _load_json_content_address(
        manifest_path, context=f"{context} existing prediction manifest preflight"
    )
    if probe.get("complete_prediction_inventory") is not True:
        return None
    requested_checkpoint = Path(checkpoint_path).resolve(strict=True)
    try:
        frozen_checkpoint = Path(str(probe["checkpoint_path"])).resolve(strict=True)
    except (KeyError, TypeError, OSError) as error:
        raise PermissionError(
            f"{context} frozen prediction checkpoint binding is malformed"
        ) from error
    if (
        probe.get("schema_version") != expected_schema_version
        or maximum_records is not None
        or probe.get("maximum_records_smoke_limit") is not None
        or probe.get("inference_batch_size") != inference_batch_size
        or frozen_checkpoint != requested_checkpoint
    ):
        raise PermissionError(
            f"{context} request lineage differs from the frozen prediction inventory"
        )
    frozen, roster, _ = validator(manifest_path)
    requested_projection = Path(analysis_projection_path).resolve(strict=True)
    requested_audit = Path(physical_audit_path).resolve(strict=True)
    expected_roster_path = (output / "target_free_source_dev_roster.json").resolve(
        strict=True
    )
    try:
        frozen_roster_path = Path(str(frozen["target_free_roster_path"])).resolve(
            strict=True
        )
        frozen_projection = Path(str(roster["analysis_projection_path"])).resolve(
            strict=True
        )
        frozen_audit = Path(str(roster["physical_audit_path"])).resolve(strict=True)
    except (KeyError, TypeError, OSError) as error:
        raise PermissionError(
            f"{context} frozen target-free roster binding is malformed"
        ) from error
    if (
        frozen_roster_path != expected_roster_path
        or frozen_projection != requested_projection
        or frozen_audit != requested_audit
        or roster.get("analysis_projection_file_sha256")
        != _file_sha256(requested_projection)
        or roster.get("physical_audit_file_sha256") != _file_sha256(requested_audit)
    ):
        raise PermissionError(
            f"{context} request inputs differ from the frozen prediction inventory"
        )
    return frozen


def infer_external19_target_free_source_dev_diagnostic(
    *,
    checkpoint_path: str | Path,
    analysis_projection_path: str | Path,
    physical_audit_path: str | Path,
    tusz_root: str | Path,
    output_dir: str | Path,
    device_name: str,
    inference_batch_size: int,
    maximum_records: int | None = None,
) -> dict[str, Any]:
    """Optional pinned external19 full-dev diagnostic; never a native18 claim."""

    if inference_batch_size < 1 or isinstance(inference_batch_size, bool):
        raise ValueError("external19 inference batch size must be positive")
    if maximum_records is not None and maximum_records < 1:
        raise ValueError("external19 maximum_records must be positive")
    output = Path(output_dir).resolve(strict=False)
    replay = _preflight_existing_complete_prediction_manifest(
        output=output,
        manifest_name="external19_prediction_manifest.json",
        expected_schema_version=EXTERNAL19_PREDICTION_MANIFEST_SCHEMA_VERSION,
        checkpoint_path=checkpoint_path,
        analysis_projection_path=analysis_projection_path,
        physical_audit_path=physical_audit_path,
        inference_batch_size=inference_batch_size,
        maximum_records=maximum_records,
        validator=_validate_external19_frozen_prediction_manifest,
        context="external19",
    )
    if replay is not None:
        return replay
    roster = build_target_free_source_dev_roster(
        analysis_projection_path=analysis_projection_path,
        physical_audit_path=physical_audit_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    roster = _install_or_replay_json(
        output / "target_free_source_dev_roster.json",
        roster,
        context="external19 target-free source-dev roster",
    )
    model, model_receipt = native18.load_external_native19_model(
        checkpoint_path, device=device_name
    )
    checkpoint_sha = str(model_receipt["checkpoint_sha256"])
    root = Path(tusz_root).resolve(strict=True)
    all_rows = [dict(row) for row in roster["records"]]
    rows = all_rows if maximum_records is None else all_rows[:maximum_records]
    result_rows: list[dict[str, Any]] = []
    began = time.perf_counter()
    for index, identity_row in enumerate(rows):
        identity = str(identity_row["analysis_identity_id"])
        record_dir = output / "records" / identity
        record_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = record_dir / "receipt.json"
        reused = _reuse_external19_row(
            receipt_path,
            identity_row=identity_row,
            prediction_roster_index=index,
            roster_receipt_sha256=str(roster["receipt_sha256"]),
            checkpoint_sha256=checkpoint_sha,
            inference_batch_size=inference_batch_size,
        )
        if reused is not None:
            result_rows.append(reused)
            continue
        base = {
            "schema_version": EXTERNAL19_PREDICTION_ROW_SCHEMA_VERSION,
            "profile_id": native18.EXTERNAL_NATIVE19_PROFILE_ID,
            "analysis_identity_id": identity,
            "patient_id": identity_row["local_patient_id"],
            "edf_relative_path": identity_row["local_edf_path"],
            "model_split": "source_dev",
            "prediction_roster_index": index,
            "duration_seconds_fraction": identity_row["duration_seconds_fraction"],
            "target_sample_count_256hz": identity_row[
                "target_sample_count_256hz"
            ],
            "target_free_roster_receipt_sha256": roster["receipt_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "inference_batch_size": inference_batch_size,
            "released_threshold": native18.RELEASED_THRESHOLD,
            "paper_native18_reproduction": False,
            "external_artifact_provenance_and_training_exposure_known": False,
            "reference_annotation_or_target_opened": False,
            "source_eval_opened": False,
        }
        record_began = time.perf_counter()
        if int(identity_row["target_sample_count_256hz"]) < native18.TILE_SAMPLES:
            row = {
                **base,
                "status": "external19_upstream_skip_below_60s",
                "EDF_opened": False,
                "posterior_path": None,
                "posterior_npy_sha256": None,
                "predicted_events": [],
                "predicted_event_count": 0,
                "zero_alarm_assigned": True,
                "wall_seconds": time.perf_counter() - record_began,
            }
        else:
            try:
                transformed = _transform_external19_edf(
                    _safe_edf(
                        root,
                        str(identity_row["local_edf_path"]),
                        split="source_dev",
                    ),
                    expected_source_tensor_sha256=str(
                        identity_row["canonical_source_tensor_sha256"]
                    ),
                    expected_target_sample_count=int(
                        identity_row["target_sample_count_256hz"]
                    ),
                )
                result = native18.infer_external_upstream_native19_full_record(
                    model,
                    transformed,
                    device=device_name,
                    batch_size=inference_batch_size,
                    threshold=native18.RELEASED_THRESHOLD,
                )
                posterior_path = record_dir / "dense_posterior.npy"
                _save_numpy_atomic(posterior_path, result.posterior, replace=True)
                events = _event_rows_from_decoded(result.decoded)
                row = {
                    **base,
                    "status": "external19_prediction_complete",
                    "EDF_opened": True,
                    "posterior_path": str(posterior_path),
                    "posterior_npy_sha256": _file_sha256(posterior_path),
                    "posterior_sample_count": len(result.posterior),
                    "predicted_events": events,
                    "predicted_event_count": len(events),
                    "transform_receipt": transformed.receipt,
                    "inference_receipt": result.receipt,
                    "wall_seconds": time.perf_counter() - record_began,
                }
            except Exception as error:
                if not _is_expected_external19_missing_axis_failure(error):
                    raise
                row = {
                    **base,
                    "status": "external19_typed_technical_failure",
                    "EDF_opened": True,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "posterior_path": None,
                    "posterior_npy_sha256": None,
                    "predicted_events": [],
                    "predicted_event_count": 0,
                    "zero_alarm_assigned": True,
                    "wall_seconds": time.perf_counter() - record_began,
                }
        row = _content_address(row)
        _write_json_atomic(receipt_path, row, replace=True)
        result_rows.append(row)
        if (index + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "stage": "external19_source_dev_progress",
                        "completed_record_count": index + 1,
                        "requested_record_count": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    complete = maximum_records is None and len(result_rows) == len(all_rows)
    status_counts = Counter(str(row["status"]) for row in result_rows)
    manifest = _content_address(
        {
            "schema_version": EXTERNAL19_PREDICTION_MANIFEST_SCHEMA_VERSION,
            "profile_id": native18.EXTERNAL_NATIVE19_PROFILE_ID,
            "claim_status": "optional_external_artifact_axis_diagnostic_only",
            "split": "source_dev",
            "diagnostic_question": (
                "whether_prior_common17_axis_projection_is_a_major_transfer_loss_source"
            ),
            "checkpoint_path": str(Path(checkpoint_path).resolve(strict=True)),
            "checkpoint_sha256": checkpoint_sha,
            "model_load_receipt": model_receipt,
            "target_free_roster_path": str(
                output / "target_free_source_dev_roster.json"
            ),
            "target_free_roster_receipt_sha256": roster["receipt_sha256"],
            "full_expected_record_count": len(all_rows),
            "materialized_record_count": len(result_rows),
            "maximum_records_smoke_limit": maximum_records,
            "inference_batch_size": inference_batch_size,
            "complete_prediction_inventory": complete,
            "prediction_frozen_for_scoring": complete,
            "status_counts": dict(sorted(status_counts.items())),
            "intention_to_evaluate_denominator_preserved": len(result_rows)
            == len(rows),
            "below_60s_policy": "released_upstream_skip_zero_alarm_primary_lane",
            "released_threshold": native18.RELEASED_THRESHOLD,
            "paper_native18_reproduction": False,
            "artifact_uploader_is_upstream_author_verified": False,
            "artifact_original_checkpoint_hash_verified": False,
            "artifact_training_exposure_documented": False,
            "posterior_and_released_events_persisted_before_reference_access": True,
            "reference_annotation_or_target_opened": False,
            "source_eval_opened": False,
            "prediction_rows": result_rows,
            "wall_seconds": time.perf_counter() - began,
            "clinical_use_authorized": False,
            "receipt_sha256": _PENDING,
        }
    )
    if complete:
        _prevalidate_external19_complete_manifest(output, manifest)
    manifest = _write_external19_prediction_manifest_frozen(
        output / "external19_prediction_manifest.json", manifest
    )
    return manifest


def _validate_frozen_prediction_manifest(
    prediction_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    source = Path(prediction_manifest_path).resolve(strict=True)
    manifest = _load_json_content_address(
        source, context="native18 frozen prediction manifest"
    )
    if (
        manifest.get("schema_version") != PREDICTION_MANIFEST_SCHEMA_VERSION
        or manifest.get("provider_id") != PROVIDER_ID
        or manifest.get("split") != "source_dev"
        or manifest.get("complete_prediction_inventory") is not True
        or manifest.get("prediction_frozen_for_scoring") is not True
        or manifest.get("maximum_records_smoke_limit") is not None
        or manifest.get("materialized_record_count")
        != manifest.get("full_expected_record_count")
        or manifest.get("full_expected_record_count") != 1821
        or isinstance(manifest.get("inference_batch_size"), bool)
        or not isinstance(manifest.get("inference_batch_size"), int)
        or int(manifest["inference_batch_size"]) < 1
        or manifest.get("reference_annotation_or_target_opened") is not False
        or manifest.get("source_eval_opened") is not False
        or manifest.get("below_60s_policy")
        != "released_upstream_skip_zero_alarm_primary_lane"
        or manifest.get("coverage_padding_sensitivity_lane_executed") is not False
    ):
        raise PermissionError("native18 source-dev prediction inventory is not frozen")
    checkpoint_path = Path(str(manifest["checkpoint_path"])).resolve(strict=True)
    if (
        checkpoint_path.is_symlink()
        or _file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]
    ):
        raise ValueError("native18 epoch100 checkpoint failed byte replay")
    roster_path = Path(str(manifest["target_free_roster_path"])).resolve(strict=True)
    roster = _load_json_content_address(
        roster_path, context="native18 target-free source-dev roster"
    )
    if (
        roster.get("schema_version") != ROSTER_SCHEMA_VERSION
        or roster.get("provider_id") != PROVIDER_ID
        or roster.get("split") != "source_dev"
        or roster.get("reference_annotation_or_TERM_target_opened") is not False
        or roster.get("source_eval_opened") is not False
        or roster.get("receipt_sha256")
        != manifest["target_free_roster_receipt_sha256"]
        or roster.get("record_count") != manifest["full_expected_record_count"]
        or roster.get("record_count") != 1821
        or roster.get("patient_count") != 53
    ):
        raise PermissionError("native18 target-free roster binding drifted")
    prediction_rows = manifest.get("prediction_rows")
    roster_rows = roster.get("records")
    if (
        not isinstance(prediction_rows, list)
        or not isinstance(roster_rows, list)
        or len(prediction_rows) != len(roster_rows)
    ):
        raise ValueError("native18 prediction roster is incomplete")
    allowed_status = {
        "prediction_complete",
        "upstream_skip_below_60s",
    }
    replayed_statuses: Counter[str] = Counter()
    for index, (row, identity_row) in enumerate(zip(prediction_rows, roster_rows, strict=True)):
        if not isinstance(row, Mapping) or _FORBIDDEN_TARGET_FIELDS.intersection(row):
            raise PermissionError("native18 prediction row contains a target")
        checked = _validate_content_address(
            row, context=f"native18 prediction row {index}"
        )
        if (
            checked.get("schema_version") != PREDICTION_ROW_SCHEMA_VERSION
            or checked.get("provider_id") != PROVIDER_ID
            or checked.get("prediction_roster_index") != index
            or checked.get("analysis_identity_id")
            != identity_row["analysis_identity_id"]
            or checked.get("target_free_roster_receipt_sha256")
            != roster["receipt_sha256"]
            or checked.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
            or checked.get("inference_batch_size")
            != manifest["inference_batch_size"]
            or checked.get("reference_annotation_or_target_opened") is not False
            or checked.get("source_eval_opened") is not False
            or checked.get("status") not in allowed_status
            or checked.get("released_threshold") != native18.RELEASED_THRESHOLD
        ):
            raise PermissionError("native18 frozen prediction row lineage drifted")
        expected_samples = int(identity_row["target_sample_count_256hz"])
        if checked["status"] == "prediction_complete":
            if expected_samples < native18.TILE_SAMPLES:
                raise PermissionError("below-60s row bypassed released upstream skip")
            posterior_path = Path(str(checked["posterior_path"])).resolve(strict=True)
            if posterior_path.is_symlink() or _file_sha256(posterior_path) != checked[
                "posterior_npy_sha256"
            ]:
                raise ValueError("native18 frozen posterior bytes drifted")
            posterior = np.load(posterior_path, mmap_mode="r", allow_pickle=False)
            if posterior.shape != (expected_samples,) or posterior.dtype != np.dtype(
                "float32"
            ):
                raise ValueError("native18 frozen posterior geometry drifted")
            replay = native18.decode_upstream_native18_posterior(
                np.asarray(posterior), threshold=native18.RELEASED_THRESHOLD
            )
            del posterior
            expected_spans = [
                (int(event["start_sample"]), int(event["stop_sample_exclusive"]))
                for event in checked["predicted_events"]
            ]
            if list(replay.event_sample_spans) != expected_spans:
                raise ValueError("native18 frozen decoded events failed replay")
        elif checked["status"] == "upstream_skip_below_60s":
            if (
                expected_samples >= native18.TILE_SAMPLES
                or checked.get("EDF_opened") is not False
                or checked.get("posterior_path") is not None
                or checked.get("predicted_events") != []
            ):
                raise PermissionError("released below-60s skip row drifted")
        replayed_statuses[str(checked["status"])] += 1
    if (
        sum(replayed_statuses.values()) != 1821
        or replayed_statuses["upstream_skip_below_60s"] != 215
        or replayed_statuses["prediction_complete"] != 1606
        or dict(sorted(replayed_statuses.items())) != manifest.get("status_counts")
    ):
        raise PermissionError("native18 full-roster status contract drifted")
    return manifest, roster, _file_sha256(source)


def _validate_external19_frozen_prediction_manifest(
    prediction_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Replay a complete target-free external19 diagnostic inventory.

    The literal 19-axis ``get_data()`` path rejects records missing a required
    referential electrode.  Those typed failures and the released ``<60 s``
    skips are preserved for the full 1,821-record intention-to-assess lane;
    they are excluded only from the separately named public-get-data
    signal-side-evaluable lane.  The public ``eval_test.py`` actually calls
    ``get_data_18()``, so this 19-axis reconstruction is neither an upstream
    official lane nor a paper-native18 claim.
    """

    source = Path(prediction_manifest_path).resolve(strict=True)
    manifest = _load_json_content_address(
        source, context="external19 frozen prediction manifest"
    )
    if (
        manifest.get("schema_version")
        != EXTERNAL19_PREDICTION_MANIFEST_SCHEMA_VERSION
        or manifest.get("profile_id") != native18.EXTERNAL_NATIVE19_PROFILE_ID
        or manifest.get("split") != "source_dev"
        or manifest.get("claim_status")
        != "optional_external_artifact_axis_diagnostic_only"
        or manifest.get("complete_prediction_inventory") is not True
        or manifest.get("prediction_frozen_for_scoring") is not True
        or manifest.get("maximum_records_smoke_limit") is not None
        or manifest.get("full_expected_record_count") != 1821
        or manifest.get("materialized_record_count") != 1821
        or isinstance(manifest.get("inference_batch_size"), bool)
        or not isinstance(manifest.get("inference_batch_size"), int)
        or int(manifest["inference_batch_size"]) < 1
        or manifest.get("reference_annotation_or_target_opened") is not False
        or manifest.get("source_eval_opened") is not False
        or manifest.get("below_60s_policy")
        != "released_upstream_skip_zero_alarm_primary_lane"
        or manifest.get("released_threshold") != native18.RELEASED_THRESHOLD
        or manifest.get("paper_native18_reproduction") is not False
        or manifest.get("artifact_uploader_is_upstream_author_verified") is not False
        or manifest.get("artifact_original_checkpoint_hash_verified") is not False
        or manifest.get("artifact_training_exposure_documented") is not False
    ):
        raise PermissionError(
            "external19 source-dev diagnostic prediction inventory is not frozen"
        )
    checkpoint_path = Path(str(manifest["checkpoint_path"])).resolve(strict=True)
    if (
        checkpoint_path.is_symlink()
        or manifest.get("checkpoint_sha256")
        != native18.EXTERNAL_CHECKPOINT_SHA256
        or _file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]
    ):
        raise ValueError("external19 checkpoint failed pinned byte replay")
    roster_path = Path(str(manifest["target_free_roster_path"])).resolve(strict=True)
    if roster_path != (source.parent / "target_free_source_dev_roster.json").resolve(
        strict=True
    ):
        raise PermissionError("external19 target-free roster escaped inventory root")
    roster = _load_json_content_address(
        roster_path, context="external19 target-free source-dev roster"
    )
    if (
        roster.get("schema_version") != ROSTER_SCHEMA_VERSION
        or roster.get("provider_id") != PROVIDER_ID
        or roster.get("split") != "source_dev"
        or roster.get("record_count") != 1821
        or roster.get("reference_annotation_or_TERM_target_opened") is not False
        or roster.get("source_eval_opened") is not False
        or roster.get("receipt_sha256")
        != manifest.get("target_free_roster_receipt_sha256")
    ):
        raise PermissionError("external19 target-free roster binding drifted")
    prediction_rows = manifest.get("prediction_rows")
    roster_rows = roster.get("records")
    if (
        not isinstance(prediction_rows, list)
        or not isinstance(roster_rows, list)
        or len(prediction_rows) != 1821
        or len(roster_rows) != 1821
    ):
        raise ValueError("external19 prediction roster is incomplete")
    allowed_status = {
        "external19_prediction_complete",
        "external19_upstream_skip_below_60s",
        "external19_typed_technical_failure",
    }
    replayed_statuses: Counter[str] = Counter()
    for index, (row, identity_row) in enumerate(
        zip(prediction_rows, roster_rows, strict=True)
    ):
        if not isinstance(row, Mapping) or _FORBIDDEN_TARGET_FIELDS.intersection(row):
            raise PermissionError("external19 prediction row contains a target")
        checked = _validate_content_address(
            row, context=f"external19 prediction row {index}"
        )
        status = str(checked.get("status"))
        if (
            checked.get("schema_version")
            != EXTERNAL19_PREDICTION_ROW_SCHEMA_VERSION
            or checked.get("profile_id") != native18.EXTERNAL_NATIVE19_PROFILE_ID
            or checked.get("model_split") != "source_dev"
            or checked.get("prediction_roster_index") != index
            or checked.get("analysis_identity_id")
            != identity_row["analysis_identity_id"]
            or checked.get("patient_id") != identity_row["local_patient_id"]
            or checked.get("edf_relative_path") != identity_row["local_edf_path"]
            or checked.get("target_free_roster_receipt_sha256")
            != roster["receipt_sha256"]
            or checked.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
            or checked.get("inference_batch_size")
            != manifest["inference_batch_size"]
            or checked.get("reference_annotation_or_target_opened") is not False
            or checked.get("source_eval_opened") is not False
            or checked.get("released_threshold") != native18.RELEASED_THRESHOLD
            or checked.get("paper_native18_reproduction") is not False
            or checked.get("external_artifact_provenance_and_training_exposure_known")
            is not False
            or status not in allowed_status
        ):
            raise PermissionError("external19 frozen prediction row lineage drifted")
        expected_samples = int(identity_row["target_sample_count_256hz"])
        if int(checked["target_sample_count_256hz"]) != expected_samples:
            raise ValueError("external19 prediction duration drifted")
        if status == "external19_prediction_complete":
            if expected_samples < native18.TILE_SAMPLES:
                raise PermissionError("external19 below-60s row bypassed released skip")
            expected_path = (
                source.parent
                / "records"
                / str(identity_row["analysis_identity_id"])
                / "dense_posterior.npy"
            ).resolve(strict=True)
            posterior_path = Path(str(checked["posterior_path"])).resolve(strict=True)
            if (
                posterior_path != expected_path
                or posterior_path.is_symlink()
                or _file_sha256(posterior_path) != checked["posterior_npy_sha256"]
            ):
                raise ValueError("external19 frozen posterior bytes drifted")
            posterior = np.load(posterior_path, mmap_mode="r", allow_pickle=False)
            if (
                posterior.shape != (expected_samples,)
                or posterior.dtype != np.dtype("float32")
                or int(checked["posterior_sample_count"]) != expected_samples
            ):
                raise ValueError("external19 frozen posterior geometry drifted")
            replay = native18.decode_upstream_native18_posterior(
                np.asarray(posterior), threshold=native18.RELEASED_THRESHOLD
            )
            del posterior
            if _event_rows_from_decoded(replay) != checked.get("predicted_events"):
                raise ValueError("external19 frozen decoded events failed replay")
            if int(checked["predicted_event_count"]) != len(replay.event_sample_spans):
                raise ValueError("external19 predicted event count drifted")
        elif status == "external19_upstream_skip_below_60s" and (
            expected_samples >= native18.TILE_SAMPLES
            or checked.get("EDF_opened") is not False
            or checked.get("posterior_path") is not None
            or checked.get("posterior_npy_sha256") is not None
            or checked.get("predicted_events") != []
            or checked.get("predicted_event_count") != 0
            or checked.get("zero_alarm_assigned") is not True
        ):
            raise PermissionError("external19 released below-60s skip row drifted")
        elif status == "external19_typed_technical_failure":
            if (
                expected_samples < native18.TILE_SAMPLES
                or checked.get("EDF_opened") is not True
                or checked.get("failure_type") != "ValueError"
                or checked.get("failure_message")
                != "upstream get_data native19 requires all 19 electrodes"
                or checked.get("posterior_path") is not None
                or checked.get("posterior_npy_sha256") is not None
                or checked.get("predicted_events") != []
                or checked.get("predicted_event_count") != 0
                or checked.get("zero_alarm_assigned") is not True
            ):
                raise PermissionError("external19 typed failure is outside whitelist")
        replayed_statuses[status] += 1
    if (
        sum(replayed_statuses.values()) != 1821
        or replayed_statuses["external19_upstream_skip_below_60s"] != 215
        or replayed_statuses["external19_typed_technical_failure"] <= 0
        or replayed_statuses["external19_prediction_complete"] <= 0
    ):
        raise PermissionError("external19 full-roster admission contract drifted")
    if dict(sorted(replayed_statuses.items())) != manifest.get("status_counts"):
        raise ValueError("external19 status-count summary drifted")
    return manifest, roster, _file_sha256(source)


def _materialize_prediction_freeze_gate(
    *,
    output_dir: Path,
    prediction_manifest: Mapping[str, Any],
    prediction_manifest_file_sha256: str,
    roster: Mapping[str, Any],
) -> dict[str, Any]:
    rows = prediction_manifest["prediction_rows"]
    external19 = (
        prediction_manifest.get("schema_version")
        == EXTERNAL19_PREDICTION_MANIFEST_SCHEMA_VERSION
    )
    gate_provider_id = (
        native18.EXTERNAL_NATIVE19_PROFILE_ID if external19 else PROVIDER_ID
    )
    gate = _content_address(
        {
            "schema_version": FREEZE_GATE_SCHEMA_VERSION,
            "provider_id": gate_provider_id,
            "prediction_profile": (
                "external19_public_get_data_signal_side_reconstruction_diagnostic"
                if external19
                else "paper_architecture_native18_tusz_only_cleanroom"
            ),
            "prediction_manifest_receipt_sha256": prediction_manifest[
                "receipt_sha256"
            ],
            "prediction_manifest_file_sha256": prediction_manifest_file_sha256,
            "target_free_roster_receipt_sha256": roster["receipt_sha256"],
            "record_count": len(rows),
            "prediction_row_receipt_roster_sha256": _canonical_sha256(
                [row["receipt_sha256"] for row in rows]
            ),
            "posterior_payload_roster_sha256": _canonical_sha256(
                [
                    {
                        "analysis_identity_id": row["analysis_identity_id"],
                        "status": row["status"],
                        "posterior_npy_sha256": row.get("posterior_npy_sha256"),
                    }
                    for row in rows
                ]
            ),
            "decoded_event_roster_sha256": _canonical_sha256(
                [
                    {
                        "analysis_identity_id": row["analysis_identity_id"],
                        "predicted_events": row["predicted_events"],
                    }
                    for row in rows
                ]
            ),
            "posterior_and_event_bytes_replayed_before_gate": True,
            "reference_manifest_opened_before_gate": False,
            "source_eval_opened": False,
            "receipt_sha256": _PENDING,
        }
    )
    gate_path = output_dir / "prediction_freeze_gate.json"
    if gate_path.exists() or gate_path.is_symlink():
        observed = _load_json_content_address(
            gate_path, context="native18 prediction freeze gate"
        )
        if observed != gate:
            raise PermissionError("native18 prediction freeze gate already differs")
    else:
        _write_json_atomic(gate_path, gate, replace=False)
    return gate


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )


def _merge_reference_events_at_native_clock(
    record: Mapping[str, Any]
) -> list[dict[str, float]]:
    return [
        {
            "start_seconds": start / native18.TARGET_FS_HZ,
            "stop_seconds": stop / native18.TARGET_FS_HZ,
        }
        for start, stop in _upstream_event_spans(
            record, sample_count=int(record["target_sample_count_256hz"])
        )
    ]


def independent_onset_collar_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerances: Sequence[float] = (1.0, 3.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Score onset points independently from strict interval-overlap matching."""

    reference_count = sum(len(row["reference_events"]) for row in rows)
    prediction_count = sum(len(row["predicted_events"]) for row in rows)
    result: dict[str, Any] = {
        "matching_contract": (
            "ordered_one_to_one_onset_points_with_absolute_collar_no_interval_"
            "overlap_precondition"
        ),
        "interval_overlap_checked_before_onset_matching": False,
        "reference_onset_count": reference_count,
        "predicted_onset_count": prediction_count,
        "tolerances": {},
    }
    for tolerance in tolerances:
        errors: list[float] = []
        for row in rows:
            errors.extend(
                signed_error
                for _, _, signed_error in ordered_onset_collar_matching(
                    row["reference_events"],
                    row["predicted_events"],
                    early_seconds=float(tolerance),
                    late_seconds=float(tolerance),
                )
            )
        recall = _safe_rate(len(errors), reference_count)
        precision = _safe_rate(len(errors), prediction_count)
        result["tolerances"][f"{tolerance:g}s"] = {
            "matched_onset_count": len(errors),
            "reference_onset_denominator": reference_count,
            "predicted_onset_denominator": prediction_count,
            "sensitivity_or_hit_rate": recall,
            "precision": precision,
            "f1": _f1(precision, recall),
            "signed_error_mean_seconds": (
                None if not errors else float(np.mean(errors))
            ),
            "absolute_error_median_seconds": (
                None if not errors else float(statistics.median(abs(x) for x in errors))
            ),
        }
    return result


def _count_metrics(
    *, ref_true: int, true_positive: int, false_positive: int, seconds: float
) -> dict[str, Any]:
    sensitivity = _safe_rate(true_positive, ref_true)
    precision = _safe_rate(true_positive, true_positive + false_positive)
    return {
        "reference_positive_count": int(ref_true),
        "true_positive_count": int(true_positive),
        "false_positive_count": int(false_positive),
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": _f1(precision, sensitivity),
        "false_positives_per_24h": _safe_rate(
            false_positive, seconds / 86400.0
        ),
    }


def _timescoring_authority(project_root: Path):
    vendor = (
        project_root / "third_party/epilepsy_performance_metrics_426f8d2b"
    ).resolve(strict=True)
    if (vendor / ".git/HEAD").read_text(encoding="utf-8").strip() != TIMESCORING_COMMIT:
        raise PermissionError("pinned timescoring checkout commit drifted")
    import sys

    source = vendor / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import timescoring.annotations as annotations_module  # type: ignore
    import timescoring.scoring as scoring_module  # type: ignore
    from timescoring.annotations import Annotation  # type: ignore
    from timescoring.scoring import EventScoring, SampleScoring  # type: ignore

    if (
        Path(str(annotations_module.__file__)).resolve(strict=True)
        != (source / "timescoring/annotations.py").resolve(strict=True)
        or Path(str(scoring_module.__file__)).resolve(strict=True)
        != (source / "timescoring/scoring.py").resolve(strict=True)
    ):
        raise PermissionError("runtime timescoring import does not use pinned source")
    return Annotation, EventScoring, SampleScoring, {
        "repository": "https://github.com/esl-epfl/epilepsy_performance_metrics",
        "commit": TIMESCORING_COMMIT,
        "version": TIMESCORING_VERSION,
        "sample_scoring_rate_hz": native18.TARGET_FS_HZ,
        "event_parameters": {
            "toleranceStart": 30,
            "toleranceEnd": 60,
            "minOverlap": 0,
            "maxEventDuration": 300,
            "minDurationBetweenEvents": 90,
        },
    }


def _szcore_metrics(
    rows: Sequence[Mapping[str, Any]], *, project_root: Path
) -> dict[str, Any]:
    Annotation, EventScoring, SampleScoring, authority = _timescoring_authority(
        project_root
    )
    parameters = EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=300,
        minDurationBetweenEvents=90,
    )
    event_totals: Counter[str] = Counter()
    sample_totals: Counter[str] = Counter()
    total_seconds = 0.0
    for row in rows:
        sample_count = int(row["target_sample_count_256hz"])
        references = [
            (float(event["start_seconds"]), float(event["stop_seconds"]))
            for event in row["reference_events"]
        ]
        predictions = [
            (float(event["start_seconds"]), float(event["stop_seconds"]))
            for event in row["predicted_events"]
        ]
        reference = Annotation(
            references, native18.TARGET_FS_HZ, sample_count
        )
        hypothesis = Annotation(
            predictions, native18.TARGET_FS_HZ, sample_count
        )
        event_score = EventScoring(reference, hypothesis, parameters)
        sample_score = SampleScoring(
            reference, hypothesis, fs=native18.TARGET_FS_HZ
        )
        event_totals.update(
            {
                "ref_true": int(event_score.refTrue),
                "tp": int(event_score.tp),
                "fp": int(event_score.fp),
            }
        )
        sample_totals.update(
            {
                "ref_true": int(sample_score.refTrue),
                "tp": int(sample_score.tp),
                "fp": int(sample_score.fp),
            }
        )
        total_seconds += sample_count / native18.TARGET_FS_HZ
    return {
        "authority": authority,
        "event_pooled": _count_metrics(
            ref_true=event_totals["ref_true"],
            true_positive=event_totals["tp"],
            false_positive=event_totals["fp"],
            seconds=total_seconds,
        ),
        "sample_256hz_pooled": _count_metrics(
            ref_true=sample_totals["ref_true"],
            true_positive=sample_totals["tp"],
            false_positive=sample_totals["fp"],
            seconds=total_seconds,
        ),
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def seizure_free_alarm_distribution(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seizure_free = [row for row in rows if not row["reference_events"]]
    alarm_counts = [len(row["predicted_events"]) for row in seizure_free]
    alarm_rates: list[float] = []
    warning_fractions: list[float] = []
    total_duration = 0.0
    total_warning = 0.0
    for row, alarm_count in zip(seizure_free, alarm_counts, strict=True):
        duration = float(row["duration_seconds"])
        warning = sum(
            float(event["stop_seconds"]) - float(event["start_seconds"])
            for event in row["predicted_events"]
        )
        alarm_rates.append(alarm_count / (duration / 86400.0))
        warning_fractions.append(warning / duration)
        total_duration += duration
        total_warning += warning
    histogram = Counter(alarm_counts)
    return {
        "seizure_free_recording_count": len(seizure_free),
        "zero_alarm_recording_count": sum(count == 0 for count in alarm_counts),
        "zero_alarm_recording_proportion": _safe_rate(
            sum(count == 0 for count in alarm_counts), len(alarm_counts)
        ),
        "alarms_per_24h_per_record_distribution": _distribution(alarm_rates),
        "alarm_count_per_record_distribution": _distribution(
            [float(value) for value in alarm_counts]
        ),
        "alarm_count_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "time_in_warning_fraction_per_record_distribution": _distribution(
            warning_fractions
        ),
        "pooled_time_in_warning_fraction": _safe_rate(
            total_warning, total_duration
        ),
    }


def _load_source_dev_reference_manifest(
    path: Path, *, roster: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Load a legacy manifest, content-addressed dev view, or LAN dev sidecar.

    The LookAroundNet roster exporter already materializes a label-only JSONL
    for source-dev.  Adapting that sidecar after the prediction freeze gate
    avoids reopening the larger train+dev common17 manifest merely to score
    the frozen source-dev predictions.
    """

    if path.suffix.lower() == ".jsonl":
        if roster is None:
            raise ValueError("LAN source-dev label sidecar requires frozen roster")
        summary_path = (path.parent.parent / "summary.json").resolve(strict=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        try:
            split_summary = summary["splits"]["source_dev"]
            bound_path = Path(str(split_summary["labels_path"])).resolve(strict=True)
        except (KeyError, TypeError, OSError) as error:
            raise ValueError("LAN source-dev summary binding is malformed") from error
        if bound_path != path:
            raise PermissionError("LAN source-dev label sidecar path binding drifted")
        label_sha256 = _file_sha256(path)
        if split_summary.get("labels_sha256") != label_sha256:
            raise PermissionError("LAN source-dev label sidecar content drifted")
        if (
            split_summary.get("record_count") != 1821
            or split_summary.get("patient_count") != 53
            or roster.get("record_count") != 1821
            or roster.get("patient_count") != 53
        ):
            raise PermissionError("LAN source-dev denominator binding drifted")
        roster_rows = roster.get("records")
        if not isinstance(roster_rows, list) or len(roster_rows) != 1821:
            raise ValueError("frozen source-dev roster rows are incomplete")
        roster_by_identity = {
            str(row["analysis_identity_id"]): row for row in roster_rows
        }
        if len(roster_by_identity) != 1821:
            raise PermissionError("frozen source-dev roster identities repeat")
        label_fields = {
            "record_index",
            "analysis_identity_id",
            "patient_id",
            "model_split",
            "reference_csv_bi_path",
            "seizure_event_count",
            "seizure_events",
            "reference_csv_bi_sha256",
        }
        adapted_rows: list[dict[str, Any]] = []
        identities: set[str] = set()
        patients: set[str] = set()
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError("LAN source-dev label sidecar contains blank row")
                try:
                    label = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"LAN source-dev label row {line_number} is invalid JSON"
                    ) from error
                if not isinstance(label, Mapping) or set(label) != label_fields:
                    raise ValueError("LAN source-dev label row fields drifted")
                identity = str(label["analysis_identity_id"])
                identity_row = roster_by_identity.get(identity)
                if identity_row is None or identity in identities:
                    raise PermissionError("LAN source-dev label identity drifted")
                patient = str(label["patient_id"])
                if (
                    label["record_index"] != len(adapted_rows)
                    or label["model_split"] != "source_dev"
                    or patient != str(identity_row["local_patient_id"])
                    or not isinstance(label["seizure_events"], list)
                    or label["seizure_event_count"] != len(label["seizure_events"])
                ):
                    raise PermissionError("LAN source-dev label row binding drifted")
                adapted = {
                    "analysis_identity_id": identity,
                    "patient_id": patient,
                    "model_split": "source_dev",
                    "target_sample_count_256hz": int(
                        identity_row["target_sample_count_256hz"]
                    ),
                    "seizure_events": label["seizure_events"],
                    "audited_observed_channel_ids": list(
                        identity_row.get("audited_observed_channel_ids", [])
                    ),
                }
                _upstream_event_spans(adapted)
                adapted_rows.append(adapted)
                identities.add(identity)
                patients.add(patient)
        if (
            len(adapted_rows) != 1821
            or set(roster_by_identity) != identities
            or len(patients) != 53
        ):
            raise PermissionError("LAN source-dev label roster drifted")
        return _content_address(
            {
                "schema_version": "native18_source_dev_lan_sidecar_adapter_v1",
                "split": "source_dev",
                "record_count": len(adapted_rows),
                "patient_count": len(patients),
                "labels_path": str(path),
                "labels_sha256": label_sha256,
                "roster_summary_path": str(summary_path),
                "roster_summary_sha256": _file_sha256(summary_path),
                "source_eval_opened": False,
                "records": adapted_rows,
                "receipt_sha256": _PENDING,
            }
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SOURCE_DEV_REFERENCE_SCHEMA_VERSION:
        return load_common17_manifest(path, require_complete=True)
    value = _validate_content_address(raw, context="native18 source-dev reference")
    required = {
        "schema_version",
        "split",
        "parent_common17_manifest_receipt_sha256",
        "record_count",
        "patient_count",
        "records",
        "source_eval_opened",
        "receipt_sha256",
    }
    if set(value) != required:
        raise ValueError("native18 source-dev reference fields drifted")
    if (
        value["split"] != "source_dev"
        or value["record_count"] != 1821
        or value["patient_count"] != 53
        or value["source_eval_opened"] is not False
        or not isinstance(value["parent_common17_manifest_receipt_sha256"], str)
        or len(value["parent_common17_manifest_receipt_sha256"]) != 64
        or not isinstance(value["records"], list)
        or len(value["records"]) != 1821
    ):
        raise PermissionError("native18 source-dev reference denominator drifted")
    row_fields = {
        "analysis_identity_id",
        "patient_id",
        "model_split",
        "target_sample_count_256hz",
        "seizure_events",
        "audited_observed_channel_ids",
    }
    identities: set[str] = set()
    patients: set[str] = set()
    for row in value["records"]:
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise ValueError("native18 source-dev reference row fields drifted")
        identity = str(row["analysis_identity_id"])
        if row["model_split"] != "source_dev" or identity in identities:
            raise PermissionError("native18 source-dev reference identity drifted")
        if not isinstance(row["audited_observed_channel_ids"], list):
            raise TypeError("native18 source-dev observed channels must be a list")
        _upstream_event_spans(row)
        identities.add(identity)
        patients.add(str(row["patient_id"]))
    if len(identities) != 1821 or len(patients) != 53:
        raise PermissionError("native18 source-dev reference roster drifted")
    return value


def evaluate_frozen_source_dev(
    *,
    prediction_manifest_path: str | Path,
    reference_manifest_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Freeze first, then score strict, independent-onset and SzCORE tracks."""

    began = time.perf_counter()
    prediction_source = Path(prediction_manifest_path).resolve(strict=True)
    # Prediction-only schema dispatch.  No reference-bearing source is opened
    # until the selected validator has replayed the complete inventory and the
    # freeze gate has been materialized below.
    prediction_probe = _load_json_content_address(
        prediction_source, context="source-dev prediction manifest schema preflight"
    )
    external19 = (
        prediction_probe.get("schema_version")
        == EXTERNAL19_PREDICTION_MANIFEST_SCHEMA_VERSION
    )
    if external19:
        prediction, roster, prediction_file_sha = (
            _validate_external19_frozen_prediction_manifest(prediction_source)
        )
    elif prediction_probe.get("schema_version") == PREDICTION_MANIFEST_SCHEMA_VERSION:
        prediction, roster, prediction_file_sha = _validate_frozen_prediction_manifest(
            prediction_source
        )
    else:
        raise PermissionError("unsupported source-dev prediction profile")
    output = Path(output_dir).resolve(strict=False)
    if output.is_symlink():
        raise PermissionError("native18 evaluation output may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    gate = _materialize_prediction_freeze_gate(
        output_dir=output,
        prediction_manifest=prediction,
        prediction_manifest_file_sha256=prediction_file_sha,
        roster=roster,
    )

    # This is intentionally the first reference-bearing read in this function.
    reference_source = Path(reference_manifest_path).resolve(strict=True)
    reference_manifest = _load_source_dev_reference_manifest(
        reference_source, roster=roster
    )
    reference_by_identity = {
        str(row["analysis_identity_id"]): dict(row)
        for row in reference_manifest["records"]
        if row["model_split"] == "source_dev"
    }
    if set(reference_by_identity) != {
        str(row["analysis_identity_id"]) for row in roster["records"]
    }:
        raise PermissionError("post-freeze source-dev reference roster drifted")
    prediction_by_identity = {
        str(row["analysis_identity_id"]): row
        for row in prediction["prediction_rows"]
    }
    joined: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    strata_by_record: dict[str, tuple[str, str]] = {}
    for identity_row in roster["records"]:
        identity = str(identity_row["analysis_identity_id"])
        reference = reference_by_identity[identity]
        prediction_row = prediction_by_identity[identity]
        target_samples = int(identity_row["target_sample_count_256hz"])
        if int(reference["target_sample_count_256hz"]) != target_samples:
            raise ValueError("post-freeze reference duration drifted")
        reference_events = _merge_reference_events_at_native_clock(reference)
        predicted_events = [
            {
                "start_seconds": float(event["start_seconds"]),
                "stop_seconds": float(event["stop_seconds"]),
            }
            for event in prediction_row["predicted_events"]
        ]
        duration = target_samples / native18.TARGET_FS_HZ
        benchmark = {
            "patient_id": str(identity_row["local_patient_id"]),
            "recording_id": identity,
            "split": "source_dev",
            "duration_seconds": duration,
            "reference_events": reference_events,
            "predicted_events": predicted_events,
        }
        benchmark_rows.append(benchmark)
        observed = {
            str(channel).upper()
            for channel in reference.get("audited_observed_channel_ids", [])
        }
        strata_by_record[identity] = (
            (
                "midline_direct_fz_pz"
                if {"FZ", "PZ"}.issubset(observed)
                else "midline_missing_fz_or_pz"
            ),
            (
                "duration_lt_60s"
                if target_samples < native18.TILE_SAMPLES
                else "duration_ge_60s"
            ),
        )
        joined.append(
            {
                **benchmark,
                "target_sample_count_256hz": target_samples,
                "prediction_status": prediction_row["status"],
            }
        )
    normalized_score = score_normalized_source_dev_rows(
        benchmark_rows,
        project_root=Path(project_root).resolve(strict=True),
        strata_by_record=strata_by_record,
        szcore_sample_rate_hz=native18.TARGET_FS_HZ,
    )
    complete_status = (
        "external19_prediction_complete" if external19 else "prediction_complete"
    )
    signal_side_identities = {
        str(row["analysis_identity_id"])
        for row in prediction["prediction_rows"]
        if row["status"] == complete_status
    }
    signal_side_rows = [
        row
        for row in benchmark_rows
        if str(row["recording_id"]) in signal_side_identities
    ]
    signal_side_score = score_normalized_source_dev_rows(
        signal_side_rows,
        project_root=Path(project_root).resolve(strict=True),
        strata_by_record={
            identity: strata_by_record[identity]
            for identity in sorted(signal_side_identities)
        },
        szcore_sample_rate_hz=native18.TARGET_FS_HZ,
    )
    score_tracks = normalized_score["score"]
    strict = score_tracks[
        "strict_zero_dilation_ordered_one_to_one_overlap"
    ]["metrics"]
    independent_onset = score_tracks[
        "independent_onset_collar_ordered_one_to_one"
    ]
    szcore = score_tracks["official_szcore_compatible_timescoring_0_0_7"]
    seizure_free = score_tracks["seizure_free_alarm_distribution"]
    overlap_conditioned_onset = score_tracks[
        "overlap_conditioned_onset_diagnostic_not_primary"
    ]
    status_counts = Counter(
        str(row["status"]) for row in prediction["prediction_rows"]
    )
    failure_types = Counter(
        str(row.get("failure_type"))
        for row in prediction["prediction_rows"]
        if row["status"]
        in {"typed_technical_failure", "external19_typed_technical_failure"}
    )
    skip_status = (
        "external19_upstream_skip_below_60s"
        if external19
        else "upstream_skip_below_60s"
    )
    profile_id = (
        native18.EXTERNAL_NATIVE19_PROFILE_ID if external19 else PROVIDER_ID
    )
    denominator_lanes = (
        {
            "public_get_data_signal_side_evaluable": {
                "status": "scored",
                "admission": (
                    "public get_data() signal-side reconstruction successes only; "
                    "released <60s skips and typed missing-electrode failures are "
                    "excluded. This is not the official eval_test.py denominator, "
                    "because that script calls get_data_18()."
                ),
                "record_count": len(signal_side_rows),
                "excluded_record_count": len(benchmark_rows)
                - len(signal_side_rows),
                "metrics": signal_side_score,
            },
            "full_intention_to_assess_1821": {
                "status": "scored",
                "admission": (
                    "all canonical source-dev records; released <60s skips and "
                    "whitelisted technical failures are retained as zero alarms "
                    "with their references and durations intact"
                ),
                "record_count": len(benchmark_rows),
                "zero_alarm_failure_and_skip_count": (
                    status_counts["external19_upstream_skip_below_60s"]
                    + status_counts["external19_typed_technical_failure"]
                ),
                "metrics": normalized_score,
            },
        }
        if external19
        else {
            "released_signal_side_evaluable_ge60s": {
                "status": "scored",
                "admission": (
                    "released native18 records with completed >=60s inference only; "
                    "the 215 released <60s skips are excluded. This is a source-dev "
                    "signal-side diagnostic, not a paper checkpoint or independent test."
                ),
                "record_count": len(signal_side_rows),
                "excluded_record_count": len(benchmark_rows) - len(signal_side_rows),
                "metrics": signal_side_score,
            },
            "full_intention_to_assess_native18": {
                "status": "scored",
                "admission": (
                    "all 1,821 canonical source-dev records; the 215 released <60s "
                    "skips retain references and duration as zero alarms"
                ),
                "record_count": len(benchmark_rows),
                "metrics": normalized_score,
            }
        }
    )
    receipt = _content_address(
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "provider_id": profile_id,
            "prediction_profile": (
                "external19_public_get_data_signal_side_reconstruction_diagnostic"
                if external19
                else "paper_architecture_native18_tusz_only_cleanroom"
            ),
            "status": "completed_source_dev_postfreeze_evaluation",
            "claim_boundary": (
                (
                    "external19 artifact plus public-get_data signal-side "
                    "reconstruction diagnostic at released threshold0.8; public "
                    "eval_test.py instead calls get_data_18(); uploader/original-"
                    "checkpoint equivalence and training exposure are unknown; "
                    "not paper native18, not source-eval, and not a clean test estimate"
                )
                if external19
                else (
                    "fixed epoch100 TUSZ-only native18 and released threshold0.8 "
                    "on source-dev; not source-eval, not the paper's Siena+TUSZ "
                    "best-dev checkpoint selection"
                )
            ),
            "prediction_freeze_gate": gate,
            "reference_opened_only_after_prediction_freeze_gate": True,
            "prediction_manifest_path": str(
                Path(prediction_manifest_path).resolve(strict=True)
            ),
            "prediction_manifest_file_sha256": prediction_file_sha,
            "prediction_manifest_receipt_sha256": prediction["receipt_sha256"],
            "reference_manifest_path": str(reference_source),
            "reference_manifest_file_sha256": _file_sha256(reference_source),
            "reference_manifest_receipt_sha256": reference_manifest[
                "receipt_sha256"
            ],
            "record_count": len(joined),
            "patient_count": len({row["patient_id"] for row in joined}),
            "reference_event_count": sum(
                len(row["reference_events"]) for row in joined
            ),
            "prediction_status_counts": dict(sorted(status_counts.items())),
            "typed_technical_failure_types": dict(sorted(failure_types.items())),
            "below_60s_upstream_skip_record_count": status_counts[
                skip_status
            ],
            "below_60s_upstream_skip_reference_event_count_postfreeze": sum(
                len(row["reference_events"])
                for row in joined
                if row["prediction_status"] == skip_status
            ),
            "typed_technical_failure_reference_event_count_postfreeze": sum(
                len(row["reference_events"])
                for row in joined
                if row["prediction_status"]
                in {"typed_technical_failure", "external19_typed_technical_failure"}
            ),
            "technical_failure_count": sum(failure_types.values()),
            "technical_failures_retained_as_zero_alarm_in_full_ITA_denominator": True,
            "lt60s_released_skips_retained_as_zero_alarm_in_primary_denominator": True,
            "metric_denominator_lanes": denominator_lanes,
            "top_level_metric_alias": (
                "full_intention_to_assess_1821"
                if external19
                else "full_intention_to_assess_native18"
            ),
            "strict_zero_dilation_ordered_one_to_one": strict,
            "strict_false_alarms_per_24h": strict[
                "alarm_false_alarms_per_24h"
            ],
            "independent_onset_collar": independent_onset,
            "overlap_conditioned_onset_diagnostic_not_primary": {
                **overlap_conditioned_onset,
            },
            "official_szcore_compatible_timescoring_0_0_7": szcore,
            "seizure_free_alarm_distribution": seizure_free,
            "zero_alarm_summary": score_tracks["zero_alarm_summary"],
            "missing_midline_and_short_record_strata": normalized_score["strata"],
            "normalized_three_track_score_schema": normalized_score["schema_version"],
            "decoder": {
                "posterior_threshold_comparison": "strictly_greater_than_0.8",
                "morphology_clock_hz": native18.TARGET_FS_HZ,
                "opening_kernel_samples": native18.MORPHOLOGY_KERNEL_SAMPLES,
                "closing_kernel_samples": native18.MORPHOLOGY_KERNEL_SAMPLES,
                "remove_runs_shorter_than_samples": native18.MINIMUM_EVENT_SAMPLES,
                "one_hz_aggregation_used": False,
            },
            "paper_native18_reproduction": False,
            "paper_native18_architecture_and_preprocessing_path": not external19,
            "paper_training_exposure_reproduced": False,
            "external_artifact_uploader_is_upstream_author_verified": (
                None if not external19 else False
            ),
            "external_artifact_original_checkpoint_hash_verified": (
                None if not external19 else False
            ),
            "external_artifact_training_exposure_documented": (
                None if not external19 else False
            ),
            "external19_midline_adaptation_executed": (
                None if not external19 else False
            ),
            "external19_literal_failure_whitelist": (
                None
                if not external19
                else {
                    "failure_type": "ValueError",
                    "failure_message": (
                        "upstream get_data native19 requires all 19 electrodes"
                    ),
                }
            ),
            "source_dev_selected_epoch": False if not external19 else None,
            "source_dev_selected_threshold": False if not external19 else None,
            "source_eval_opened": False,
            "evaluation_wall_seconds": time.perf_counter() - began,
            "clinical_use_authorized": False,
            "receipt_sha256": _PENDING,
        }
    )
    _write_json_atomic(output / "evaluation_receipt.json", receipt, replace=False)
    return receipt


def run_real_native18_smoke(
    *,
    manifest_path: str | Path,
    analysis_projection_path: str | Path,
    physical_audit_path: str | Path,
    tusz_root: str | Path,
    output_dir: str | Path,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run one real cached train step and one real target-free dev record."""

    output = Path(output_dir).resolve(strict=False)
    if output.is_symlink():
        raise PermissionError("native18 smoke output may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    cache = output / "selected_tile_cache"
    cache_receipt = materialize_selected_tile_cache(
        manifest_path=manifest_path,
        cache_dir=cache,
        maximum_new_tiles=1,
        progress_every_records=1,
    )
    if cache_receipt["valid_cached_tile_count"] < 1:
        raise RuntimeError("native18 real smoke did not materialize a train tile")
    plan = _load_json_content_address(
        cache / "selected_tile_plan.json", context="native18 smoke tile plan"
    )
    cached_ids = {
        str(row["tile_id"]) for row in cache_receipt["tile_inventory"]
    }
    tile_row = next(
        row for row in plan["selected_tiles"] if row["tile_id"] in cached_ids
    )
    signal_path, target_path, _ = _tile_paths(cache, str(tile_row["tile_id"]))
    device = torch.device(device_name)
    model = native18.build_upstream_native18_model(
        seed=DEFAULT_MODEL_SEED, device=str(device)
    )
    optimizer = native18.build_upstream_native18_optimizer(model)
    signal = torch.from_numpy(
        np.asarray(np.load(signal_path, allow_pickle=False), dtype=np.float32)
    )[None]
    target = torch.from_numpy(
        np.asarray(np.load(target_path, allow_pickle=False), dtype=np.float32)
    )[None]
    train_began = time.perf_counter()
    numeric_train = native18.train_upstream_native18_batches(
        model,
        optimizer,
        [(signal, target)],
        device=str(device),
    )
    train_seconds = time.perf_counter() - train_began

    roster = build_target_free_source_dev_roster(
        analysis_projection_path=analysis_projection_path,
        physical_audit_path=physical_audit_path,
    )
    root = Path(tusz_root).resolve(strict=True)
    candidates = sorted(
        (
            dict(row)
            for row in roster["records"]
            if int(row["target_sample_count_256hz"]) >= native18.TILE_SAMPLES
        ),
        key=lambda row: (
            int(row["target_sample_count_256hz"]),
            str(row["analysis_identity_id"]),
        ),
    )
    attempted_failures: list[dict[str, str]] = []
    dev_row: dict[str, Any] | None = None
    dev_result: native18.Native18InferenceResult | None = None
    dev_transform_seconds = 0.0
    dev_inference_seconds = 0.0
    for candidate in candidates:
        try:
            edf_path = _safe_edf(
                root, str(candidate["local_edf_path"]), split="source_dev"
            )
            transform_began = time.perf_counter()
            transformed = _transform_native18_edf(
                edf_path,
                expected_source_tensor_sha256=str(
                    candidate["canonical_source_tensor_sha256"]
                ),
                expected_target_sample_count=int(
                    candidate["target_sample_count_256hz"]
                ),
            )
            dev_transform_seconds = time.perf_counter() - transform_began
            inference_began = time.perf_counter()
            dev_result = native18.infer_upstream_native18_full_record(
                model,
                transformed,
                device=str(device),
                batch_size=1,
                threshold=native18.RELEASED_THRESHOLD,
            )
            dev_inference_seconds = time.perf_counter() - inference_began
            dev_row = candidate
            break
        except Exception as error:
            attempted_failures.append(
                {
                    "analysis_identity_id": str(
                        candidate["analysis_identity_id"]
                    ),
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                }
            )
            if len(attempted_failures) >= 10:
                break
    if dev_row is None or dev_result is None:
        raise RuntimeError("native18 target-free dev smoke found no executable record")
    posterior_path = output / "target_free_dev_smoke_posterior.npy"
    _save_numpy_atomic(posterior_path, dev_result.posterior, replace=True)
    eeg_hours = len(dev_result.posterior) / native18.TARGET_FS_HZ / 3600.0
    full_dev_hours = 435.08444444444444
    full_plan_storage = plan["cache_storage_estimate"]
    receipt = _content_address(
        {
            "schema_version": "native18_tusz_only_real_smoke_v1",
            "provider_id": PROVIDER_ID,
            "claim_status": "real_IO_and_numeric_smoke_not_accuracy_evaluation",
            "source_train": {
                "tile_id": tile_row["tile_id"],
                "analysis_identity_id": tile_row["analysis_identity_id"],
                "category": tile_row["category"],
                "positive_sample_count": tile_row["positive_sample_count"],
                "cache_inventory_receipt_sha256": cache_receipt[
                    "receipt_sha256"
                ],
                "cache_materialization_wall_seconds": cache_receipt[
                    "wall_seconds"
                ],
                "exact_model_RAdam_BCE_step_receipt": numeric_train,
                "exact_model_train_step_seconds": train_seconds,
            },
            "target_free_source_dev": {
                "analysis_identity_id": dev_row["analysis_identity_id"],
                "patient_id": dev_row["local_patient_id"],
                "duration_seconds": len(dev_result.posterior)
                / native18.TARGET_FS_HZ,
                "posterior_path": str(posterior_path),
                "posterior_npy_sha256": _file_sha256(posterior_path),
                "predicted_event_count": len(
                    dev_result.decoded.event_sample_spans
                ),
                "transform_seconds": dev_transform_seconds,
                "inference_seconds": dev_inference_seconds,
                "inference_seconds_per_EEG_hour": _safe_rate(
                    dev_inference_seconds, eeg_hours
                ),
                "rough_full_435h_inference_seconds_from_one_record": (
                    _safe_rate(dev_inference_seconds, eeg_hours) * full_dev_hours
                ),
                "attempted_target_free_failures_before_success": attempted_failures,
                "reference_annotation_or_target_opened": False,
            },
            "resource_estimates": {
                "full_selected_tile_count": plan["selected_window_count"],
                "full_selected_positive_sample_count": plan[
                    "selected_positive_sample_count"
                ],
                "full_selected_positive_sample_fraction": plan[
                    "selected_positive_sample_fraction"
                ],
                "full_cache_selected_payload_GiB": full_plan_storage[
                    "selected_payload_GiB"
                ],
                "full_cache_peak_upper_bound_GiB": cache_receipt[
                    "storage_contract"
                ]["estimated_complete_cache_peak_upper_bound_GiB"],
                "whole_record_disk_cache_persisted": False,
                "atomic_temporary_extra_upper_bound_MiB": cache_receipt[
                    "storage_contract"
                ]["atomic_peak_extra_upper_bound_bytes"]
                / 1048576.0,
                "official_logical_batch_size": 86,
                "microbatch_required_under_current_GPU_occupancy": True,
                "microbatch_BatchNorm1d_not_equivalent_to_batch86": True,
            },
            "fixed100_training_was_not_executed_by_smoke": True,
            "random_one_step_model_may_not_be_used_for_accuracy_claim": True,
            "source_eval_opened": False,
            "vLLM_service_stopped": False,
            "wall_seconds": time.perf_counter() - began,
            "receipt_sha256": _PENDING,
        }
    )
    _write_json_atomic(output / "real_smoke_receipt.json", receipt, replace=True)
    return receipt


def _parse_monitoring_epochs(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("monitoring epochs must be comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("monitoring epochs may not be empty")
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TUSZ-only upstream-native18 fixed100 clean-room runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--manifest", required=True)
    plan_parser.add_argument("--output", required=True)
    plan_parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    plan_parser.add_argument("--parent-manifest-receipt-sha256")

    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("--manifest", required=True)
    cache_parser.add_argument("--cache-dir", required=True)
    cache_parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    cache_parser.add_argument("--maximum-source-records", type=int)
    cache_parser.add_argument("--maximum-new-tiles", type=int)
    cache_parser.add_argument("--tusz-root")
    cache_parser.add_argument("--progress-every-records", type=int, default=25)
    cache_parser.add_argument(
        "--workers", type=int, default=1, help="record-parallel CPU workers (1-8)"
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--microbatch-size", type=int, default=1)
    train_parser.add_argument(
        "--precision", choices=("fp32", "cuda_bf16"), default="fp32"
    )
    train_parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    train_parser.add_argument("--maximum-steps", type=int)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--checkpoint-every-batches", type=int, default=25)
    train_parser.add_argument("--verify-cache-payload-hashes", action="store_true")
    train_parser.add_argument(
        "--monitoring-epochs",
        type=_parse_monitoring_epochs,
        default=(20, 40, 60, 80, 100),
    )

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--analysis-projection", required=True)
    infer_parser.add_argument("--physical-audit", required=True)
    infer_parser.add_argument("--tusz-root", required=True)
    infer_parser.add_argument("--output-dir", required=True)
    infer_parser.add_argument("--device", default="cuda:0")
    infer_parser.add_argument("--batch-size", type=int, default=1)
    infer_parser.add_argument("--maximum-records", type=int)

    external_parser = subparsers.add_parser("infer-external19")
    external_parser.add_argument("--checkpoint", required=True)
    external_parser.add_argument("--analysis-projection", required=True)
    external_parser.add_argument("--physical-audit", required=True)
    external_parser.add_argument("--tusz-root", required=True)
    external_parser.add_argument("--output-dir", required=True)
    external_parser.add_argument("--device", default="cuda:0")
    external_parser.add_argument("--batch-size", type=int, default=1)
    external_parser.add_argument("--maximum-records", type=int)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--prediction-manifest", required=True)
    score_parser.add_argument("--reference-manifest", required=True)
    score_parser.add_argument("--output-dir", required=True)
    score_parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[2])
    )

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--manifest", required=True)
    smoke_parser.add_argument("--analysis-projection", required=True)
    smoke_parser.add_argument("--physical-audit", required=True)
    smoke_parser.add_argument("--tusz-root", required=True)
    smoke_parser.add_argument("--output-dir", required=True)
    smoke_parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    if arguments.command == "plan":
        result = build_selected_tile_plan(
            arguments.manifest,
            seed=arguments.seed,
            parent_manifest_receipt_sha256=(
                arguments.parent_manifest_receipt_sha256
            ),
        )
        _write_json_atomic(Path(arguments.output), result, replace=False)
    elif arguments.command == "cache":
        result = materialize_selected_tile_cache(
            manifest_path=arguments.manifest,
            cache_dir=arguments.cache_dir,
            seed=arguments.seed,
            maximum_source_records=arguments.maximum_source_records,
            maximum_new_tiles=arguments.maximum_new_tiles,
            progress_every_records=arguments.progress_every_records,
            workers=arguments.workers,
            tusz_root=arguments.tusz_root,
        )
    elif arguments.command == "train":
        result = train_tusz_only_native18(
            cache_dir=arguments.cache_dir,
            output_dir=arguments.output_dir,
            microbatch_size=arguments.microbatch_size,
            device_name=arguments.device,
            precision=arguments.precision,
            model_seed=arguments.model_seed,
            maximum_steps=arguments.maximum_steps,
            resume=arguments.resume,
            checkpoint_every_batches=arguments.checkpoint_every_batches,
            verify_cache_payload_hashes=arguments.verify_cache_payload_hashes,
            monitoring_epoch_numbers=arguments.monitoring_epochs,
        )
    elif arguments.command == "infer":
        result = infer_target_free_source_dev(
            checkpoint_path=arguments.checkpoint,
            analysis_projection_path=arguments.analysis_projection,
            physical_audit_path=arguments.physical_audit,
            tusz_root=arguments.tusz_root,
            output_dir=arguments.output_dir,
            device_name=arguments.device,
            inference_batch_size=arguments.batch_size,
            maximum_records=arguments.maximum_records,
        )
    elif arguments.command == "infer-external19":
        result = infer_external19_target_free_source_dev_diagnostic(
            checkpoint_path=arguments.checkpoint,
            analysis_projection_path=arguments.analysis_projection,
            physical_audit_path=arguments.physical_audit,
            tusz_root=arguments.tusz_root,
            output_dir=arguments.output_dir,
            device_name=arguments.device,
            inference_batch_size=arguments.batch_size,
            maximum_records=arguments.maximum_records,
        )
    elif arguments.command == "score":
        result = evaluate_frozen_source_dev(
            prediction_manifest_path=arguments.prediction_manifest,
            reference_manifest_path=arguments.reference_manifest,
            output_dir=arguments.output_dir,
            project_root=arguments.project_root,
        )
    elif arguments.command == "smoke":
        result = run_real_native18_smoke(
            manifest_path=arguments.manifest,
            analysis_projection_path=arguments.analysis_projection,
            physical_audit_path=arguments.physical_audit,
            tusz_root=arguments.tusz_root,
            output_dir=arguments.output_dir,
            device_name=arguments.device,
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError("unreachable native18 command")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CACHE_INVENTORY_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "FIXED_EPOCH_COUNT",
    "PLAN_SCHEMA_VERSION",
    "PREDICTION_MANIFEST_SCHEMA_VERSION",
    "ROSTER_SCHEMA_VERSION",
    "build_selected_tile_plan",
    "build_selected_tile_plan_from_records",
    "build_target_free_source_dev_roster",
    "evaluate_frozen_source_dev",
    "independent_onset_collar_metrics",
    "infer_external19_target_free_source_dev_diagnostic",
    "infer_target_free_source_dev",
    "load_epoch100_native18_checkpoint",
    "main",
    "materialize_selected_tile_cache",
    "run_real_native18_smoke",
    "seizure_free_alarm_distribution",
    "train_tusz_only_native18",
]
