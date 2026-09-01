"""Executable, deliberately non-promotable common17 ST16 challenger.

This runner is the shortest honest path from the existing common17 TUSZ
inventory to a locally trained SeizureTransformer challenger.  It is kept
separate from the formal provider epoch executor because that executor's
cross-process/scalability admission is intentionally still closed.

Safety and scientific boundaries are executable, rather than comments:

* training accepts only the labelled ``source_train`` common17 manifest;
* prediction accepts only the target-free identity projection and only
  ``source_dev``; ``source_eval`` is rejected before an EDF is opened;
* ST16 is derived directly from the observed common17 referential carrier;
  no FZ/PZ synthesis, zero fill, interpolation, ST18 slicing, or checkpoint
  reuse is permitted;
* the engineering batch size is explicit and currently capped at eight while
  the Qwen vLLM service occupies about 40.6 GiB of the 48 GiB GPU.  A partial
  final batch must be explicitly admitted and is recorded in the plan;
* prediction persists the complete pre-threshold, pre-morphology, pre-NMS
  sample posterior after weighted OLA.  Detector selection can therefore be
  replayed without repeating model inference or losing close events;
* every emitted checkpoint and prediction receipt says exploratory and
  non-promotable.  Nothing here is a source-eval or clinical claim.

The training path creates only the selected epoch's transformed tiles in an
owned recoverable cache, grouped by recording.  This avoids retaining the
complete multi-hundred-hour corpus on disk and avoids re-transforming one long
EDF for every sampled tile.  Valid tiles survive an interrupted cache-build or
training invocation and are reused by content-derived tile id; the cache is
deleted only after the corresponding epoch completes.  The exact provider
transform remains the already frozen whole-record ST16 transform.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .canonical_edf_materialization import load_canonical_edf_record
from .detector_signal_lineage_authority_v1 import (
    authorize_detector_signal_lineage_from_canonical_record,
)
from .eventnet_common17_streaming_v1 import (
    COMMON17_CHANNEL_ORDER,
    load_common17_manifest,
)
from . import seizuretransformer_cleanroom_registry_v1 as st
from .seizuretransformer_streaming_ola_adapter_v1 import (
    SeizureTransformerStreamingResult,
    build_seizuretransformer_tile_plan,
    run_seizuretransformer_streaming_ola,
    validate_seizuretransformer_streaming_result,
)
from .tusz_complete_detector_roster_v2 import (
    validate_tusz_analysis_identity_projection_v2,
)
from third_party.SeizureTransformer.time_step_level.model import (
    SeizureTransformer,
)


SCHEMA_VERSION: Final[str] = "st16_common17_exploratory_runner_v1"
PROVIDER_ID: Final[str] = "st16_common17_exploratory_nonpromotable_v1"
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "st16_common17_exploratory_checkpoint_v1"
)
PREDICTION_SCHEMA_VERSION: Final[str] = (
    "st16_common17_source_dev_dense_prediction_inventory_v1"
)
PREDICTION_PREPROCESS_STAGE_SCHEMA_VERSION: Final[str] = (
    "st16_common17_source_dev_preprocess_stage_v1"
)
PREDICTION_PREPROCESS_WORKERS_MAXIMUM: Final[int] = 4
PREDICTION_PREFETCH_DEPTH_MAXIMUM: Final[int] = 8
TARGET_FS_HZ: Final[int] = st.TARGET_FS_HZ
TILE_SAMPLES: Final[int] = st.TILE_SAMPLES
TRAIN_HOP_SAMPLES: Final[int] = st.TRAIN_HOP_SAMPLES
TILES_PER_PATIENT_PER_EPOCH: Final[int] = 8
ENGINEERING_BATCH_SIZE_MAXIMUM: Final[int] = 8
PARTIAL_BATCH_POLICIES: Final[frozenset[str]] = frozenset(
    {"fail", "emit_explicit"}
)
_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
CACHE_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "st16_common17_exploratory_epoch_cache_contract_v1"
)
CACHE_TILE_SCHEMA_VERSION: Final[str] = (
    "st16_common17_exploratory_epoch_cache_tile_v1"
)
# One deliberately narrow migration exception for the markerless cache created
# by the already-running 2026-08-25 epoch-0 process.  No other markerless cache
# is eligible for adoption; see scripts/adopt_st16_markerless_epoch_cache_v1.py.
LEGACY_MARKERLESS_ADOPTION_PLAN_SHA256: Final[str] = (
    "4e6f8818f4db82912c7a3b97deef39b9f0e8669851afa74821cacf32f7a3addf"
)
LEGACY_LOCAL_STAGING_EQUIVALENCE_RECEIPT_SHA256: Final[str] = (
    "ed667b5d0777a292cdbd8cb7ca09797c005ff56d058ac8332ce309aff9f80d4d"
)


@dataclass(frozen=True)
class _ArrayTileReader:
    """Strongly bound reader over one exact provider-transformed recording."""

    signal: np.ndarray
    variant_id: str
    typed_units: tuple[str, ...]
    sampling_rate_numerator: int
    sampling_rate_denominator: int
    sample_count: int
    source_signal_sha256: str
    preprocessing_receipt_sha256: str
    input_clock_receipt_sha256: str

    def read_samples(self, start_sample: int, sample_count: int) -> np.ndarray:
        if (
            isinstance(start_sample, bool)
            or not isinstance(start_sample, int)
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or start_sample < 0
            or sample_count < 1
            or start_sample + sample_count > self.sample_count
        ):
            raise ValueError("ST16 reader request lies outside the recording")
        return np.ascontiguousarray(
            self.signal[:, start_sample : start_sample + sample_count],
            dtype="<f4",
        )


@dataclass(frozen=True)
class _StagedTransformCarrier:
    """Parent-side mmap view of a content-verified CPU worker transform."""

    signal: np.ndarray
    receipt: dict[str, Any]
    valid_support_sample_count: int
    preprocess_stage_receipt_sha256: str


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    value: Mapping[str, Any], *, artifact_name: str
) -> dict[str, Any]:
    """Return a copy only when its canonical self-address is byte-replayable."""

    result = deepcopy(dict(value))
    observed = result.get("receipt_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"{artifact_name} lacks a SHA-256 content address")
    result["receipt_sha256"] = _PENDING
    if _canonical_sha256(result) != observed:
        raise ValueError(f"{artifact_name} content address failed replay")
    result["receipt_sha256"] = observed
    return result


def _write_json_atomic(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    target = path.resolve(strict=False)
    if not replace and (target.exists() or target.is_symlink()):
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            # A hard-link install is an atomic no-clobber publication on the
            # same filesystem.  Unlike check-then-os.replace, a concurrent
            # writer can never be silently overwritten.
            os.link(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_numpy_atomic(
    path: Path, value: np.ndarray, *, replace: bool = False
) -> None:
    target = path.resolve(strict=False)
    if not replace and (target.exists() or target.is_symlink()):
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


def _safe_edf(root: Path, relative_path: str, *, expected_split: str) -> Path:
    relative = Path(relative_path)
    expected_prefix = {"source_train": "train", "source_dev": "dev"}.get(
        expected_split
    )
    if expected_prefix is None:
        raise PermissionError("source_eval and unknown splits are forbidden")
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != expected_prefix
        or relative.suffix.lower() != ".edf"
    ):
        raise PermissionError("EDF path crosses the authorized split root")
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    if path.is_symlink() or not path.is_file():
        raise ValueError("EDF must be a regular non-symlink file")
    return path


def _sample_center_index(seconds: object) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError("event seconds must be numeric")
    value = Fraction(str(seconds)) * TARGET_FS_HZ - Fraction(1, 2)
    return math.ceil(value)


def _merge_spans(spans: Sequence[Sequence[int]], sample_count: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for raw in spans:
        if len(raw) != 2:
            raise ValueError("event span must contain two indices")
        start, stop = int(raw[0]), int(raw[1])
        start = max(0, min(sample_count, start))
        stop = max(start, min(sample_count, stop))
        if stop > start:
            rows.append([start, stop])
    rows.sort()
    merged: list[list[int]] = []
    for start, stop in rows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def event_sample_spans(record: Mapping[str, Any]) -> list[list[int]]:
    """Project TERM intervals onto the frozen 256-Hz sample-center clock."""

    sample_count = int(record["target_sample_count_256hz"])
    raw_spans: list[list[int]] = []
    events = record.get("seizure_events")
    if not isinstance(events, list):
        raise TypeError("training record lacks seizure_events")
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("seizure event must be an object")
        start = _sample_center_index(event["start_seconds"])
        stop = _sample_center_index(event["stop_seconds"])
        raw_spans.append([start, stop])
    return _merge_spans(raw_spans, sample_count)


def _tile_id(analysis_identity_id: str, target_start_sample: int) -> str:
    return "ST16TILE-" + hashlib.sha256(
        f"{analysis_identity_id}|{target_start_sample}|{TILE_SAMPLES}|v1".encode()
    ).hexdigest()[:32]


def _validate_engineering_batch_contract(
    batch_size: int, partial_batch_policy: str
) -> None:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive explicit integer")
    if batch_size > ENGINEERING_BATCH_SIZE_MAXIMUM:
        raise PermissionError(
            "batch sizes above 8 are not memory-admitted while the Qwen vLLM "
            "service occupies approximately 40.6 GiB; batch16 remains an "
            "unexecuted registry maximum, not a runnable setting"
        )
    if partial_batch_policy not in PARTIAL_BATCH_POLICIES:
        raise ValueError(
            "partial_batch_policy must be explicitly 'fail' or 'emit_explicit'"
        )


def build_exploratory_epoch_plan_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_receipt_sha256: str,
    epoch_index: int,
    batch_size: int,
    partial_batch_policy: str,
) -> dict[str, Any]:
    """Build a deterministic training-only ST16 plan and explicit tail ledger."""

    _validate_engineering_batch_contract(batch_size, partial_batch_policy)
    if (
        isinstance(epoch_index, bool)
        or not isinstance(epoch_index, int)
        or epoch_index < 0
    ):
        raise ValueError("epoch_index must be nonnegative")
    if not isinstance(manifest_receipt_sha256, str) or len(manifest_receipt_sha256) != 64:
        raise ValueError("manifest receipt must be a SHA-256 string")

    tile_lookup: dict[str, dict[str, Any]] = {}
    pools: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"all": [], "positive": []}
    )
    spans_by_record: dict[str, list[list[int]]] = {}
    counts_by_patient: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    excluded_short: list[str] = []
    short_context_identities: list[str] = []
    eligible_record_count = 0
    for raw in records:
        record = dict(raw)
        if record.get("model_split") != "source_train":
            raise PermissionError("ST16 training plan accepts source_train only")
        identity = str(record["analysis_identity_id"])
        patient = str(record["patient_id"])
        relative_path = str(record["edf_relative_path"])
        sample_count = int(record["target_sample_count_256hz"])
        if not identity or not patient or sample_count < 1:
            raise ValueError("training record identity/count is malformed")
        spans = event_sample_spans(record)
        if sample_count < 2:
            excluded_short.append(identity)
            continue
        positive_count = sum(stop - start for start, stop in spans)
        counts_by_patient[patient][0] += positive_count
        counts_by_patient[patient][1] += sample_count - positive_count
        eligible_record_count += 1
        spans_by_record[identity] = spans
        if sample_count < TILE_SAMPLES:
            identifier = _tile_id(identity, 0)
            positive = any(event_start < sample_count for event_start, _ in spans)
            tile_lookup[identifier] = {
                "tile_id": identifier,
                "analysis_identity_id": identity,
                "patient_id": patient,
                "edf_relative_path": relative_path,
                "target_start_sample": 0,
                "target_stop_sample_exclusive": TILE_SAMPLES,
                "positive_tile": positive,
                "short_context_policy_id": st.ST16_SHORT_CONTEXT_POLICY_ID,
                "valid_support_sample_count": sample_count,
                "context_sample_count": TILE_SAMPLES - sample_count,
            }
            pools[patient]["all"].append(identifier)
            if positive:
                pools[patient]["positive"].append(identifier)
            short_context_identities.append(identity)
            continue
        for start in range(0, sample_count - TILE_SAMPLES + 1, TRAIN_HOP_SAMPLES):
            identifier = _tile_id(identity, start)
            positive = any(
                event_start < start + TILE_SAMPLES and event_stop > start
                for event_start, event_stop in spans
            )
            if identifier in tile_lookup:
                raise RuntimeError("ST16 tile identity collision")
            tile_lookup[identifier] = {
                "tile_id": identifier,
                "analysis_identity_id": identity,
                "patient_id": patient,
                "edf_relative_path": relative_path,
                "target_start_sample": start,
                "target_stop_sample_exclusive": start + TILE_SAMPLES,
                "positive_tile": positive,
            }
            pools[patient]["all"].append(identifier)
            if positive:
                pools[patient]["positive"].append(identifier)

    if not pools:
        raise ValueError("no source_train patient has an admitted ST16 training tile")
    missing_class_count_patients = sorted(set(counts_by_patient).difference(pools))
    class_weight = st.fit_patient_equal_class_weights_pure_primitive(
        {patient: tuple(counts_by_patient[patient]) for patient in sorted(pools)},
        fit_roster_sha256=_canonical_sha256(sorted(pools)),
    )
    native = st.build_patient_balanced_epoch_plan_pure_primitive(
        pools,
        variant_id=st.ST16_VARIANT_ID,
        outer_fold=0,
        stage="final_refit",
        epoch_index=epoch_index,
    )
    patient_count = int(native["patient_count"])
    native_batches_per_round = math.ceil(patient_count / 16)
    explicit_batches: list[list[dict[str, Any]]] = []
    partial_batch_sizes: list[int] = []
    selected_tile_ids: set[str] = set()
    for draw_index in range(TILES_PER_PATIENT_PER_EPOCH):
        native_round = native["batches"][
            draw_index * native_batches_per_round :
            (draw_index + 1) * native_batches_per_round
        ]
        rows = [dict(row) for batch in native_round for row in batch]
        if len(rows) != patient_count or len({row["patient_key"] for row in rows}) != patient_count:
            raise RuntimeError("native ST16 draw round lost or duplicated a patient")
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            if len(batch) < batch_size:
                partial_batch_sizes.append(len(batch))
                if partial_batch_policy == "fail":
                    raise ValueError(
                        "patient count is not divisible by batch_size; choose "
                        "partial_batch_policy='emit_explicit' to admit and ledger "
                        "the final partial batch"
                    )
            expanded = []
            for row in batch:
                tile_id = str(row["tile_id"])
                selected_tile_ids.add(tile_id)
                expanded.append(
                    {
                        "draw_index": draw_index,
                        "patient_key": str(row["patient_key"]),
                        "tile_id": tile_id,
                    }
                )
            explicit_batches.append(expanded)

    selected_catalog = {
        tile_id: tile_lookup[tile_id] for tile_id in sorted(selected_tile_ids)
    }
    selected_record_ids = sorted(
        {row["analysis_identity_id"] for row in selected_catalog.values()}
    )
    selected_spans = {
        identity: spans_by_record[identity] for identity in selected_record_ids
    }
    plan = _content_address(
        {
            "schema_version": "st16_common17_exploratory_epoch_plan_v1",
            "provider_id": PROVIDER_ID,
            "claim_status": "exploratory_nonpromotable_source_train_only",
            "variant_id": st.ST16_VARIANT_ID,
            "manifest_receipt_sha256": manifest_receipt_sha256,
            "epoch_index": epoch_index,
            "target_clock_hz": TARGET_FS_HZ,
            "tile_samples": TILE_SAMPLES,
            "train_hop_samples": TRAIN_HOP_SAMPLES,
            "patient_count": patient_count,
            "eligible_record_count": eligible_record_count,
            "excluded_short_record_count": len(excluded_short),
            "excluded_short_identity_roster_sha256": _canonical_sha256(
                sorted(excluded_short)
            ),
            "short_context_record_count": len(short_context_identities),
            "short_context_identity_roster_sha256": _canonical_sha256(
                sorted(short_context_identities)
            ),
            "short_context_policy_id": st.ST16_SHORT_CONTEXT_POLICY_ID,
            "patients_without_60_second_record_count": len(
                missing_class_count_patients
            ),
            "patients_without_60_second_record_roster_sha256": _canonical_sha256(
                missing_class_count_patients
            ),
            "all_tile_count": len(tile_lookup),
            "unique_selected_tile_count": len(selected_catalog),
            "unique_selected_record_count": len(selected_record_ids),
            "selected_tile_catalog": selected_catalog,
            "selected_record_positive_spans": selected_spans,
            "class_weight_receipt": class_weight,
            "batch_contract": {
                "batch_size": batch_size,
                "engineering_batch_size_maximum": ENGINEERING_BATCH_SIZE_MAXIMUM,
                "batch16_registry_maximum_memory_admitted": False,
                "batch16_nonadmission_reason": (
                    "not empirically admitted with concurrent Qwen vLLM using "
                    "approximately 40.6 GiB of the 48 GiB GPU"
                ),
                "partial_batch_policy": partial_batch_policy,
                "partial_batch_count": len(partial_batch_sizes),
                "partial_batch_sizes": partial_batch_sizes,
                "drop_last": False,
                "padding_or_patient_duplication": False,
                "one_tile_per_patient_per_batch": True,
                "batch_count": len(explicit_batches),
            },
            "batches": explicit_batches,
            "permissions": {
                "source_train_TERM_targets_used_for_training": True,
                "source_dev_targets_opened": False,
                "source_eval_opened": False,
                "EDF_annotations_used": False,
                "spreadsheet_doctor_text_history_video_behavior_used": False,
            },
            "receipt_sha256": _PENDING,
        }
    )
    return plan


def build_exploratory_epoch_plan(
    manifest_path: str | Path,
    *,
    epoch_index: int,
    batch_size: int,
    partial_batch_policy: str,
) -> dict[str, Any]:
    manifest = load_common17_manifest(manifest_path, require_complete=True)
    return build_exploratory_epoch_plan_from_records(
        [
            row
            for row in manifest["records"]
            if row["model_split"] == "source_train"
        ],
        manifest_receipt_sha256=str(manifest["receipt_sha256"]),
        epoch_index=epoch_index,
        batch_size=batch_size,
        partial_batch_policy=partial_batch_policy,
    )


def _copy_regular_file_content_verified(source: Path, target: Path) -> str:
    """Copy one immutable EDF once and verify the locally staged bytes.

    The source is hashed during the copy, so the network/source filesystem is
    not read a second time.  The local copy is then replay-hashed before use.
    Stable source inode/size/mtime observations bracket the copy.
    """

    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        for block in iter(lambda: input_handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            output_handle.write(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    after = source.stat()
    stable_fields_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    stable_fields_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stable_fields_before != stable_fields_after:
        raise RuntimeError("ST16 source EDF changed while it was staged")
    source_read_sha256 = digest.hexdigest()
    if (
        target.stat().st_size != before.st_size
        or _file_sha256(target) != source_read_sha256
    ):
        raise RuntimeError("ST16 local EDF staging failed byte replay")
    return source_read_sha256


def _transform_st16_record(edf_path: Path, registry: Mapping[str, Any]):
    def execute(source: Path):
        canonical = load_canonical_edf_record(source)
        lineage = authorize_detector_signal_lineage_from_canonical_record(canonical)
        referential_volts = np.asarray(
            canonical.observed_signal_volts.detach().cpu().numpy()
        )
        return st.apply_full_record_transform(
            referential_volts,
            variant_id=st.ST16_VARIANT_ID,
            signal_lineage_authority=lineage,
            registry=registry,
        )

    stage_root_text = os.environ.get("CLINICAL_EEG_ST16_LOCAL_STAGE_ROOT")
    if stage_root_text is None:
        transformed = execute(edf_path)
    else:
        stage_candidate = Path(stage_root_text)
        if stage_candidate.is_symlink():
            raise PermissionError("ST16 local EDF stage root may not be a symlink")
        stage_root = stage_candidate.resolve(strict=True)
        if not stage_root.is_dir():
            raise NotADirectoryError(stage_root)
        with tempfile.TemporaryDirectory(
            prefix="clinical_eeg_st16_edf_", dir=stage_root
        ) as temporary_directory:
            staged_path = Path(temporary_directory) / edf_path.name
            _copy_regular_file_content_verified(edf_path, staged_path)
            transformed = execute(staged_path)
    if (
        transformed.signal.dtype != np.dtype("float32")
        or transformed.signal.shape[0] != len(st.ST16_TYPED_UNITS)
        or tuple(transformed.receipt["output"]["typed_units"])
        != st.ST16_TYPED_UNITS
    ):
        raise RuntimeError("ST16 provider transform output drifted")
    return transformed


_CACHE_WORKER_REGISTRY: dict[str, Any] | None = None


def _cache_contract_core(plan: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = (
        Path(__file__).resolve().parents[2] / st.CONFIG_RELATIVE_PATH
    ).resolve(strict=True)
    return {
        "schema_version": CACHE_CONTRACT_SCHEMA_VERSION,
        "claim_status": "temporary_nonpromotable_training_cache",
        "provider_id": PROVIDER_ID,
        "variant_id": st.ST16_VARIANT_ID,
        "epoch_plan_receipt_sha256": str(plan["receipt_sha256"]),
        "target_clock_hz": TARGET_FS_HZ,
        "tile_samples": TILE_SAMPLES,
        "st16_typed_units": list(st.ST16_TYPED_UNITS),
        "transform_registry_relative_path": str(st.CONFIG_RELATIVE_PATH),
        "transform_registry_file_sha256": _file_sha256(registry_path),
    }


def _fresh_cache_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _content_address(
        {
            **_cache_contract_core(plan),
            "legacy_markerless_adoption": None,
            "receipt_sha256": _PENDING,
        }
    )


def _validate_cache_contract(
    contract: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    verified = _validate_content_address(
        contract, artifact_name="ST16 epoch-cache contract"
    )
    for key, expected in _cache_contract_core(plan).items():
        if verified.get(key) != expected:
            raise PermissionError(f"ST16 epoch-cache contract drifted at {key}")
    adoption = verified.get("legacy_markerless_adoption")
    if adoption is not None:
        if not isinstance(adoption, Mapping):
            raise PermissionError("ST16 legacy cache adoption binding is malformed")
        required = {
            "epoch_plan_receipt_sha256": (
                LEGACY_MARKERLESS_ADOPTION_PLAN_SHA256
            ),
            "local_staging_equivalence_receipt_sha256": (
                LEGACY_LOCAL_STAGING_EQUIVALENCE_RECEIPT_SHA256
            ),
            "adoption_status": "one_time_markerless_cache_adoption_passed",
            "adoption_receipt_file_name": "cache_adoption_receipt.json",
        }
        if any(adoption.get(key) != value for key, value in required.items()):
            raise PermissionError("ST16 legacy cache adoption is not authorized")
        adoption_receipt_sha256 = adoption.get("adoption_receipt_sha256")
        if (
            not isinstance(adoption_receipt_sha256, str)
            or len(adoption_receipt_sha256) != 64
            or verified["epoch_plan_receipt_sha256"]
            != LEGACY_MARKERLESS_ADOPTION_PLAN_SHA256
        ):
            raise PermissionError("ST16 legacy cache adoption lineage drifted")
    return verified


def _load_cache_contract(
    cache: Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    path = cache / "cache_contract.json"
    if path.is_symlink():
        raise PermissionError("ST16 epoch-cache contract may not be a symlink")
    if not path.is_file():
        # Markerless directories are never silently adopted.  A truly empty
        # newly-created cache gets a fresh contract; the one historical live
        # cache has a separate, explicit audit/adoption command.
        entries = list(cache.iterdir())
        if entries:
            raise PermissionError(
                "markerless ST16 epoch cache is not reusable; run the explicit "
                "one-time adoption audit only for the authorized 2026-08-25 cache"
            )
        expected = _fresh_cache_contract(plan)
        try:
            _write_json_atomic(path, expected, replace=False)
        except FileExistsError:
            # Another same-run producer may have published the contract first.
            pass
    if not path.is_file() or path.is_symlink():
        raise PermissionError("ST16 epoch-cache contract publication failed")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    verified = _validate_cache_contract(loaded, plan=plan)
    adoption = verified.get("legacy_markerless_adoption")
    if adoption is not None:
        receipt_name = adoption.get("adoption_receipt_file_name")
        if receipt_name != "cache_adoption_receipt.json":
            raise PermissionError("ST16 cache adoption receipt name drifted")
        adoption_path = cache / receipt_name
        if adoption_path.is_symlink() or not adoption_path.is_file():
            raise PermissionError("ST16 cache adoption receipt is unavailable")
        adoption_receipt = _validate_content_address(
            json.loads(adoption_path.read_text(encoding="utf-8")),
            artifact_name="ST16 markerless-cache adoption receipt",
        )
        if (
            adoption_receipt["receipt_sha256"]
            != adoption["adoption_receipt_sha256"]
            or adoption_receipt.get("epoch_plan_receipt_sha256")
            != verified["epoch_plan_receipt_sha256"]
            or adoption_receipt.get(
                "local_staging_equivalence_receipt_sha256"
            )
            != LEGACY_LOCAL_STAGING_EQUIVALENCE_RECEIPT_SHA256
        ):
            raise PermissionError("ST16 cache adoption receipt lineage drifted")
    return verified


def _cache_tile_sidecar(
    row: Mapping[str, Any],
    *,
    target: Path,
    cache_contract_receipt_sha256: str,
    transform_receipt: Mapping[str, Any] | None = None,
    npy_size_bytes: int | None = None,
    npy_sha256: str | None = None,
) -> dict[str, Any]:
    start = int(row["target_start_sample"])
    observed_size = target.stat().st_size
    if npy_size_bytes is not None and npy_size_bytes != observed_size:
        raise ValueError("ST16 cache tile size changed before sidecar publication")
    observed_sha256 = _file_sha256(target) if npy_sha256 is None else npy_sha256
    if not isinstance(observed_sha256, str) or len(observed_sha256) != 64:
        raise ValueError("ST16 cache tile SHA-256 is malformed")
    body: dict[str, Any] = {
            "schema_version": CACHE_TILE_SCHEMA_VERSION,
            "claim_status": "temporary_nonpromotable_training_cache_tile",
            "cache_contract_receipt_sha256": cache_contract_receipt_sha256,
            "tile_id": str(row["tile_id"]),
            "analysis_identity_id": str(row["analysis_identity_id"]),
            "edf_relative_path": str(row["edf_relative_path"]),
            "target_start_sample": start,
            "target_stop_sample_exclusive": start + TILE_SAMPLES,
            "shape": [len(st.ST16_TYPED_UNITS), TILE_SAMPLES],
            "dtype": "float32",
            "npy_file_name": target.name,
            "npy_size_bytes": observed_size,
            "npy_sha256": observed_sha256,
            "receipt_sha256": _PENDING,
    }
    valid_support = row.get("valid_support_sample_count")
    if valid_support is not None:
        if transform_receipt is None:
            raise ValueError("short ST16 cache tile lacks transform ledger")
        ledger = transform_receipt.get("short_record_context")
        if (
            not isinstance(ledger, Mapping)
            or ledger.get("policy_id") != st.ST16_SHORT_CONTEXT_POLICY_ID
            or ledger.get("provider_observed_sample_count") != valid_support
            or ledger.get("context_may_receive_target_loss_or_metric_weight")
            is not False
        ):
            raise PermissionError("short ST16 transform ledger drifted")
        body.update(
            {
                "short_context_policy_id": st.ST16_SHORT_CONTEXT_POLICY_ID,
                "valid_support_sample_count": int(valid_support),
                "context_sample_count": TILE_SAMPLES - int(valid_support),
                "short_context_ledger": deepcopy(dict(ledger)),
                "valid_support_mask_payload_receipt": deepcopy(
                    dict(ledger["valid_support_mask_payload_receipt"])
                ),
                "padding_may_receive_target_loss_or_metric_weight": False,
            }
        )
    return _content_address(body)


def _validate_cached_tile(
    row: Mapping[str, Any],
    *,
    target: Path,
    cache_contract_receipt_sha256: str,
) -> dict[str, Any] | None:
    sidecar = target.with_suffix(".json")
    if target.is_symlink() or sidecar.is_symlink():
        raise PermissionError("ST16 epoch cache may not contain symlinks")
    if not target.is_file() or not sidecar.is_file():
        return None
    try:
        metadata = _validate_content_address(
            json.loads(sidecar.read_text(encoding="utf-8")),
            artifact_name=f"ST16 cache tile {row['tile_id']}",
        )
        start = int(row["target_start_sample"])
        required = {
            "schema_version": CACHE_TILE_SCHEMA_VERSION,
            "claim_status": "temporary_nonpromotable_training_cache_tile",
            "cache_contract_receipt_sha256": cache_contract_receipt_sha256,
            "tile_id": str(row["tile_id"]),
            "analysis_identity_id": str(row["analysis_identity_id"]),
            "edf_relative_path": str(row["edf_relative_path"]),
            "target_start_sample": start,
            "target_stop_sample_exclusive": start + TILE_SAMPLES,
            "shape": [len(st.ST16_TYPED_UNITS), TILE_SAMPLES],
            "dtype": "float32",
            "npy_file_name": target.name,
            "npy_size_bytes": target.stat().st_size,
            "npy_sha256": _file_sha256(target),
        }
        valid_support = row.get("valid_support_sample_count")
        if valid_support is not None:
            ledger = metadata.get("short_context_ledger")
            required.update(
                {
                    "short_context_policy_id": st.ST16_SHORT_CONTEXT_POLICY_ID,
                    "valid_support_sample_count": int(valid_support),
                    "context_sample_count": TILE_SAMPLES - int(valid_support),
                    "valid_support_mask_payload_receipt": (
                        ledger.get("valid_support_mask_payload_receipt")
                        if isinstance(ledger, Mapping)
                        else None
                    ),
                    "padding_may_receive_target_loss_or_metric_weight": False,
                }
            )
        if any(metadata.get(key) != value for key, value in required.items()):
            return None
        if valid_support is not None:
            ledger = metadata.get("short_context_ledger")
            pending = deepcopy(dict(ledger)) if isinstance(ledger, Mapping) else {}
            supplied = pending.get("receipt_sha256")
            pending["receipt_sha256"] = _PENDING
            if (
                supplied != _canonical_sha256(pending)
                or ledger.get("policy_id") != st.ST16_SHORT_CONTEXT_POLICY_ID
                or ledger.get("provider_observed_sample_count") != valid_support
                or ledger.get("context_may_receive_target_loss_or_metric_weight")
                is not False
            ):
                return None
        value = np.load(target, mmap_mode="r", allow_pickle=False)
        valid = bool(
            value.shape == (len(st.ST16_TYPED_UNITS), TILE_SAMPLES)
            and value.dtype == np.dtype("float32")
            and value.nbytes
            == len(st.ST16_TYPED_UNITS) * TILE_SAMPLES * 4
        )
        del value
        return metadata if valid else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _materialize_epoch_record_worker(
    payload: tuple[str, str, str, list[dict[str, Any]], str],
) -> dict[str, Any]:
    """Transform one EDF and atomically persist its requested unique tiles."""

    global _CACHE_WORKER_REGISTRY
    (
        tusz_root_text,
        cache_root_text,
        identity,
        raw_rows,
        cache_contract_receipt_sha256,
    ) = payload
    tusz_root = Path(tusz_root_text).resolve(strict=True)
    cache_root = Path(cache_root_text).resolve(strict=True)
    rows = sorted(raw_rows, key=lambda row: int(row["target_start_sample"]))
    if not rows or any(
        str(row["analysis_identity_id"]) != identity
        or str(row["edf_relative_path"])
        != str(rows[0]["edf_relative_path"])
        for row in rows
    ):
        raise PermissionError("ST16 cache worker payload crossed record lineage")
    if _CACHE_WORKER_REGISTRY is None:
        _CACHE_WORKER_REGISTRY = st.load_registry(
            Path(__file__).resolve().parents[2] / st.CONFIG_RELATIVE_PATH
        )
    edf_path = _safe_edf(
        tusz_root,
        str(rows[0]["edf_relative_path"]),
        expected_split="source_train",
    )
    transformed = _transform_st16_record(edf_path, _CACHE_WORKER_REGISTRY)
    outputs: list[dict[str, Any]] = []
    for row in rows:
        start = int(row["target_start_sample"])
        stop = start + TILE_SAMPLES
        if stop > transformed.signal.shape[1]:
            raise RuntimeError("selected ST16 tile exceeds transformed support")
        tile = np.ascontiguousarray(transformed.signal[:, start:stop], dtype="<f4")
        target = cache_root / f"{row['tile_id']}.npy"
        # Invalid/interrupted files are deliberately replaceable.  Publication
        # of the matching content-addressed sidecar happens only after the NPY
        # is durable, so every crash point is detectable on the next resume.
        _save_numpy_atomic(target, tile, replace=True)
        sidecar = _cache_tile_sidecar(
            row,
            target=target,
            cache_contract_receipt_sha256=cache_contract_receipt_sha256,
            transform_receipt=getattr(transformed, "receipt", None),
        )
        _write_json_atomic(target.with_suffix(".json"), sidecar, replace=True)
        outputs.append(
            {
                "tile_id": str(row["tile_id"]),
                "path": str(target),
                "size_bytes": target.stat().st_size,
                "npy_sha256": sidecar["npy_sha256"],
                "tile_receipt_sha256": sidecar["receipt_sha256"],
            }
        )
    del transformed
    return {"analysis_identity_id": identity, "tiles": outputs}


def materialize_epoch_tile_cache(
    plan: Mapping[str, Any],
    *,
    tusz_root: str | Path,
    cache_root: str | Path,
    selected_tile_ids: Sequence[str] | None = None,
    progress_every_records: int | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Materialize unique selected tiles and reuse validated interrupted output."""

    root = Path(tusz_root).resolve(strict=True)
    cache = Path(cache_root).resolve(strict=True)
    full_catalog = dict(plan["selected_tile_catalog"])
    requested = (
        set(full_catalog)
        if selected_tile_ids is None
        else {str(value) for value in selected_tile_ids}
    )
    if not requested or not requested.issubset(full_catalog):
        raise ValueError("requested epoch-cache tiles are empty or outside the plan")
    cache_contract = _load_cache_contract(cache, plan=plan)
    cache_contract_sha256 = str(cache_contract["receipt_sha256"])
    catalog = {tile_id: full_catalog[tile_id] for tile_id in sorted(requested)}
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog.values():
        by_record[str(row["analysis_identity_id"])].append(dict(row))
    paths: dict[str, Path] = {}
    began = time.perf_counter()
    materialized_bytes = 0
    reused_tile_count = 0
    reused_content_verified_tile_count = 0
    invalid_existing_tile_count = 0
    missing_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for identity, rows in by_record.items():
        for row in rows:
            target = cache / f"{row['tile_id']}.npy"
            sidecar = target.with_suffix(".json")
            for candidate in (target, sidecar):
                if candidate.is_symlink():
                    raise PermissionError("ST16 epoch cache may not contain symlinks")
                if candidate.exists() and not candidate.is_file():
                    raise PermissionError(
                        "ST16 epoch cache tile paths must be regular files"
                    )
            metadata = _validate_cached_tile(
                row,
                target=target,
                cache_contract_receipt_sha256=cache_contract_sha256,
            )
            if metadata is not None:
                paths[str(row["tile_id"])] = target
                reused_tile_count += 1
                reused_content_verified_tile_count += 1
                materialized_bytes += target.stat().st_size
            else:
                if target.exists() or sidecar.exists():
                    invalid_existing_tile_count += 1
                missing_by_record[identity].append(row)
    worker_text = os.environ.get("CLINICAL_EEG_ST16_CACHE_WORKERS", "1")
    try:
        cache_worker_count = int(worker_text)
    except ValueError as exc:
        raise ValueError("ST16 cache worker count must be an integer") from exc
    if not 1 <= cache_worker_count <= 4:
        raise ValueError("ST16 cache worker count must be in [1, 4]")
    record_identities = sorted(missing_by_record)
    tasks = [
        (
            str(root),
            str(cache),
            identity,
            [dict(row) for row in missing_by_record[identity]],
            cache_contract_sha256,
        )
        for identity in record_identities
    ]
    if not tasks:
        iterator = iter(())
        executor = None
    elif cache_worker_count == 1:
        iterator = map(_materialize_epoch_record_worker, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=cache_worker_count,
            mp_context=mp.get_context("spawn"),
        )
        iterator = executor.map(_materialize_epoch_record_worker, tasks, chunksize=1)
    try:
        for record_index, result in enumerate(iterator, start=1):
            identity = str(result["analysis_identity_id"])
            if identity != record_identities[record_index - 1]:
                raise RuntimeError("parallel ST16 cache result order drifted")
            for row in result["tiles"]:
                target = Path(str(row["path"])).resolve(strict=True)
                tile_id = str(row["tile_id"])
                expected_target = (cache / f"{tile_id}.npy").resolve(strict=True)
                if target != expected_target or tile_id not in catalog:
                    raise PermissionError("ST16 cache worker output escaped its plan")
                verified = _validate_cached_tile(
                    catalog[tile_id],
                    target=target,
                    cache_contract_receipt_sha256=cache_contract_sha256,
                )
                if (
                    verified is None
                    or verified["npy_sha256"] != row["npy_sha256"]
                    or verified["receipt_sha256"]
                    != row["tile_receipt_sha256"]
                ):
                    raise RuntimeError(
                        "ST16 cache worker output failed parent-process replay"
                    )
                paths[tile_id] = target
                materialized_bytes += int(row["size_bytes"])
            if (
                progress_every_records is not None
                and progress_every_records > 0
                and (
                    record_index % progress_every_records == 0
                    or record_index == len(record_identities)
                )
            ):
                print(
                    json.dumps(
                        {
                            "stage": "st16_epoch_cache_progress",
                            "record_count_completed": record_index,
                            "record_count_total": len(record_identities),
                            "tile_count_materialized": len(paths),
                            "tile_count_reused": reused_tile_count,
                            "elapsed_seconds": time.perf_counter() - began,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if set(paths) != set(catalog):
        raise RuntimeError("ST16 epoch cache is incomplete")
    receipt = _content_address(
        {
            "schema_version": "st16_common17_exploratory_epoch_cache_v1",
            "claim_status": "temporary_nonpromotable_training_cache",
            "epoch_plan_receipt_sha256": plan["receipt_sha256"],
            "cache_contract_receipt_sha256": cache_contract_sha256,
            "unique_tile_count": len(paths),
            "full_epoch_unique_tile_count": len(full_catalog),
            "complete_full_epoch_cache": set(paths) == set(full_catalog),
            "requested_unique_record_count": len(by_record),
            "unique_record_transform_count": len(missing_by_record),
            "cache_worker_count": cache_worker_count,
            "reused_valid_tile_count": reused_tile_count,
            "reused_content_verified_tile_count": (
                reused_content_verified_tile_count
            ),
            "invalid_existing_tile_count": invalid_existing_tile_count,
            "materialized_bytes": materialized_bytes,
            "float32_exact_provider_tiles": True,
            "every_reused_tile_bound_by_content_address": True,
            "same_shape_numeric_corruption_is_rejected": True,
            "local_EDF_stage_root": os.environ.get(
                "CLINICAL_EEG_ST16_LOCAL_STAGE_ROOT"
            ),
            "local_EDF_staging_changes_numeric_transform": False,
            "cache_retained_across_interruption": True,
            "cache_deleted_only_after_completed_epoch": True,
            "wall_seconds": time.perf_counter() - began,
            "receipt_sha256": _PENDING,
        }
    )
    return paths, receipt


def _weighted_dense_bce(
    probabilities: Tensor,
    targets: Tensor,
    *,
    positive_weight: float,
    observed_masks: Tensor | None = None,
) -> Tensor:
    probability = probabilities.float().clamp(1e-7, 1.0 - 1e-7)
    target = targets.float()
    per_sample = -(
        positive_weight * target * torch.log(probability)
        + (1.0 - target) * torch.log1p(-probability)
    )
    weights = torch.where(target == 1, positive_weight, 1.0)
    if observed_masks is not None:
        mask = observed_masks.to(device=probability.device)
        if (
            mask.shape != target.shape
            or not bool(torch.all((mask == 0) | (mask == 1)))
            or not bool(torch.all(mask.sum(dim=1) > 0))
            or not bool(torch.all(target[mask == 0] == 0))
        ):
            raise ValueError("ST16 observed-support mask is malformed")
        # Preserve the exact native-tile arithmetic path bitwise.
        if not bool(torch.all(mask == 1)):
            valid = mask.float()
            return (
                (per_sample * valid).sum(dim=1)
                / (weights * valid).sum(dim=1)
            ).mean()
    return (per_sample.sum(dim=1) / weights.sum(dim=1)).mean()


def _atomic_torch_save(path: Path, value: object) -> None:
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
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


def train_exploratory_st16(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    partial_batch_policy: str,
    device_name: str,
    learning_rate: float = 1e-4,
    weight_decay: float = 2e-5,
    maximum_steps: int | None = None,
    resume: bool = False,
    checkpoint_every_batches: int = 25,
) -> dict[str, Any]:
    """Run scratch ST16 training with exact batch-boundary recovery.

    ``maximum_steps`` limits only the current invocation.  A partial checkpoint
    is always written before returning and is rejected by prediction.  Resume
    replays the exact manifest, hyperparameters, epoch plan, optimizer, and
    Torch CPU/CUDA RNG state before continuing at ``next_batch``.
    """

    _validate_engineering_batch_contract(batch_size, partial_batch_policy)
    if (
        epochs < 1
        or learning_rate <= 0
        or weight_decay < 0
        or isinstance(checkpoint_every_batches, bool)
        or not isinstance(checkpoint_every_batches, int)
        or checkpoint_every_batches < 1
    ):
        raise ValueError("training hyperparameters are invalid")
    if maximum_steps is not None and maximum_steps < 1:
        raise ValueError("maximum_steps must be positive when supplied")
    manifest_source = Path(manifest_path).resolve(strict=True)
    manifest = load_common17_manifest(manifest_source, require_complete=True)
    tusz_root = Path(manifest["source_bindings"]["tusz_root"]).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path = output / "last.pt"

    seed = st.derive_training_seed(
        variant_id=st.ST16_VARIANT_ID, outer_fold=0, stage="final_refit"
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("ST16 CUDA training requires BF16-capable CUDA")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        torch.cuda.manual_seed_all(seed)
        device_free_before, device_total_memory = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device_free_before = None
        device_total_memory = None
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SeizureTransformer(
        in_channels=16,
        in_samples=TILE_SAMPLES,
        dim_feedforward=2048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    ).to(device)
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=False,
    )
    training_config = {
        "manifest_receipt_sha256": manifest["receipt_sha256"],
        "variant_id": st.ST16_VARIANT_ID,
        "epochs": epochs,
        "batch_size": batch_size,
        "partial_batch_policy": partial_batch_policy,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "checkpoint_every_batches": checkpoint_every_batches,
        "device_type": device.type,
        "precision": (
            "CUDA_bfloat16_autocast_forward_float32_loss"
            if device.type == "cuda"
            else "CPU_float32"
        ),
    }
    global_step = 0
    start_epoch = 0
    start_batch = 0
    completed_history: list[dict[str, Any]] = []
    current_accumulator: dict[str, Any] | None = None
    if resume_checkpoint_path.exists() and not resume:
        raise FileExistsError(
            "existing ST16 resume checkpoint requires resume=True"
        )
    if resume:
        if not resume_checkpoint_path.is_file() or resume_checkpoint_path.is_symlink():
            raise FileNotFoundError(
                "resume=True requires a regular output_dir/last.pt checkpoint"
            )
        loaded = torch.load(
            resume_checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            not isinstance(loaded, Mapping)
            or loaded.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or loaded.get("provider_id") != PROVIDER_ID
            or loaded.get("claim_status") != "exploratory_nonpromotable"
            or loaded.get("training_config") != training_config
            or loaded.get("source_eval_opened") is not False
            or loaded.get("architecture_promotable") is not False
        ):
            raise PermissionError("ST16 resume checkpoint lineage drifted")
        if loaded.get("training_complete") is True:
            raise RuntimeError("ST16 requested training is already complete")
        start_epoch = int(loaded["next_epoch"])
        start_batch = int(loaded["next_batch"])
        global_step = int(loaded["global_step"])
        if not 0 <= start_epoch < epochs or start_batch < 0:
            raise ValueError("ST16 resume cursor is outside requested training")
        model.load_state_dict(loaded["model_state"], strict=True)
        optimizer.load_state_dict(loaded["optimizer_state"])
        completed_history = list(loaded.get("completed_epoch_history", []))
        raw_accumulator = loaded.get("current_epoch_accumulator")
        current_accumulator = (
            None if raw_accumulator is None else dict(raw_accumulator)
        )
        torch.set_rng_state(loaded["torch_cpu_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(loaded["torch_cuda_rng_state_all"])
        print(
            json.dumps(
                {
                    "stage": "st16_resume_admitted",
                    "next_epoch": start_epoch,
                    "next_batch": start_batch,
                    "global_step": global_step,
                    "checkpoint_sha256": _file_sha256(resume_checkpoint_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def checkpoint_payload(
        *,
        next_epoch: int,
        next_batch: int,
        accumulator: Mapping[str, Any] | None,
        plan_receipt_sha256: str,
        training_complete: bool,
    ) -> dict[str, Any]:
        completed_epoch_count = len(completed_history)
        at_epoch_boundary = next_batch == 0 and accumulator is None
        inference_eligible = bool(
            at_epoch_boundary and completed_epoch_count >= 1
        )
        role = (
            "completed_epoch_inference_eligible"
            if inference_eligible
            else "partial_epoch_resume_only"
        )
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "claim_status": "exploratory_nonpromotable",
            "checkpoint_role": role,
            "variant_id": st.ST16_VARIANT_ID,
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "common17_channel_order": list(COMMON17_CHANNEL_ORDER),
            "st16_typed_units": list(st.ST16_TYPED_UNITS),
            "source_eval_opened": False,
            "training_config": training_config,
            "epoch_plan_receipt_sha256": plan_receipt_sha256,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "completed_epoch_count": completed_epoch_count,
            "completed_epoch_history": completed_history,
            "current_epoch_accumulator": (
                None if accumulator is None else dict(accumulator)
            ),
            "training_complete": training_complete,
            "inference_eligible": inference_eligible,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            ),
            "pickle_container_but_weights_only_true_loader_required": True,
            "architecture_promotable": False,
            "numeric_execution_promotable": False,
        }

    completed_checkpoint: Path | None = None
    stopped_early = False
    invocation_steps = 0
    began = time.perf_counter()
    cache_parent = output / "temporary_epoch_cache"
    cache_parent.mkdir(parents=True, exist_ok=True)
    last_plan_receipt_sha256 = "0" * 64
    for epoch_index in range(start_epoch, epochs):
        plan = build_exploratory_epoch_plan_from_records(
            [
                row
                for row in manifest["records"]
                if row["model_split"] == "source_train"
            ],
            manifest_receipt_sha256=str(manifest["receipt_sha256"]),
            epoch_index=epoch_index,
            batch_size=batch_size,
            partial_batch_policy=partial_batch_policy,
        )
        epoch_began = time.perf_counter()
        planned_batches = list(plan["batches"])
        last_plan_receipt_sha256 = str(plan["receipt_sha256"])
        batch_cursor = start_batch if epoch_index == start_epoch else 0
        if not 0 <= batch_cursor < len(planned_batches):
            raise ValueError("ST16 resume batch cursor is outside the epoch plan")
        if current_accumulator is not None:
            if (
                int(current_accumulator.get("epoch_index", -1)) != epoch_index
                or int(current_accumulator.get("completed_batch_count", -1))
                != batch_cursor
                or current_accumulator.get("epoch_plan_receipt_sha256")
                != plan["receipt_sha256"]
            ):
                raise PermissionError("ST16 partial-epoch accumulator drifted")
            loss_sum = float(current_accumulator["loss_sum"])
            gradient_norm_sum = float(current_accumulator["gradient_norm_sum"])
            last_gradient_norm = current_accumulator.get("last_gradient_norm")
            epoch_elapsed_before = float(
                current_accumulator.get("elapsed_seconds", 0.0)
            )
        else:
            if batch_cursor != 0:
                raise ValueError("nonzero ST16 batch cursor lacks an accumulator")
            loss_sum = 0.0
            gradient_norm_sum = 0.0
            last_gradient_norm = None
            epoch_elapsed_before = 0.0
        available_batches = planned_batches[batch_cursor:]
        if maximum_steps is None:
            execution_batches = available_batches
        else:
            remaining_steps = maximum_steps - invocation_steps
            if remaining_steps <= 0:
                stopped_early = True
                break
            execution_batches = available_batches[:remaining_steps]
        required_tile_ids = sorted(
            {
                str(draw["tile_id"])
                for batch in execution_batches
                for draw in batch
            }
        )
        cache_candidates = sorted(
            path
            for path in cache_parent.iterdir()
            if path.name.startswith(f"epoch_{epoch_index:04d}_")
            and path.is_dir()
            and not path.is_symlink()
        )
        if len(cache_candidates) > 1:
            raise RuntimeError(
                "multiple owned ST16 epoch caches exist; recovery is ambiguous"
            )
        if cache_candidates:
            temporary_cache = cache_candidates[0]
        else:
            temporary_cache = cache_parent / f"epoch_{epoch_index:04d}_recoverable"
            temporary_cache.mkdir(parents=False, exist_ok=False)
        cache_paths, cache_receipt = materialize_epoch_tile_cache(
            plan,
            tusz_root=tusz_root,
            cache_root=temporary_cache,
            selected_tile_ids=required_tile_ids,
            progress_every_records=25,
        )
        catalog = plan["selected_tile_catalog"]
        spans = plan["selected_record_positive_spans"]
        positive_weight = float(
            plan["class_weight_receipt"]["positive_weight"]
        )
        model.train()
        for local_batch_index, batch in enumerate(execution_batches):
            absolute_batch_index = batch_cursor + local_batch_index
            signals: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            observed_masks: list[np.ndarray] = []
            patient_keys: list[str] = []
            for draw in batch:
                tile = catalog[draw["tile_id"]]
                signal = np.load(
                    cache_paths[draw["tile_id"]], mmap_mode="r", allow_pickle=False
                )
                signals.append(np.asarray(signal, dtype=np.float32))
                target, mask, _ = st.build_seizuretransformer_dense_target_pure_primitive(
                    spans[tile["analysis_identity_id"]],
                    target_start_sample=int(tile["target_start_sample"]),
                    valid_support_sample_count=int(
                        tile.get("valid_support_sample_count", TILE_SAMPLES)
                    ),
                )
                if np.any(target[mask == 0] != 0):
                    raise RuntimeError("ST16 context acquired a positive target")
                targets.append(np.asarray(target, dtype=np.float32))
                observed_masks.append(np.asarray(mask, dtype=np.float32))
                patient_keys.append(str(draw["patient_key"]))
            if len(patient_keys) != len(set(patient_keys)):
                raise RuntimeError("ST16 batch repeats a patient")
            inputs = torch.from_numpy(np.stack(signals)).to(device)
            target_tensor = torch.from_numpy(np.stack(targets)).to(device)
            observed_mask_tensor = torch.from_numpy(np.stack(observed_masks)).to(device)
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                probability = model(inputs)
            loss = _weighted_dense_bce(
                probability,
                target_tensor,
                positive_weight=positive_weight,
                observed_masks=observed_mask_tensor,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("ST16 training loss is nonfinite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            global_step += 1
            invocation_steps += 1
            loss_value = float(loss.detach().cpu())
            gradient_value = float(gradient_norm.detach().cpu())
            loss_sum += loss_value
            gradient_norm_sum += gradient_value
            last_gradient_norm = gradient_value
            completed_batch_count = absolute_batch_index + 1
            accumulator = {
                "epoch_index": epoch_index,
                "completed_batch_count": completed_batch_count,
                "planned_batch_count": len(planned_batches),
                "loss_sum": loss_sum,
                "gradient_norm_sum": gradient_norm_sum,
                "last_gradient_norm": last_gradient_norm,
                "epoch_plan_receipt_sha256": plan["receipt_sha256"],
                "elapsed_seconds": (
                    epoch_elapsed_before + time.perf_counter() - epoch_began
                ),
                "latest_cache_receipt": cache_receipt,
            }
            should_checkpoint = (
                completed_batch_count % checkpoint_every_batches == 0
                or completed_batch_count == len(planned_batches)
                or (
                    maximum_steps is not None
                    and invocation_steps >= maximum_steps
                )
            )
            if should_checkpoint:
                partial = checkpoint_payload(
                    next_epoch=epoch_index,
                    next_batch=completed_batch_count,
                    accumulator=accumulator,
                    plan_receipt_sha256=plan["receipt_sha256"],
                    training_complete=False,
                )
                _atomic_torch_save(resume_checkpoint_path, partial)
                print(
                    json.dumps(
                        {
                            "stage": "st16_train_checkpoint",
                            "epoch_index": epoch_index,
                            "completed_batch_count": completed_batch_count,
                            "planned_batch_count": len(planned_batches),
                            "global_step": global_step,
                            "mean_loss": loss_sum / completed_batch_count,
                            "checkpoint_sha256": _file_sha256(
                                resume_checkpoint_path
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        completed_batch_count = batch_cursor + len(execution_batches)
        epoch_complete = completed_batch_count == len(planned_batches)
        if epoch_complete:
            shutil.rmtree(temporary_cache)
        if not epoch_complete:
            stopped_early = True
            current_accumulator = accumulator
            break
        epoch_elapsed = epoch_elapsed_before + time.perf_counter() - epoch_began
        completed_history.append(
            {
                "epoch_index": epoch_index,
                "completed_batch_count": completed_batch_count,
                "planned_batch_count": len(planned_batches),
                "epoch_complete": True,
                "mean_loss": loss_sum / completed_batch_count,
                "mean_preclip_gradient_L2_norm": (
                    gradient_norm_sum / completed_batch_count
                ),
                "last_preclip_gradient_L2_norm": last_gradient_norm,
                "epoch_plan_receipt_sha256": plan["receipt_sha256"],
                "cache_receipt": cache_receipt,
                "elapsed_seconds": epoch_elapsed,
            }
        )
        current_accumulator = None
        training_complete = epoch_index + 1 == epochs
        checkpoint = checkpoint_payload(
            next_epoch=epoch_index + 1,
            next_batch=0,
            accumulator=None,
            plan_receipt_sha256=plan["receipt_sha256"],
            training_complete=training_complete,
        )
        completed_checkpoint = output / f"epoch_{epoch_index:04d}.pt"
        _atomic_torch_save(completed_checkpoint, checkpoint)
        _atomic_torch_save(resume_checkpoint_path, checkpoint)
        print(
            json.dumps(
                {
                    "stage": "st16_epoch_complete_checkpoint",
                    "epoch_index": epoch_index,
                    "global_step": global_step,
                    "checkpoint_path": str(completed_checkpoint),
                    "checkpoint_sha256": _file_sha256(completed_checkpoint),
                    "training_complete": training_complete,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        start_batch = 0
        if maximum_steps is not None and invocation_steps >= maximum_steps:
            stopped_early = epoch_index + 1 < epochs
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        device_free_after, _ = torch.cuda.mem_get_info(device)
        numeric_resource_receipt = {
            "device": str(device),
            "precision": "CUDA_bfloat16_autocast_forward_float32_loss",
            "device_free_before_MiB": device_free_before / 1048576.0,
            "device_free_after_MiB": device_free_after / 1048576.0,
            "device_total_MiB": device_total_memory / 1048576.0,
            "process_peak_allocated_MiB": (
                torch.cuda.max_memory_allocated(device) / 1048576.0
            ),
            "process_peak_reserved_MiB": (
                torch.cuda.max_memory_reserved(device) / 1048576.0
            ),
        }
    else:
        numeric_resource_receipt = {
            "device": str(device),
            "precision": "CPU_float32",
            "device_free_before_MiB": None,
            "device_free_after_MiB": None,
            "device_total_MiB": None,
            "process_peak_allocated_MiB": None,
            "process_peak_reserved_MiB": None,
        }
    receipt = _content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "training_invocation",
            "status": (
                "step_limited_with_partial_resume_checkpoint"
                if stopped_early and completed_checkpoint is None
                else (
                    "step_limited_after_completed_epoch_checkpoint"
                    if stopped_early
                    else "completed_requested_exploratory_epochs"
                )
            ),
            "claim_status": "exploratory_nonpromotable",
            "manifest_path": str(manifest_source),
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "source_train_only": True,
            "source_dev_or_source_eval_opened": False,
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "partial_batch_policy": partial_batch_policy,
            "maximum_steps": maximum_steps,
            "resume_requested": resume,
            "checkpoint_every_batches": checkpoint_every_batches,
            "invocation_steps": invocation_steps,
            "global_step": global_step,
            "resume_checkpoint_path": str(resume_checkpoint_path),
            "resume_checkpoint_exists": resume_checkpoint_path.is_file(),
            "resume_checkpoint_sha256": (
                _file_sha256(resume_checkpoint_path)
                if resume_checkpoint_path.is_file()
                else None
            ),
            "completed_checkpoint_path": (
                str(completed_checkpoint) if completed_checkpoint else None
            ),
            "completed_checkpoint_sha256": (
                _file_sha256(completed_checkpoint) if completed_checkpoint else None
            ),
            "completed_epoch_history": completed_history,
            "current_epoch_accumulator": current_accumulator,
            "numeric_resource_receipt": numeric_resource_receipt,
            "wall_seconds": time.perf_counter() - began,
            "formal_executor_used": False,
            "architecture_promotable": False,
            "numeric_execution_promotable": False,
            "clinical_use_authorized": False,
            "receipt_sha256": _PENDING,
        }
    )
    _write_json_atomic(output / "training_receipt.json", receipt, replace=True)
    return receipt


def _load_exploratory_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[SeizureTransformer, dict[str, Any], str]:
    source = Path(path).resolve(strict=True)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("ST16 checkpoint must be a mapping")
    required = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "claim_status": "exploratory_nonpromotable",
        "variant_id": st.ST16_VARIANT_ID,
        "common17_channel_order": list(COMMON17_CHANNEL_ORDER),
        "st16_typed_units": list(st.ST16_TYPED_UNITS),
        "source_eval_opened": False,
        "architecture_promotable": False,
        "checkpoint_role": "completed_epoch_inference_eligible",
        "inference_eligible": True,
    }
    if any(checkpoint.get(key) != value for key, value in required.items()):
        raise PermissionError("checkpoint is not the bound nonpromotable ST16 artifact")
    if (
        int(checkpoint.get("completed_epoch_count", 0)) < 1
        or int(checkpoint.get("next_batch", -1)) != 0
        or checkpoint.get("current_epoch_accumulator") is not None
    ):
        raise PermissionError("partial ST16 resume checkpoint cannot run prediction")
    model = SeizureTransformer(
        in_channels=16,
        in_samples=TILE_SAMPLES,
        dim_feedforward=2048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, dict(checkpoint), _file_sha256(source)


def select_target_free_prediction_rows(
    projection: Mapping[str, Any], *, split: str
) -> list[dict[str, Any]]:
    """Select prediction rows while fail-closing source-eval before I/O."""

    if split == "source_eval":
        raise PermissionError(
            "source_eval is locked; ST16 checkpoint/decoder/operating points must "
            "be frozen before a separately authorized one-time evaluation"
        )
    if split != "source_dev":
        raise PermissionError("exploratory ST16 prediction is source_dev-only")
    validated = validate_tusz_analysis_identity_projection_v2(dict(projection))
    forbidden = {
        "seizure_events",
        "reference_events",
        "reference_csv_bi_sha256",
        "annotation",
        "doctor_text",
        "excel",
    }
    rows: list[dict[str, Any]] = []
    for raw in validated["records"]:
        if forbidden.intersection(raw):
            raise PermissionError("target-free prediction row contains a forbidden field")
        if raw["model_split"] == split:
            rows.append(dict(raw))
    if not rows:
        raise ValueError("target-free source_dev projection is empty")
    rows.sort(key=lambda row: row["analysis_identity_id"])
    return rows


def _model_predictor(
    model: SeizureTransformer, device: torch.device
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(
            device
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            probability = model(inputs)
        return probability.float().cpu().numpy()

    return predict


def predict_transformed_record_dense(
    transformed: Any,
    *,
    predictor: Callable[[np.ndarray], object],
    inference_batch_size: int,
) -> SeizureTransformerStreamingResult:
    """Run complete OLA and return the lossless pre-threshold dense posterior."""

    if (
        isinstance(inference_batch_size, bool)
        or not isinstance(inference_batch_size, int)
        or not 1 <= inference_batch_size <= ENGINEERING_BATCH_SIZE_MAXIMUM
    ):
        raise ValueError("inference_batch_size must be explicit in [1, 8]")
    receipt = transformed.receipt
    reader = _ArrayTileReader(
        signal=np.asarray(transformed.signal),
        variant_id=st.ST16_VARIANT_ID,
        typed_units=st.ST16_TYPED_UNITS,
        sampling_rate_numerator=TARGET_FS_HZ,
        sampling_rate_denominator=1,
        sample_count=int(transformed.signal.shape[1]),
        source_signal_sha256=str(receipt["canonical_source_tensor_sha256"]),
        preprocessing_receipt_sha256=str(receipt["receipt_sha256"]),
        input_clock_receipt_sha256=str(receipt["input_clock_receipt_sha256"]),
    )
    plan = build_seizuretransformer_tile_plan(
        reader.sample_count,
        variant_id=st.ST16_VARIANT_ID,
        input_clock_receipt_sha256=reader.input_clock_receipt_sha256,
        source_signal_sha256=reader.source_signal_sha256,
        preprocessing_receipt_sha256=reader.preprocessing_receipt_sha256,
    )
    result = run_seizuretransformer_streaming_ola(
        reader, predictor, plan, batch_size=inference_batch_size
    )
    validate_seizuretransformer_streaming_result(result)
    if (
        result.receipt["ola_coverage_receipt"][
            "complete_record_posterior_coverage"
        ]
        is not True
    ):
        raise RuntimeError("ST16 OLA did not cover the complete recording")
    return result


def retain_observed_support_dense_probability(
    transformed: Any,
    dense: SeizureTransformerStreamingResult,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Discard model-context predictions before decoding, metrics or Findings.

    A short ST16 carrier must remain 60 seconds long for the frozen model, but
    only the prefix backed by the source EDF is an EEG observation.  The OLA
    receipt therefore continues to bind the complete fixed-shape model
    forward, while the returned sidecar is physically clipped to the immutable
    valid-support mask.  This is an inference/evidence firewall in addition to
    the already-masked training target, loss and metrics.
    """

    validate_seizuretransformer_streaming_result(dense)
    transform_receipt = transformed.receipt
    plan = dense.receipt["tile_plan"]
    source_binding = plan["source_binding"]
    expected_binding = {
        "source_signal_sha256": transform_receipt[
            "canonical_source_tensor_sha256"
        ],
        "preprocessing_receipt_sha256": transform_receipt["receipt_sha256"],
        "input_clock_receipt_sha256": transform_receipt[
            "input_clock_receipt_sha256"
        ],
    }
    if (
        dense.receipt.get("variant_id") != st.ST16_VARIANT_ID
        or transform_receipt.get("variant_id") != st.ST16_VARIANT_ID
        or source_binding != expected_binding
        or plan.get("sample_count") != transformed.signal.shape[1]
    ):
        raise PermissionError("ST16 dense posterior/transform lineage mismatch")
    if isinstance(transformed, _StagedTransformCarrier):
        observed = transformed.valid_support_sample_count
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or not 0 < observed <= transformed.signal.shape[1]
        ):
            raise ValueError("staged ST16 valid-support authority is malformed")
        mask = np.zeros(transformed.signal.shape[1], dtype=np.uint8)
        mask[:observed] = 1
        mask.setflags(write=False)
        mask_receipt = _content_address(
            {
                "schema_version": (
                    "st16_staged_transform_valid_support_mask_receipt_v1"
                ),
                "preprocess_stage_receipt_sha256": (
                    transformed.preprocess_stage_receipt_sha256
                ),
                "transform_receipt_sha256": transformed.receipt[
                    "receipt_sha256"
                ],
                "model_context_sample_count": int(transformed.signal.shape[1]),
                "observed_support_sample_count": observed,
                "context_sample_count": int(transformed.signal.shape[1] - observed),
                "mask_payload_receipt": st._payload_receipt(
                    mask,
                    semantic=(
                        "SeizureTransformer_transform_valid_loss_metric_support"
                    ),
                ),
                "context_may_receive_target_loss_or_metric_weight": False,
                "receipt_sha256": _PENDING,
            }
        )
    else:
        mask, mask_receipt = st.seizuretransformer_transform_valid_support_mask(
            transformed
        )
    posterior = np.asarray(dense.posterior_probability)
    if posterior.shape != mask.shape or posterior.dtype != np.dtype("float32"):
        raise ValueError("ST16 posterior and transform support mask disagree")
    observed_count = int(np.count_nonzero(mask))
    if (
        observed_count < 1
        or not np.all(mask[:observed_count] == 1)
        or not np.all(mask[observed_count:] == 0)
    ):
        raise ValueError("ST16 valid support must be one observed prefix")
    observed = np.ascontiguousarray(posterior[:observed_count], dtype="<f4")
    observed.setflags(write=False)
    receipt = _content_address(
        {
            "schema_version": "st16_observed_support_dense_probability_v1",
            "claim_status": "detector_navigation_only_nonpromotable",
            "transform_receipt_sha256": transformed.receipt["receipt_sha256"],
            "OLA_result_receipt_sha256": dense.receipt["receipt_sha256"],
            "valid_support_mask_receipt_sha256": mask_receipt["receipt_sha256"],
            "model_context_sample_count": int(posterior.shape[0]),
            "observed_support_sample_count": observed_count,
            "discarded_context_prediction_sample_count": int(
                posterior.shape[0] - observed_count
            ),
            "observed_probability_payload_receipt": st._payload_receipt(
                observed,
                semantic="ST16_observed_support_dense_seizure_probability",
            ),
            "context_probability_persisted": False,
            "context_probability_may_enter_threshold_or_event_decoding": False,
            "context_probability_may_enter_loss_or_metric": False,
            "context_probability_may_authorize_Finding_or_clinical_fact": False,
            "model_carrier_OLA_sample_range": [0, int(posterior.shape[0])],
            "observed_sidecar_sample_range": [0, observed_count],
            "observed_prefix_predictions_may_depend_on_model_context": (
                observed_count < posterior.shape[0]
            ),
            "short_record_Finding_use_requires_checkpoint_specific_context_"
            "sensitivity_gate": observed_count < posterior.shape[0],
            "receipt_sha256": _PENDING,
        }
    )
    return observed, receipt


def _reuse_prediction_receipt(
    path: Path, *, checkpoint_sha256: str, projection_receipt_sha256: str
) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    row = _validate_content_address(
        json.loads(path.read_text(encoding="utf-8")),
        artifact_name=f"ST16 source-dev prediction row {path.parent.name}",
    )
    if (
        row.get("checkpoint_sha256") != checkpoint_sha256
        or row.get("analysis_projection_receipt_sha256")
        != projection_receipt_sha256
        or row.get("model_split") != "source_dev"
        or row.get("source_eval_opened") is not False
    ):
        raise PermissionError("existing ST16 prediction receipt crosses run lineage")
    if row.get("status") == "dense_prediction_complete":
        sidecar = Path(row["dense_probability_path"])
        if not sidecar.is_file() or _file_sha256(sidecar) != row["dense_probability_sha256"]:
            raise ValueError("existing ST16 dense sidecar failed byte replay")
    return row


_PREDICTION_WORKER_REGISTRY: dict[str, Any] | None = None


def _validate_prediction_prefetch_contract(
    preprocess_workers: int, preprocess_prefetch: int
) -> None:
    for value, name, maximum in (
        (
            preprocess_workers,
            "preprocess_workers",
            PREDICTION_PREPROCESS_WORKERS_MAXIMUM,
        ),
        (
            preprocess_prefetch,
            "preprocess_prefetch",
            PREDICTION_PREFETCH_DEPTH_MAXIMUM,
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"{name} must be an explicit integer in [1, {maximum}]")
    if preprocess_workers == 1 and preprocess_prefetch != 1:
        raise ValueError("serial preprocessing requires preprocess_prefetch=1")
    if preprocess_workers > 1 and not (
        preprocess_workers <= preprocess_prefetch <= 2 * preprocess_workers
    ):
        raise ValueError(
            "parallel preprocessing requires workers <= prefetch <= 2*workers"
        )


def _prediction_stage_paths(stage_root: Path, identity: str) -> tuple[Path, Path]:
    if (
        not identity.startswith("TUSZANALYSIS-")
        or "/" in identity
        or "\\" in identity
        or ".." in identity
    ):
        raise PermissionError("prediction preprocessing identity is not path-safe")
    record_stage = stage_root / identity
    return record_stage / "transformed_signal.npy", record_stage / "receipt.json"


def _prediction_stage_receipt(
    *,
    identity_row: Mapping[str, Any],
    roster_index: int,
    projection_receipt_sha256: str,
    registry_sha256: str,
    transformed: Any,
    signal_path: Path,
) -> dict[str, Any]:
    signal = np.asarray(transformed.signal)
    transform_receipt = deepcopy(dict(transformed.receipt))
    valid_support_mask, valid_support_receipt = (
        st.seizuretransformer_transform_valid_support_mask(transformed)
    )
    observed_support_count = int(np.count_nonzero(valid_support_mask))
    return _content_address(
        {
            "schema_version": PREDICTION_PREPROCESS_STAGE_SCHEMA_VERSION,
            "claim_status": "temporary_target_free_source_dev_preprocess_stage",
            "provider_id": PROVIDER_ID,
            "analysis_identity_id": str(identity_row["analysis_identity_id"]),
            "recording_id": str(identity_row["local_edf_path"]),
            "patient_id": str(identity_row["local_patient_id"]),
            "model_split": "source_dev",
            "prediction_roster_index": roster_index,
            "analysis_projection_receipt_sha256": projection_receipt_sha256,
            "transform_registry_sha256": registry_sha256,
            "variant_id": st.ST16_VARIANT_ID,
            "st16_typed_units": list(st.ST16_TYPED_UNITS),
            "sampling_rate_hz": TARGET_FS_HZ,
            "shape": [int(value) for value in signal.shape],
            "dtype": "float32",
            "signal_file_name": signal_path.name,
            "signal_file_size_bytes": signal_path.stat().st_size,
            "signal_file_sha256": _file_sha256(signal_path),
            "transform_receipt": transform_receipt,
            "transform_receipt_sha256": transform_receipt["receipt_sha256"],
            "valid_support": {
                "schema_version": "st16_prediction_stage_valid_support_v1",
                "observed_support_sample_count": observed_support_count,
                "model_context_sample_count": int(signal.shape[1]),
                "context_sample_count": int(
                    signal.shape[1] - observed_support_count
                ),
                "valid_support_mask_payload_receipt": st._payload_receipt(
                    valid_support_mask,
                    semantic=(
                        "SeizureTransformer_transform_valid_loss_metric_support"
                    ),
                ),
                "transform_valid_support_mask_receipt": valid_support_receipt,
                "context_probability_may_be_persisted_or_decoded": False,
                "context_probability_may_authorize_Finding": False,
            },
            "source_eval_opened": False,
            "reference_annotation_or_target_opened": False,
            "GPU_model_inference_executed_by_preprocess_worker": False,
            "receipt_sha256": _PENDING,
        }
    )


def _validate_prediction_stage(
    *,
    stage_root: Path,
    identity_row: Mapping[str, Any],
    roster_index: int,
    projection_receipt_sha256: str,
    registry_sha256: str,
) -> tuple[dict[str, Any], np.memmap]:
    identity = str(identity_row["analysis_identity_id"])
    signal_path, receipt_path = _prediction_stage_paths(stage_root, identity)
    if (
        signal_path.is_symlink()
        or receipt_path.is_symlink()
        or not signal_path.is_file()
        or not receipt_path.is_file()
    ):
        raise ValueError("prediction preprocessing stage is incomplete")
    receipt = _validate_content_address(
        json.loads(receipt_path.read_text(encoding="utf-8")),
        artifact_name=f"ST16 prediction preprocessing stage {identity}",
    )
    required = {
        "schema_version": PREDICTION_PREPROCESS_STAGE_SCHEMA_VERSION,
        "claim_status": "temporary_target_free_source_dev_preprocess_stage",
        "provider_id": PROVIDER_ID,
        "analysis_identity_id": identity,
        "recording_id": str(identity_row["local_edf_path"]),
        "patient_id": str(identity_row["local_patient_id"]),
        "model_split": "source_dev",
        "prediction_roster_index": roster_index,
        "analysis_projection_receipt_sha256": projection_receipt_sha256,
        "transform_registry_sha256": registry_sha256,
        "variant_id": st.ST16_VARIANT_ID,
        "st16_typed_units": list(st.ST16_TYPED_UNITS),
        "sampling_rate_hz": TARGET_FS_HZ,
        "dtype": "float32",
        "signal_file_name": signal_path.name,
        "signal_file_size_bytes": signal_path.stat().st_size,
        "signal_file_sha256": _file_sha256(signal_path),
        "source_eval_opened": False,
        "reference_annotation_or_target_opened": False,
        "GPU_model_inference_executed_by_preprocess_worker": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise PermissionError("prediction preprocessing stage lineage drifted")
    transform_receipt = receipt.get("transform_receipt")
    if (
        not isinstance(transform_receipt, Mapping)
        or transform_receipt.get("receipt_sha256")
        != receipt.get("transform_receipt_sha256")
        or transform_receipt.get("scope_receipt", {}).get(
            "seizure_target_or_reference_label_used"
        )
        is not False
        or transform_receipt.get("scope_receipt", {}).get("EDF_annotation_used")
        is not False
    ):
        raise PermissionError("staged ST16 transform receipt is not target-free")
    signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(signal, np.memmap)
        or signal.dtype != np.dtype("float32")
        or signal.shape[0] != len(st.ST16_TYPED_UNITS)
        or list(signal.shape) != receipt.get("shape")
        or signal.shape[1] < TILE_SAMPLES
    ):
        del signal
        raise ValueError("staged ST16 transform tensor is malformed")
    short_context = (
        transform_receipt.get("schema_version")
        == "seizuretransformer_short_context_transform_receipt_v1"
    )
    expected_signal_receipt = st._payload_receipt(
        signal,
        semantic=(
            "SeizureTransformer_short_record_fixed_context_carrier"
            if short_context
            else "SeizureTransformer_provider_native_full_record"
        ),
    )
    if transform_receipt.get("output", {}).get("payload_receipt") != (
        expected_signal_receipt
    ):
        signal._mmap.close()
        del signal
        raise PermissionError("staged ST16 signal/transform payload binding drifted")
    support = receipt.get("valid_support")
    if not isinstance(support, Mapping):
        signal._mmap.close()
        del signal
        raise PermissionError("staged ST16 valid-support authority is absent")
    observed_support_count = support.get("observed_support_sample_count")
    if (
        support.get("schema_version")
        != "st16_prediction_stage_valid_support_v1"
        or isinstance(observed_support_count, bool)
        or not isinstance(observed_support_count, int)
        or not 0 < observed_support_count <= signal.shape[1]
        or support.get("model_context_sample_count") != signal.shape[1]
        or support.get("context_sample_count")
        != signal.shape[1] - observed_support_count
        or support.get("context_probability_may_be_persisted_or_decoded") is not False
        or support.get("context_probability_may_authorize_Finding") is not False
    ):
        signal._mmap.close()
        del signal
        raise PermissionError("staged ST16 valid-support authority drifted")
    expected_mask = np.zeros(signal.shape[1], dtype=np.uint8)
    expected_mask[:observed_support_count] = 1
    if support.get("valid_support_mask_payload_receipt") != st._payload_receipt(
        expected_mask,
        semantic="SeizureTransformer_transform_valid_loss_metric_support",
    ):
        signal._mmap.close()
        del signal
        raise PermissionError("staged ST16 valid-support mask payload drifted")
    transform_support_receipt = support.get(
        "transform_valid_support_mask_receipt"
    )
    if not isinstance(transform_support_receipt, Mapping):
        signal._mmap.close()
        del signal
        raise PermissionError("staged ST16 transform-support receipt is absent")
    _validate_content_address(
        transform_support_receipt,
        artifact_name="staged ST16 transform valid-support mask",
    )
    if (
        transform_support_receipt.get("transform_receipt_sha256")
        != transform_receipt.get("receipt_sha256")
        or transform_support_receipt.get("observed_support_sample_count")
        != observed_support_count
        or transform_support_receipt.get("mask_payload_receipt")
        != support.get("valid_support_mask_payload_receipt")
    ):
        signal._mmap.close()
        del signal
        raise PermissionError("staged ST16 transform-support binding drifted")
    return receipt, signal


def _prediction_preprocess_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """CPU-only transform worker; it never loads a model or opens CUDA."""

    global _PREDICTION_WORKER_REGISTRY
    began = time.perf_counter()
    identity_row = dict(payload["identity_row"])
    identity = str(identity_row["analysis_identity_id"])
    roster_index = int(payload["prediction_roster_index"])
    try:
        if torch.cuda.is_initialized():
            raise PermissionError("prediction preprocess worker inherited CUDA state")
        root = Path(str(payload["tusz_root"])).resolve(strict=True)
        stage_root = Path(str(payload["stage_root"])).resolve(strict=True)
        projection_sha = str(payload["analysis_projection_receipt_sha256"])
        if _PREDICTION_WORKER_REGISTRY is None:
            _PREDICTION_WORKER_REGISTRY = st.load_registry(
                Path(__file__).resolve().parents[2] / st.CONFIG_RELATIVE_PATH
            )
        registry_sha = str(_PREDICTION_WORKER_REGISTRY["registry_sha256"])
        if registry_sha != str(payload["transform_registry_sha256"]):
            raise PermissionError("prediction worker transform registry drifted")
        try:
            stage_receipt, staged_signal = _validate_prediction_stage(
                stage_root=stage_root,
                identity_row=identity_row,
                roster_index=roster_index,
                projection_receipt_sha256=projection_sha,
                registry_sha256=registry_sha,
            )
            staged_signal._mmap.close()
            del staged_signal
            reused = True
        except (OSError, TypeError, ValueError, PermissionError, json.JSONDecodeError):
            edf_path = _safe_edf(
                root,
                str(identity_row["local_edf_path"]),
                expected_split="source_dev",
            )
            transformed = _transform_st16_record(edf_path, _PREDICTION_WORKER_REGISTRY)
            signal_path, receipt_path = _prediction_stage_paths(stage_root, identity)
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            _save_numpy_atomic(signal_path, transformed.signal, replace=True)
            stage_receipt = _prediction_stage_receipt(
                identity_row=identity_row,
                roster_index=roster_index,
                projection_receipt_sha256=projection_sha,
                registry_sha256=registry_sha,
                transformed=transformed,
                signal_path=signal_path,
            )
            _write_json_atomic(receipt_path, stage_receipt, replace=True)
            # Re-open and byte/hash replay in the worker before publication to parent.
            verified, staged_signal = _validate_prediction_stage(
                stage_root=stage_root,
                identity_row=identity_row,
                roster_index=roster_index,
                projection_receipt_sha256=projection_sha,
                registry_sha256=registry_sha,
            )
            staged_signal._mmap.close()
            del staged_signal
            if verified["receipt_sha256"] != stage_receipt["receipt_sha256"]:
                raise RuntimeError("prediction preprocessing stage replay drifted")
            reused = False
        return {
            "status": "preprocess_stage_complete",
            "analysis_identity_id": identity,
            "prediction_roster_index": roster_index,
            "preprocess_stage_receipt_sha256": stage_receipt["receipt_sha256"],
            "preprocess_stage_reused": reused,
            "preprocess_worker_pid": os.getpid(),
            "GPU_model_inference_executed_by_preprocess_worker": False,
            "preprocess_worker_CUDA_initialized": torch.cuda.is_initialized(),
            "wall_seconds": time.perf_counter() - began,
        }
    except Exception as error:
        return {
            "status": "typed_preprocess_failure",
            "analysis_identity_id": identity,
            "prediction_roster_index": roster_index,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "preprocess_worker_pid": os.getpid(),
            "GPU_model_inference_executed_by_preprocess_worker": False,
            "preprocess_worker_CUDA_initialized": torch.cuda.is_initialized(),
            "wall_seconds": time.perf_counter() - began,
        }


def _ordered_bounded_prediction_preprocess(
    payloads: Sequence[Mapping[str, Any]],
    *,
    preprocess_workers: int,
    preprocess_prefetch: int,
):
    """Yield spawn-worker results in roster order with a hard future bound."""

    executor = ProcessPoolExecutor(
        max_workers=preprocess_workers,
        mp_context=mp.get_context("spawn"),
    )
    futures: dict[int, Future[dict[str, Any]]] = {}
    next_submit = 0
    next_yield = 0
    maximum_observed = 0
    try:
        while next_yield < len(payloads):
            while (
                next_submit < len(payloads)
                and len(futures) < preprocess_prefetch
            ):
                futures[next_submit] = executor.submit(
                    _prediction_preprocess_worker, dict(payloads[next_submit])
                )
                next_submit += 1
                maximum_observed = max(maximum_observed, len(futures))
            result = futures.pop(next_yield).result()
            result["maximum_inflight_preprocess_futures_observed"] = maximum_observed
            yield result
            next_yield += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _cleanup_prediction_stage(stage_root: Path, identity: str) -> None:
    signal_path, receipt_path = _prediction_stage_paths(stage_root, identity)
    for candidate in (signal_path, receipt_path):
        if candidate.is_symlink():
            raise PermissionError("prediction preprocessing cleanup found a symlink")
        if candidate.is_file():
            candidate.unlink()
    try:
        signal_path.parent.rmdir()
    except OSError:
        pass


def predict_source_dev_dense(
    *,
    checkpoint_path: str | Path,
    analysis_projection_path: str | Path,
    tusz_root: str | Path,
    output_dir: str | Path,
    device_name: str,
    inference_batch_size: int,
    maximum_records: int | None = None,
    preprocess_workers: int = 1,
    preprocess_prefetch: int = 1,
) -> dict[str, Any]:
    """Materialize resumable complete source-dev pre-threshold dense predictions."""

    if maximum_records is not None and maximum_records < 1:
        raise ValueError("maximum_records must be positive when supplied")
    _validate_prediction_prefetch_contract(
        preprocess_workers, preprocess_prefetch
    )
    device = torch.device(device_name)
    if device.type == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("ST16 CUDA inference requires BF16-capable CUDA")
    projection_source = Path(analysis_projection_path).resolve(strict=True)
    projection = json.loads(projection_source.read_text(encoding="utf-8"))
    rows = select_target_free_prediction_rows(projection, split="source_dev")
    full_expected_count = len(rows)
    if maximum_records is not None:
        rows = rows[:maximum_records]
    model, checkpoint, checkpoint_sha = _load_exploratory_checkpoint(
        checkpoint_path, device=device
    )
    predictor = _model_predictor(model, device)
    root = Path(tusz_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    registry = st.load_registry(Path(__file__).resolve().parents[2] / st.CONFIG_RELATIVE_PATH)
    registry_sha = str(registry["registry_sha256"])
    execution_mode = (
        "serial_parent"
        if preprocess_workers == 1
        else "bounded_spawn_CPU_prefetch_parent_only_GPU_inference"
    )
    stage_root = output / ".source_dev_preprocess_stage_v1"
    if preprocess_workers > 1:
        if stage_root.is_symlink():
            raise PermissionError("prediction preprocessing stage root may not be a symlink")
        stage_root.mkdir(parents=True, exist_ok=True)

    reused_by_index: dict[int, dict[str, Any]] = {}
    pending_payloads: list[dict[str, Any]] = []
    for index, identity_row in enumerate(rows):
        identity = str(identity_row["analysis_identity_id"])
        receipt_path = output / "records" / identity / "receipt.json"
        reused = _reuse_prediction_receipt(
            receipt_path,
            checkpoint_sha256=checkpoint_sha,
            projection_receipt_sha256=str(projection["receipt_sha256"]),
        )
        if reused is not None:
            reused_by_index[index] = reused
        elif preprocess_workers > 1:
            pending_payloads.append(
                {
                    "tusz_root": str(root),
                    "stage_root": str(stage_root),
                    "identity_row": dict(identity_row),
                    "prediction_roster_index": index,
                    "analysis_projection_receipt_sha256": str(
                        projection["receipt_sha256"]
                    ),
                    "transform_registry_sha256": registry_sha,
                }
            )
    prepared_iterator = (
        iter(())
        if preprocess_workers == 1
        else _ordered_bounded_prediction_preprocess(
            pending_payloads,
            preprocess_workers=preprocess_workers,
            preprocess_prefetch=preprocess_prefetch,
        )
    )
    result_rows: list[dict[str, Any]] = []
    preprocess_worker_pids: set[int] = set()
    maximum_inflight_observed = 0
    began = time.perf_counter()
    for index, identity_row in enumerate(rows):
        identity = str(identity_row["analysis_identity_id"])
        record_dir = output / "records" / identity
        receipt_path = record_dir / "receipt.json"
        reused = reused_by_index.get(index)
        if reused is not None:
            result_rows.append(reused)
            continue
        record_dir.mkdir(parents=True, exist_ok=True)
        record_began = time.perf_counter()
        base = {
            "schema_version": "st16_common17_source_dev_dense_prediction_row_v1",
            "provider_id": PROVIDER_ID,
            "claim_status": "exploratory_nonpromotable",
            "analysis_identity_id": identity,
            "recording_id": identity_row["local_edf_path"],
            "patient_id": identity_row["local_patient_id"],
            "model_split": "source_dev",
            "checkpoint_sha256": checkpoint_sha,
            "analysis_projection_receipt_sha256": projection["receipt_sha256"],
            "prediction_roster_index": index,
            "source_eval_opened": False,
            "reference_annotation_or_target_opened": False,
            "threshold_morphology_hysteresis_or_NMS_applied": False,
            "preprocessing_execution_mode": execution_mode,
            "GPU_model_inference_process_pid": os.getpid(),
        }
        staged_signal: np.memmap | None = None
        preprocess_result: dict[str, Any] | None = None
        try:
            if preprocess_workers == 1:
                edf_path = _safe_edf(
                    root,
                    str(identity_row["local_edf_path"]),
                    expected_split="source_dev",
                )
                transformed = _transform_st16_record(edf_path, registry)
                preprocessing_fields = {
                    "preprocess_worker_pid": None,
                    "preprocess_stage_receipt_sha256": None,
                    "preprocess_stage_reused": False,
                    "GPU_model_inference_executed_by_preprocess_worker": False,
                    "preprocess_worker_CUDA_initialized": False,
                }
            else:
                preprocess_result = next(prepared_iterator)
                if (
                    preprocess_result.get("analysis_identity_id") != identity
                    or preprocess_result.get("prediction_roster_index") != index
                ):
                    raise RuntimeError("parallel prediction preprocessing order drifted")
                preprocess_worker_pids.add(
                    int(preprocess_result["preprocess_worker_pid"])
                )
                if preprocess_result.get("preprocess_worker_CUDA_initialized") is not False:
                    raise PermissionError("prediction preprocess worker initialized CUDA")
                maximum_inflight_observed = max(
                    maximum_inflight_observed,
                    int(
                        preprocess_result.get(
                            "maximum_inflight_preprocess_futures_observed", 0
                        )
                    ),
                )
                if preprocess_result.get("status") != "preprocess_stage_complete":
                    raise RuntimeError(
                        "CPU_PREPROCESS_FAILURE:"
                        f"{preprocess_result.get('failure_type')}:"
                        f"{preprocess_result.get('failure_message')}"
                    )
                stage_receipt, staged_signal = _validate_prediction_stage(
                    stage_root=stage_root,
                    identity_row=identity_row,
                    roster_index=index,
                    projection_receipt_sha256=str(projection["receipt_sha256"]),
                    registry_sha256=registry_sha,
                )
                if (
                    stage_receipt["receipt_sha256"]
                    != preprocess_result["preprocess_stage_receipt_sha256"]
                ):
                    raise RuntimeError("parent preprocessing-stage replay drifted")
                transformed = _StagedTransformCarrier(
                    signal=staged_signal,
                    receipt=deepcopy(dict(stage_receipt["transform_receipt"])),
                )
                preprocessing_fields = {
                    "preprocess_worker_pid": int(
                        preprocess_result["preprocess_worker_pid"]
                    ),
                    "preprocess_stage_receipt_sha256": stage_receipt[
                        "receipt_sha256"
                    ],
                    "preprocess_stage_reused": bool(
                        preprocess_result["preprocess_stage_reused"]
                    ),
                    "GPU_model_inference_executed_by_preprocess_worker": False,
                    "preprocess_worker_CUDA_initialized": False,
                }
            dense = predict_transformed_record_dense(
                transformed,
                predictor=predictor,
                inference_batch_size=inference_batch_size,
            )
            observed_dense, observed_support_receipt = (
                retain_observed_support_dense_probability(transformed, dense)
            )
            sidecar = record_dir / "dense_probability.npy"
            # A sidecar without its row receipt is an interrupted unpublished
            # attempt and is safely replaceable on resume.
            _save_numpy_atomic(sidecar, observed_dense, replace=True)
            row = {
                **base,
                **preprocessing_fields,
                "status": "dense_prediction_complete",
                "sample_count": int(observed_dense.shape[0]),
                "model_context_sample_count": int(
                    dense.posterior_probability.shape[0]
                ),
                "discarded_context_prediction_sample_count": (
                    observed_support_receipt[
                        "discarded_context_prediction_sample_count"
                    ]
                ),
                "observed_support_probability_receipt": (
                    observed_support_receipt
                ),
                "context_probability_persisted": False,
                "context_probability_may_authorize_Finding": False,
                "sampling_rate_hz": TARGET_FS_HZ,
                "dense_probability_path": str(sidecar),
                "dense_probability_sha256": _file_sha256(sidecar),
                "dense_probability_dtype": "float32",
                "pre_threshold_dense_complete": True,
                "pre_NMS_candidate_information_complete": True,
                "OLA_result_receipt_sha256": dense.receipt["receipt_sha256"],
                "OLA_plan_receipt_sha256": dense.receipt["tile_plan"][
                    "receipt_sha256"
                ],
                "OLA_coverage_receipt": dense.receipt["ola_coverage_receipt"],
                "transform_receipt_sha256": transformed.receipt["receipt_sha256"],
                "wall_seconds": time.perf_counter() - record_began,
            }
        except Exception as error:  # retained denominator; no silent record drop
            failure_type = type(error).__name__
            failure_message = str(error)
            failure_stage = "GPU_inference_or_publication"
            if failure_message.startswith("CPU_PREPROCESS_FAILURE:"):
                failure_stage = "CPU_preprocessing"
                _, raw_type, raw_message = failure_message.split(":", 2)
                failure_type = raw_type
                failure_message = raw_message
            row = {
                **base,
                "status": "typed_technical_failure",
                "failure_stage": failure_stage,
                "failure_type": failure_type,
                "failure_message": failure_message,
                "preprocess_worker_pid": (
                    preprocess_result.get("preprocess_worker_pid")
                    if preprocess_result is not None
                    else None
                ),
                "preprocess_stage_receipt_sha256": (
                    preprocess_result.get("preprocess_stage_receipt_sha256")
                    if preprocess_result is not None
                    else None
                ),
                "GPU_model_inference_executed_by_preprocess_worker": False,
                "preprocess_worker_CUDA_initialized": (
                    preprocess_result.get("preprocess_worker_CUDA_initialized")
                    if preprocess_result is not None
                    else False
                ),
                "dense_probability_path": None,
                "dense_probability_sha256": None,
                "pre_threshold_dense_complete": False,
                "pre_NMS_candidate_information_complete": False,
                "wall_seconds": time.perf_counter() - record_began,
            }
        finally:
            if staged_signal is not None:
                staged_signal._mmap.close()
                del staged_signal
        row = _content_address(row)
        _write_json_atomic(receipt_path, row, replace=False)
        if preprocess_workers > 1:
            _cleanup_prediction_stage(stage_root, identity)
        result_rows.append(row)

    complete_rows = [row for row in result_rows if row["status"] == "dense_prediction_complete"]
    failures = [row for row in result_rows if row["status"] == "typed_technical_failure"]
    inventory_complete = maximum_records is None and len(result_rows) == full_expected_count
    manifest = _content_address(
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "claim_status": "exploratory_nonpromotable",
            "checkpoint_path": str(Path(checkpoint_path).resolve(strict=True)),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_next_epoch": checkpoint["next_epoch"],
            "analysis_projection_path": str(projection_source),
            "analysis_projection_file_sha256": _file_sha256(projection_source),
            "analysis_projection_receipt_sha256": projection["receipt_sha256"],
            "split": "source_dev",
            "source_eval_opened": False,
            "reference_annotation_or_target_opened": False,
            "full_expected_record_count": full_expected_count,
            "materialized_record_count": len(result_rows),
            "dense_prediction_complete_count": len(complete_rows),
            "typed_technical_failure_count": len(failures),
            "complete_prediction_inventory": inventory_complete,
            "maximum_records_smoke_limit": maximum_records,
            "inference_batch_size": inference_batch_size,
            "preprocessing_execution_mode": execution_mode,
            "preprocess_workers": preprocess_workers,
            "preprocess_prefetch": preprocess_prefetch,
            "maximum_inflight_preprocess_futures_observed": maximum_inflight_observed,
            "preprocess_worker_pids": sorted(preprocess_worker_pids),
            "GPU_model_inference_parent_pid": os.getpid(),
            "parent_only_GPU_model_inference": True,
            "prediction_roster_order_preserved": [
                row["analysis_identity_id"] for row in result_rows
            ]
            == [str(row["analysis_identity_id"]) for row in rows],
            "intention_to_evaluate_denominator_preserved": len(result_rows)
            == len(rows),
            "pre_threshold_dense_sidecar_for_every_success": True,
            "threshold_morphology_hysteresis_or_NMS_applied": False,
            "prediction_rows": result_rows,
            "wall_seconds": time.perf_counter() - began,
            "architecture_promotable": False,
            "source_dev_metrics_authorized_only_after_inventory_freeze": inventory_complete,
            "source_eval_metrics_authorized": False,
            "receipt_sha256": _PENDING,
        }
    )
    _write_json_atomic(output / "prediction_manifest.json", manifest, replace=True)
    return manifest


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "ENGINEERING_BATCH_SIZE_MAXIMUM",
    "PARTIAL_BATCH_POLICIES",
    "PREDICTION_SCHEMA_VERSION",
    "PROVIDER_ID",
    "SCHEMA_VERSION",
    "build_exploratory_epoch_plan",
    "build_exploratory_epoch_plan_from_records",
    "event_sample_spans",
    "materialize_epoch_tile_cache",
    "predict_source_dev_dense",
    "predict_transformed_record_dense",
    "retain_observed_support_dense_probability",
    "select_target_free_prediction_rows",
    "train_exploratory_st16",
]
