"""Streaming common-17 EventNet experiment for complete TUSZ recordings.

This is an additive experimental lane.  It deliberately uses one fixed set of
17 *observed referential* scalp electrodes for every record.  FZ and PZ are
never read into the model tensor and no absent electrode is zero-filled or
interpolated.  Only global ``TERM,seiz`` intervals are accepted as detector
targets; channel annotations and clinical text have no input path here.

The module provides four small, composable pieces used by the command-line
runner:

* a content-addressed source-train/source-dev manifest;
* a random-access, polyphase-resampled 120 s training-tile reader;
* a patient-balanced streaming epoch plan and resumable trainer; and
* full-record source-dev inference plus event-level operating-point metrics.

The official TUSZ train/dev split is patient-disjoint.  Every canonical
physical record, seizure-free record, background second and ``TERM,seiz``
event remains in the manifest/evaluation denominator.  Training samples from
the complete tile pools; sampling does not redefine the evaluation cohort.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pyedflib
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample_poly
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .continuous_detection_benchmark import _aggregate_metrics
from .continuous_detection_source_dev_join import (
    parse_tusz_term_seiz_reference_bytes,
)
from .eventnet_cleanroom_registry_v1 import (
    CONTEXT_SAMPLES_PER_SIDE,
    EN17_CHANNEL_ORDER,
    EN17_VARIANT_ID,
    MAXIMUM_DURATION_SECONDS,
    MODEL_INPUT_SAMPLES,
    TARGET_FS_HZ,
    TARGET_TILE_SAMPLES,
    EventNetCleanroomUNet,
    _polyphase_taps,
    build_eventnet_targets_pure_primitive,
    build_randomly_initialized_model,
    enumerate_target_tiles,
    enumerate_training_target_tiles,
    eventnet_multitask_loss_from_logits_pure_primitive,
    materialize_model_tile,
)
from .montage_reference_observability import classify_signal_labels


SCHEMA_VERSION: Final[str] = "eventnet_common17_streaming_experiment_v1"
MANIFEST_SCHEMA_VERSION: Final[str] = "tusz_common17_detector_manifest_v1"
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "eventnet_common17_checkpoint_round_sampler_v2"
)
LEGACY_POST_NMS_PREDICTION_SCHEMA_VERSION: Final[str] = (
    "eventnet_common17_dev_prediction_global_posterior_runtime_v3"
)
PREDICTION_SCHEMA_VERSION: Final[str] = (
    "eventnet_common17_dev_prediction_replayable_pre_nms_runtime_v4"
)
PRE_NMS_CANDIDATE_CACHE_SCHEMA_VERSION: Final[str] = (
    "eventnet_common17_global_pre_nms_candidate_cache_v1"
)
COMMON17_CHANNEL_ORDER: Final[tuple[str, ...]] = tuple(EN17_CHANNEL_ORDER)
DEFAULT_SMOOTHING_SIGMA_SAMPLES: Final[int] = 100
DEFAULT_MINIMUM_PEAK_DISTANCE_SECONDS: Final[int] = 60
EPOCH_PLAN_ID: Final[str] = (
    "common17_patient8_deterministic_pool_shuffle_round_batches_v2"
)
DRAWS_PER_PATIENT_PER_EPOCH: Final[int] = 8
DEFAULT_WEIGHT_DECAY: Final[float] = 2e-5
GRADIENT_CLIP_GLOBAL_L2_NORM: Final[float] = 1.0
CUDA_AUTOCAST_DTYPE: Final[str] = "bfloat16"
TRAINING_LOSS_DTYPE: Final[str] = "float32"
GRADIENT_SCALER_ENABLED: Final[bool] = False
FORMAL_EPOCHS: Final[int] = 3
FORMAL_BATCH_SIZE: Final[int] = 8
FORMAL_LEARNING_RATE: Final[float] = 1e-4
FORMAL_SEED: Final[int] = 20260824
FORMAL_CHECKPOINT_EVERY_STEPS: Final[int] = 50
_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


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


def _safe_relative_edf(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] not in {"train", "dev"}
        or relative.suffix.lower() != ".edf"
    ):
        raise PermissionError("common17 accepts only train/dev relative EDF paths")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError("common17 source EDF must not be a symlink")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _fraction_from_row(value: object) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError("recording duration fraction is invalid")
    return Fraction(value[0], value[1])


def _seizure_union_seconds(events: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(row["stop_seconds"]) - float(row["start_seconds"]) for row in events)


def materialize_common17_manifest(
    *,
    fold_plan_path: str | Path,
    canonical_audit_path: str | Path,
    tusz_root: str | Path,
    output_path: str | Path,
    maximum_records_per_split: int | None = None,
) -> dict[str, Any]:
    """Create the complete train/dev detector manifest.

    ``maximum_records_per_split`` exists only for explicitly marked smoke
    manifests.  A manifest created with it cannot claim a complete denominator.
    """

    fold_path = Path(fold_plan_path).resolve(strict=True)
    audit_path = Path(canonical_audit_path).resolve(strict=True)
    root = Path(tusz_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if maximum_records_per_split is not None and maximum_records_per_split < 1:
        raise ValueError("maximum_records_per_split must be positive")
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    successful = {
        str(row["analysis_identity_id"]): row
        for row in audit["outcomes"]
        if row.get("terminal_status") == "success"
    }
    selected_count: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    expected = {"source_train": "train", "source_dev": "dev"}
    for raw in fold["source_record_duration_rows"]:
        split = str(raw["model_split"])
        if split not in expected:
            continue
        if maximum_records_per_split is not None and selected_count[split] >= maximum_records_per_split:
            continue
        if raw["official_split"] != expected[split]:
            raise ValueError("fold-plan model/official split binding drifted")
        identity = str(raw["analysis_identity_id"])
        audited = successful.get(identity)
        if audited is None:
            raise ValueError("fold-plan identity lacks a successful canonical audit")
        physical = audited["physical_signal"]
        observed = tuple(str(value) for value in physical["observed_channel_ids"])
        if not set(COMMON17_CHANNEL_ORDER).issubset(observed):
            raise ValueError(f"canonical identity lacks common17 support: {identity}")
        relative_path = str(raw["local_edf_path"])
        edf_path = _safe_relative_edf(root, relative_path)
        duration = _fraction_from_row(raw["recording_duration_seconds_fraction"])
        reference_path = edf_path.with_suffix(".csv_bi")
        if reference_path.is_symlink() or not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        parsed = parse_tusz_term_seiz_reference_bytes(
            reference_path.read_bytes(), duration_seconds=float(duration)
        )
        events = parsed.events()
        record_sample_count = math.floor(duration * TARGET_FS_HZ)
        if record_sample_count < 1:
            raise ValueError("common17 record has no 256-Hz support")
        records.append(
            {
                "analysis_identity_id": identity,
                "model_split": split,
                "official_split": str(raw["official_split"]),
                "patient_id": str(raw["local_patient_id"]),
                "edf_relative_path": relative_path,
                "recording_duration_seconds_fraction": [
                    duration.numerator,
                    duration.denominator,
                ],
                "target_sample_count_256hz": record_sample_count,
                "seizure_events": events,
                "seizure_event_count": len(events),
                "reference_csv_bi_sha256": parsed.reference_file_sha256,
                "canonical_source_tensor_sha256": physical[
                    "canonical_source_tensor_sha256"
                ],
                "audited_observed_channel_ids": list(observed),
                "common17_direct_axis_order": list(COMMON17_CHANNEL_ORDER),
            }
        )
        selected_count[split] += 1
    records.sort(
        key=lambda row: (
            row["model_split"],
            row["patient_id"],
            row["edf_relative_path"],
        )
    )
    split_summaries: dict[str, dict[str, Any]] = {}
    patient_sets: dict[str, set[str]] = {}
    for split in expected:
        rows = [row for row in records if row["model_split"] == split]
        patients = {str(row["patient_id"]) for row in rows}
        patient_sets[split] = patients
        duration_seconds = sum(
            float(Fraction(*row["recording_duration_seconds_fraction"])) for row in rows
        )
        seizure_seconds = sum(_seizure_union_seconds(row["seizure_events"]) for row in rows)
        split_summaries[split] = {
            "recording_count": len(rows),
            "patient_count": len(patients),
            "seizure_free_recording_count": sum(not row["seizure_events"] for row in rows),
            "seizure_event_count": sum(len(row["seizure_events"]) for row in rows),
            "recording_hours": duration_seconds / 3600.0,
            "seizure_hours": seizure_seconds / 3600.0,
            "background_hours": (duration_seconds - seizure_seconds) / 3600.0,
        }
    overlap = sorted(patient_sets["source_train"] & patient_sets["source_dev"])
    if overlap:
        raise PermissionError("source-train/source-dev patients overlap")
    complete = maximum_records_per_split is None
    if complete and (
        split_summaries["source_train"]["recording_count"] != 4664
        or split_summaries["source_dev"]["recording_count"] != 1821
    ):
        raise ValueError("complete canonical train/dev denominator drifted")
    manifest = _content_address(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "method_id": "canonical_physical_common17_exact_TERM_seiz_v1",
            "source_bindings": {
                "fold_plan_path": str(fold_path),
                "fold_plan_file_sha256": _file_sha256(fold_path),
                "fold_plan_receipt_sha256": fold["receipt_sha256"],
                "canonical_audit_path": str(audit_path),
                "canonical_audit_file_sha256": _file_sha256(audit_path),
                "canonical_audit_receipt_sha256": audit["receipt_sha256"],
                "tusz_root": str(root),
            },
            "channel_contract": {
                "provider_variant_id": EN17_VARIANT_ID,
                "common17_channel_order": list(COMMON17_CHANNEL_ORDER),
                "FZ_or_PZ_read_into_model_tensor": False,
                "zero_fill_or_interpolation": False,
                "direct_observed_referential_axes_only": True,
            },
            "target_contract": {
                "global_TERM_seiz_only": True,
                "channel_specific_annotations_used": False,
                "EDF_plus_annotations_used": False,
                "clinical_text_or_spreadsheet_used": False,
            },
            "complete_denominator": complete,
            "maximum_records_per_split": maximum_records_per_split,
            "patient_disjoint_train_dev": True,
            "patient_overlap": overlap,
            "split_summaries": split_summaries,
            "records": records,
            "receipt_sha256": _PENDING,
        }
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(manifest) + b"\n")
    os.replace(temporary, target)
    return manifest


def load_common17_manifest(path: str | Path, *, require_complete: bool) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("common17 manifest schema drifted")
    pending = deepcopy(value)
    supplied = pending.get("receipt_sha256")
    pending["receipt_sha256"] = _PENDING
    if supplied != _canonical_sha256(pending):
        raise ValueError("common17 manifest is not content-addressed")
    if require_complete and value.get("complete_denominator") is not True:
        raise PermissionError("formal common17 execution requires a complete manifest")
    channel = value.get("channel_contract", {})
    if (
        channel.get("common17_channel_order") != list(COMMON17_CHANNEL_ORDER)
        or channel.get("FZ_or_PZ_read_into_model_tensor") is not False
        or channel.get("zero_fill_or_interpolation") is not False
    ):
        raise ValueError("common17 channel contract drifted")
    return value


@dataclass(frozen=True)
class _EDFLayout:
    indices: tuple[int, ...]
    unit_multipliers_to_uv: tuple[float, ...]
    sampling_rate: Fraction
    source_sample_count: int


def _unit_multiplier_to_uv(value: str) -> float:
    normalized = value.strip().lower().replace("µ", "u").replace("μ", "u")
    if normalized == "uv":
        return 1.0
    if normalized == "mv":
        return 1_000.0
    if normalized == "v":
        return 1_000_000.0
    raise ValueError(f"unsupported EEG physical unit: {value!r}")


def _edf_layout(reader: pyedflib.EdfReader) -> _EDFLayout:
    labels = reader.getSignalLabels()
    classified = classify_signal_labels(labels)
    by_electrode: dict[str, int] = {}
    for row in classified["signal_label_observations"]:
        if row["signal_role"] in {
            "direct_standard_electrode",
            "direct_standard_electrode_unknown_reference",
        }:
            electrode = str(row["positive_electrode"])
            if electrode in by_electrode:
                raise ValueError(f"duplicate direct electrode {electrode}")
            by_electrode[electrode] = int(row["signal_index"])
    missing = sorted(set(COMMON17_CHANNEL_ORDER).difference(by_electrode))
    if missing:
        raise ValueError(f"EDF lacks observed common17 channels: {missing}")
    indices = tuple(by_electrode[channel] for channel in COMMON17_CHANNEL_ORDER)
    frequencies = [Fraction(str(float(reader.getSampleFrequency(index)))).limit_denominator(4096) for index in indices]
    if len(set(frequencies)) != 1:
        raise ValueError("common17 source electrodes do not share a sampling clock")
    sample_counts = [int(reader.getNSamples()[index]) for index in indices]
    if len(set(sample_counts)) != 1 or sample_counts[0] < 1:
        raise ValueError("common17 source electrodes do not share sample count")
    multipliers = tuple(_unit_multiplier_to_uv(reader.getPhysicalDimension(index)) for index in indices)
    return _EDFLayout(
        indices=indices,
        unit_multipliers_to_uv=multipliers,
        sampling_rate=frequencies[0],
        source_sample_count=sample_counts[0],
    )


def _resample_ratio(rate: Fraction) -> tuple[int, int]:
    ratio = Fraction(TARGET_FS_HZ, 1) / rate
    if max(ratio.numerator, ratio.denominator) > 4096:
        raise ValueError("common17 resampling ratio exceeds supported bound")
    return ratio.numerator, ratio.denominator


def _read_source_matrix(
    reader: pyedflib.EdfReader,
    layout: _EDFLayout,
    *,
    start: int,
    count: int,
) -> np.ndarray:
    if start < 0 or count < 1 or start + count > layout.source_sample_count:
        raise ValueError("EDF source slice lies outside common clock")
    rows = []
    for index, multiplier in zip(layout.indices, layout.unit_multipliers_to_uv):
        values = reader.readSignal(index, start=start, n=count, digital=False)
        rows.append(np.asarray(values, dtype=np.float64) * multiplier)
    result = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float64)
    if result.shape != (len(COMMON17_CHANNEL_ORDER), count) or not np.isfinite(result).all():
        raise ValueError("common17 EDF slice is malformed")
    return result


def read_common17_training_tile(
    path: str | Path,
    *,
    target_start_sample: int,
) -> np.ndarray:
    """Read one fully observed EventNet tile without materializing a recording.

    The source slice starts on a polyphase-aligned source index and includes a
    one-second margin on both sides.  Cropping therefore lands on the same
    global 256-Hz grid as whole-record resampling while keeping memory/I/O
    bounded.  Training geometry already excludes record-edge padding.
    """

    with pyedflib.EdfReader(str(Path(path).resolve(strict=True))) as reader:
        layout = _edf_layout(reader)
        up, down = _resample_ratio(layout.sampling_rate)
        target_count = (layout.source_sample_count * up) // down
        wanted_start = target_start_sample - CONTEXT_SAMPLES_PER_SIDE
        wanted_stop = target_start_sample + TARGET_TILE_SAMPLES + CONTEXT_SAMPLES_PER_SIDE
        if wanted_start < 0 or wanted_stop > target_count:
            raise ValueError("streaming training tile requires fully observed context")
        margin_source = max(1, math.ceil(float(layout.sampling_rate)))
        desired_source_start = Fraction(wanted_start * down, up)
        desired_source_stop = Fraction(wanted_stop * down, up)
        source_start = max(0, math.floor(desired_source_start - margin_source))
        source_start -= source_start % down
        source_stop = min(
            layout.source_sample_count,
            math.ceil(desired_source_stop + margin_source),
        )
        if source_stop < layout.source_sample_count:
            source_stop = min(
                layout.source_sample_count,
                int(math.ceil(source_stop / down) * down),
            )
        source = _read_source_matrix(
            reader, layout, start=source_start, count=source_stop - source_start
        )
    if up == down == 1:
        resampled = source
    else:
        resampled = resample_poly(
            source,
            up,
            down,
            axis=1,
            window=_polyphase_taps(up, down),
            padtype="line",
        )
    base_target = source_start * up // down
    crop_start = wanted_start - base_target
    crop_stop = crop_start + MODEL_INPUT_SAMPLES
    if crop_start < 0 or crop_stop > resampled.shape[1]:
        raise RuntimeError("polyphase streaming crop lacks requested target support")
    output = np.ascontiguousarray(resampled[:, crop_start:crop_stop], dtype=np.float32)
    if output.shape != (len(COMMON17_CHANNEL_ORDER), MODEL_INPUT_SAMPLES):
        raise RuntimeError("common17 streaming tile geometry drifted")
    return output


def read_common17_full_record(path: str | Path) -> np.ndarray:
    """Read and resample an entire EDF using only the direct common17 axes."""

    with pyedflib.EdfReader(str(Path(path).resolve(strict=True))) as reader:
        layout = _edf_layout(reader)
        source = _read_source_matrix(
            reader, layout, start=0, count=layout.source_sample_count
        )
    up, down = _resample_ratio(layout.sampling_rate)
    target_count = (layout.source_sample_count * up) // down
    if up == down == 1:
        transformed = source.copy()
    else:
        transformed = resample_poly(
            source,
            up,
            down,
            axis=1,
            window=_polyphase_taps(up, down),
            padtype="line",
        )
    output = np.ascontiguousarray(transformed[:, :target_count], dtype=np.float32)
    if output.shape != (len(COMMON17_CHANNEL_ORDER), target_count) or not np.isfinite(output).all():
        raise RuntimeError("common17 full-record transform geometry drifted")
    return output


def _read_common17_inference_tile_from_reader(
    reader: pyedflib.EdfReader,
    layout: _EDFLayout,
    *,
    target_start_sample: int,
) -> tuple[np.ndarray, int, int]:
    """Materialize one inference tile while retaining only a bounded EDF slice."""

    up, down = _resample_ratio(layout.sampling_rate)
    target_count = (layout.source_sample_count * up) // down
    if (
        isinstance(target_start_sample, bool)
        or not isinstance(target_start_sample, int)
        or target_start_sample < 0
        or target_start_sample >= target_count
    ):
        raise ValueError("streaming inference tile start lies outside the record")
    actual = min(TARGET_TILE_SAMPLES, target_count - target_start_sample)
    wanted_start = target_start_sample - CONTEXT_SAMPLES_PER_SIDE
    wanted_stop = target_start_sample + TARGET_TILE_SAMPLES + CONTEXT_SAMPLES_PER_SIDE
    observed_start = max(0, wanted_start)
    observed_stop = min(target_count, wanted_stop)

    if up == down == 1:
        observed = _read_source_matrix(
            reader,
            layout,
            start=observed_start,
            count=observed_stop - observed_start,
        )
    else:
        # The explicit halo is much wider than the frozen polyphase filter's
        # source-clock support.  Aligning the source start to ``down`` keeps
        # the chunk on the same absolute output phase as whole-record
        # resampling; the halo is discarded after resampling.
        margin_source = max(1, math.ceil(float(layout.sampling_rate)))
        desired_source_start = Fraction(observed_start * down, up)
        desired_source_stop = Fraction(observed_stop * down, up)
        source_start = max(0, math.floor(desired_source_start - margin_source))
        source_start -= source_start % down
        source_stop = min(
            layout.source_sample_count,
            math.ceil(desired_source_stop + margin_source),
        )
        if source_stop < layout.source_sample_count:
            source_stop = min(
                layout.source_sample_count,
                int(math.ceil(source_stop / down) * down),
            )
        source = _read_source_matrix(
            reader,
            layout,
            start=source_start,
            count=source_stop - source_start,
        )
        resampled = resample_poly(
            source,
            up,
            down,
            axis=1,
            window=_polyphase_taps(up, down),
            padtype="line",
        )
        base_target = source_start * up // down
        crop_start = observed_start - base_target
        crop_stop = crop_start + observed_stop - observed_start
        if crop_start < 0 or crop_stop > resampled.shape[1]:
            raise RuntimeError("streaming inference resample lacks requested support")
        observed = resampled[:, crop_start:crop_stop]

    model_input = np.zeros(
        (len(COMMON17_CHANNEL_ORDER), MODEL_INPUT_SAMPLES), dtype=np.float32
    )
    destination_start = observed_start - wanted_start
    destination_stop = destination_start + observed_stop - observed_start
    model_input[:, destination_start:destination_stop] = observed
    if (
        model_input.shape != (len(COMMON17_CHANNEL_ORDER), MODEL_INPUT_SAMPLES)
        or not np.isfinite(model_input).all()
    ):
        raise RuntimeError("common17 streaming inference tile geometry drifted")
    return model_input, actual, target_count


def read_common17_inference_tile(
    path: str | Path, *, target_start_sample: int
) -> tuple[np.ndarray, int, int]:
    """Read one fixed-shape inference tile without retaining the full EEG."""

    with pyedflib.EdfReader(str(Path(path).resolve(strict=True))) as reader:
        layout = _edf_layout(reader)
        return _read_common17_inference_tile_from_reader(
            reader,
            layout,
            target_start_sample=target_start_sample,
        )


def streaming_full_parity(
    path: str | Path, *, target_start_sample: int
) -> dict[str, float]:
    full = read_common17_full_record(path)
    expected = materialize_model_tile(
        full, target_start_sample=target_start_sample
    ).model_input_uv
    streamed = read_common17_training_tile(
        path, target_start_sample=target_start_sample
    )
    difference = np.abs(expected.astype(np.float64) - streamed.astype(np.float64))
    return {
        "maximum_absolute_difference_uv": float(np.max(difference)),
        "mean_absolute_difference_uv": float(np.mean(difference)),
        "allclose_at_1e-4_uv": bool(np.allclose(expected, streamed, atol=1e-4, rtol=1e-6)),
    }


@dataclass(frozen=True)
class TileDraw:
    record_index: int
    target_start_sample: int
    pool: str
    patient_id: str
    round_index: int


def _tile_pool(record: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    events = record["seizure_events"]
    sample_count = int(record["target_sample_count_256hz"])
    positive: list[int] = []
    background: list[int] = []
    for start, actual in enumerate_training_target_tiles(sample_count):
        stop = start + actual
        support_start = start - CONTEXT_SAMPLES_PER_SIDE
        support_stop = stop + CONTEXT_SAMPLES_PER_SIDE
        centers = [
            (float(event["start_seconds"]) + float(event["stop_seconds"]))
            * TARGET_FS_HZ
            / 2.0
            for event in events
        ]
        if any(start <= center < stop for center in centers):
            positive.append(start)
            continue
        intersects = any(
            float(event["stop_seconds"]) * TARGET_FS_HZ > support_start
            and float(event["start_seconds"]) * TARGET_FS_HZ < support_stop
            for event in events
        )
        if not intersects:
            background.append(start)
    return positive, background


def build_epoch_draws(
    records: Sequence[Mapping[str, Any]], *, epoch_index: int, seed: int
) -> list[TileDraw]:
    """Draw four positive/four background tiles per patient when possible."""

    if epoch_index < 0 or seed < 1:
        raise ValueError("epoch_index/seed are invalid")
    pools: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {"positive": [], "background": []}
    )
    for index, record in enumerate(records):
        positive, background = _tile_pool(record)
        patient = str(record["patient_id"])
        pools[patient]["positive"].extend((index, start) for start in positive)
        pools[patient]["background"].extend((index, start) for start in background)
    per_patient: dict[str, list[tuple[int, int, str]]] = {}
    for patient in sorted(pools):
        positive = pools[patient]["positive"]
        background = pools[patient]["background"]
        if not positive and not background:
            continue
        rng = random.Random(
            int.from_bytes(
                hashlib.sha256(f"{seed}|{epoch_index}|{patient}".encode()).digest()[:8],
                "big",
            )
        )
        rng.shuffle(positive)
        rng.shuffle(background)
        quotas = (
            (4, 4)
            if positive and background
            else (8, 0) if positive else (0, 8)
        )
        rows: list[tuple[int, int, str]] = []
        for pool_name, values, quota in (
            ("positive", positive, quotas[0]),
            ("background", background, quotas[1]),
        ):
            for offset in range(quota):
                record_index, start = values[(epoch_index * max(1, quota) + offset) % len(values)]
                rows.append((record_index, start, pool_name))
        rng.shuffle(rows)
        per_patient[patient] = rows
    draws: list[TileDraw] = []
    for draw_index in range(DRAWS_PER_PATIENT_PER_EPOCH):
        ordered_patients = sorted(
            per_patient,
            key=lambda patient: hashlib.sha256(
                f"{seed}|{epoch_index}|{draw_index}|{patient}".encode()
            ).digest(),
        )
        for patient in ordered_patients:
            record_index, start, pool_name = per_patient[patient][draw_index]
            draws.append(
                TileDraw(
                    record_index=record_index,
                    target_start_sample=start,
                    pool=pool_name,
                    patient_id=patient,
                    round_index=draw_index,
                )
            )
    if not draws:
        raise ValueError("common17 epoch has no eligible tile draws")
    return draws


def build_round_aware_batch_indices(
    draws: Sequence[TileDraw], *, batch_size: int
) -> tuple[tuple[int, ...], ...]:
    """Batch within a draw round so one patient occurs at most once per batch."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("common17 batch size must be positive")
    by_round: dict[int, list[int]] = defaultdict(list)
    for index, draw in enumerate(draws):
        if not 0 <= draw.round_index < DRAWS_PER_PATIENT_PER_EPOCH:
            raise ValueError("common17 draw has an invalid round index")
        by_round[draw.round_index].append(index)
    if set(by_round) != set(range(DRAWS_PER_PATIENT_PER_EPOCH)):
        raise ValueError("common17 epoch lacks one or more draw rounds")
    batches: list[tuple[int, ...]] = []
    for round_index in range(DRAWS_PER_PATIENT_PER_EPOCH):
        indices = by_round[round_index]
        patients = [draws[index].patient_id for index in indices]
        if len(patients) != len(set(patients)):
            raise RuntimeError("common17 draw round repeats a patient")
        for offset in range(0, len(indices), batch_size):
            batch = tuple(indices[offset : offset + batch_size])
            if len({draws[index].patient_id for index in batch}) != len(batch):
                raise RuntimeError("common17 batch repeats a patient")
            batches.append(batch)
    flattened = [index for batch in batches for index in batch]
    if sorted(flattened) != list(range(len(draws))):
        raise RuntimeError("common17 round batches do not preserve every draw")
    return tuple(batches)


class Common17TileDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        draws: Sequence[TileDraw],
        tusz_root: str | Path,
    ) -> None:
        self.records = list(records)
        self.draws = list(draws)
        self.root = Path(tusz_root).resolve(strict=True)

    def __len__(self) -> int:
        return len(self.draws)

    def __getitem__(self, index: int) -> dict[str, Any]:
        draw = self.draws[index]
        record = self.records[draw.record_index]
        path = _safe_relative_edf(self.root, str(record["edf_relative_path"]))
        signal = read_common17_training_tile(
            path, target_start_sample=draw.target_start_sample
        )
        targets = build_eventnet_targets_pure_primitive(
            record["seizure_events"],
            record_sample_count=int(record["target_sample_count_256hz"]),
            target_start_sample=draw.target_start_sample,
            actual_observed_target_samples=TARGET_TILE_SAMPLES,
        )
        return {
            "signal": signal,
            "center_target": np.array(targets.center_target, copy=True),
            "duration_target": np.array(targets.duration_target, copy=True),
            "center_mask": np.array(targets.center_loss_mask, copy=True),
            "duration_mask": np.array(targets.duration_loss_mask, copy=True),
            "center_count": targets.distinct_center_count,
            "patient_id": draw.patient_id,
            "pool": draw.pool,
        }


def _collate_tiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "signal": torch.from_numpy(np.stack([row["signal"] for row in rows])),
        "center_target": torch.from_numpy(np.stack([row["center_target"] for row in rows])),
        "duration_target": torch.from_numpy(np.stack([row["duration_target"] for row in rows])),
        "center_mask": torch.from_numpy(np.stack([row["center_mask"] for row in rows])),
        "duration_mask": torch.from_numpy(np.stack([row["duration_mask"] for row in rows])),
        "center_count": [int(row["center_count"]) for row in rows],
        "patient_id": [str(row["patient_id"]) for row in rows],
        "pool": [str(row["pool"]) for row in rows],
    }


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _frozen_training_hyperparameters(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    checkpoint_every_steps: int,
) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "checkpoint_every_steps": checkpoint_every_steps,
        "gradient_clip_global_L2_norm": GRADIENT_CLIP_GLOBAL_L2_NORM,
        "cuda_autocast_dtype": CUDA_AUTOCAST_DTYPE,
        "training_loss_dtype": TRAINING_LOSS_DTYPE,
        "gradient_scaler_enabled": GRADIENT_SCALER_ENABLED,
        "cpu_autocast_enabled": False,
        "epoch_plan_id": EPOCH_PLAN_ID,
    }


def _validate_checkpoint_control_plane(
    checkpoint: Mapping[str, Any],
    *,
    manifest_receipt_sha256: str,
    expected_hyperparameters: Mapping[str, Any] | None,
    require_training_complete: bool,
) -> None:
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("manifest_receipt_sha256") != manifest_receipt_sha256
        or checkpoint.get("common17_channel_order")
        != list(COMMON17_CHANNEL_ORDER)
        or checkpoint.get("FZ_or_PZ_model_axis_present") is not False
        or checkpoint.get("epoch_plan_id") != EPOCH_PLAN_ID
    ):
        raise ValueError("common17 checkpoint lineage drifted")
    hyperparameters = checkpoint.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("common17 checkpoint lacks frozen hyperparameters")
    invariant_hyperparameters = {
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "gradient_clip_global_L2_norm": GRADIENT_CLIP_GLOBAL_L2_NORM,
        "cuda_autocast_dtype": CUDA_AUTOCAST_DTYPE,
        "training_loss_dtype": TRAINING_LOSS_DTYPE,
        "gradient_scaler_enabled": GRADIENT_SCALER_ENABLED,
        "cpu_autocast_enabled": False,
        "epoch_plan_id": EPOCH_PLAN_ID,
    }
    if any(
        hyperparameters.get(key) != expected
        for key, expected in invariant_hyperparameters.items()
    ):
        raise ValueError("common17 checkpoint trainer contract drifted")
    if expected_hyperparameters is not None and dict(hyperparameters) != dict(
        expected_hyperparameters
    ):
        raise ValueError("common17 resume hyperparameters or batch geometry drifted")
    requested_epochs = hyperparameters.get("epochs")
    next_epoch = checkpoint.get("next_epoch")
    next_batch = checkpoint.get("next_batch")
    if (
        isinstance(requested_epochs, bool)
        or not isinstance(requested_epochs, int)
        or requested_epochs < 1
        or isinstance(next_epoch, bool)
        or not isinstance(next_epoch, int)
        or not 0 <= next_epoch <= requested_epochs
        or isinstance(next_batch, bool)
        or not isinstance(next_batch, int)
        or next_batch < 0
        or (next_epoch == requested_epochs and next_batch != 0)
    ):
        raise ValueError("common17 checkpoint cursor is invalid")
    expected_complete = next_epoch == requested_epochs and next_batch == 0
    if checkpoint.get("training_complete") is not expected_complete:
        raise ValueError("common17 checkpoint completion state drifted")
    if require_training_complete and not expected_complete:
        raise PermissionError("common17 evaluation requires completed training")


def train_common17(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    resume: bool,
    checkpoint_every_steps: int,
    maximum_steps: int | None = None,
    require_complete_manifest: bool = True,
) -> dict[str, Any]:
    if (
        epochs < 1
        or batch_size < 1
        or num_workers < 0
        or learning_rate <= 0
        or seed < 1
        or checkpoint_every_steps < 1
        or weight_decay != DEFAULT_WEIGHT_DECAY
    ):
        raise ValueError("common17 training hyperparameters are invalid")
    if require_complete_manifest and (
        epochs != FORMAL_EPOCHS
        or batch_size != FORMAL_BATCH_SIZE
        or learning_rate != FORMAL_LEARNING_RATE
        or seed != FORMAL_SEED
        or checkpoint_every_steps != FORMAL_CHECKPOINT_EVERY_STEPS
    ):
        raise ValueError("formal common17 hyperparameters drifted from frozen config")
    frozen_hyperparameters = _frozen_training_hyperparameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        checkpoint_every_steps=checkpoint_every_steps,
    )
    manifest_source = Path(manifest_path).resolve(strict=True)
    manifest = load_common17_manifest(
        manifest_source, require_complete=require_complete_manifest
    )
    records = [row for row in manifest["records"] if row["model_split"] == "source_train"]
    root = Path(manifest["source_bindings"]["tusz_root"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "last.pt"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, initialization = build_randomly_initialized_model(
        variant_id=EN17_VARIANT_ID, outer_fold=0, stage="final_refit"
    )
    run_device = torch.device(device)
    if run_device.type == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("formal common17 CUDA training requires bfloat16 support")
    model.to(run_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # FP16 overflows on the raw-uV first batches before clipping.  A6000-class
    # CUDA runs therefore use BF16 forward range, FP32 loss arithmetic, and no
    # dynamic scaler.  CPU runs use ordinary FP32 without autocast.
    scaler = torch.amp.GradScaler("cuda", enabled=GRADIENT_SCALER_ENABLED)
    start_epoch = 0
    start_batch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    if resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_checkpoint_control_plane(
            checkpoint,
            manifest_receipt_sha256=manifest["receipt_sha256"],
            expected_hyperparameters=frozen_hyperparameters,
            require_training_complete=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["next_epoch"])
        start_batch = int(checkpoint["next_batch"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint.get("history", []))
    began = time.perf_counter()
    stop_requested = False
    for epoch_index in range(start_epoch, epochs):
        draws = build_epoch_draws(records, epoch_index=epoch_index, seed=seed)
        dataset = Common17TileDataset(records=records, draws=draws, tusz_root=root)
        batch_indices = build_round_aware_batch_indices(
            draws, batch_size=batch_size
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_indices,
            num_workers=num_workers,
            collate_fn=_collate_tiles,
            pin_memory=run_device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        epoch_loss = 0.0
        epoch_center = 0.0
        epoch_duration = 0.0
        completed_batches = 0
        positive_draws = 0
        background_draws = 0
        gradient_norm_sum = 0.0
        gradient_norm_maximum = 0.0
        gradient_clipped_steps = 0
        epoch_began = time.perf_counter()
        if epoch_index == start_epoch and start_batch >= len(loader):
            raise ValueError("common17 checkpoint batch cursor exceeds epoch geometry")
        for batch_index, batch in enumerate(loader):
            if epoch_index == start_epoch and batch_index < start_batch:
                continue
            signal = batch["signal"].to(run_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=run_device.type,
                dtype=torch.bfloat16,
                enabled=run_device.type == "cuda",
            ):
                center_logits, duration_logits = model.forward_logits(signal)
            with torch.autocast(
                device_type=run_device.type,
                enabled=False,
            ):
                loss_result = eventnet_multitask_loss_from_logits_pure_primitive(
                    center_logits.float(),
                    duration_logits.float(),
                    batch["center_target"],
                    batch["duration_target"],
                    batch["center_mask"],
                    batch["duration_mask"],
                    batch["center_count"],
                    patient_keys=batch["patient_id"],
                )
            scaler.scale(loss_result.loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_GLOBAL_L2_NORM,
                error_if_nonfinite=True,
            )
            gradient_norm = float(gradient_norm_tensor.detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            completed_batches += 1
            epoch_loss += float(loss_result.loss.detach().cpu())
            epoch_center += float(loss_result.center_loss.detach().cpu())
            epoch_duration += float(loss_result.duration_loss.detach().cpu())
            positive_draws += sum(pool == "positive" for pool in batch["pool"])
            background_draws += sum(pool == "background" for pool in batch["pool"])
            gradient_norm_sum += gradient_norm
            gradient_norm_maximum = max(gradient_norm_maximum, gradient_norm)
            gradient_clipped_steps += int(
                gradient_norm > GRADIENT_CLIP_GLOBAL_L2_NORM
            )
            next_epoch = epoch_index
            next_batch = batch_index + 1
            if next_batch >= len(loader):
                next_epoch = epoch_index + 1
                next_batch = 0
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "manifest_receipt_sha256": manifest["receipt_sha256"],
                "manifest_file_sha256": _file_sha256(manifest_source),
                "common17_channel_order": list(COMMON17_CHANNEL_ORDER),
                "FZ_or_PZ_model_axis_present": False,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "next_epoch": next_epoch,
                "next_batch": next_batch,
                "training_complete": bool(
                    next_epoch == epochs and next_batch == 0
                ),
                "global_step": global_step,
                "history": history,
                "hyperparameters": frozen_hyperparameters,
                "epoch_plan_id": EPOCH_PLAN_ID,
                "round_aware_batch_sampler": True,
                "one_tile_per_patient_per_batch": True,
                "cuda_autocast_dtype": CUDA_AUTOCAST_DTYPE,
                "training_loss_dtype": TRAINING_LOSS_DTYPE,
                "gradient_scaler_enabled": GRADIENT_SCALER_ENABLED,
                "last_gradient_global_L2_norm_before_clip": gradient_norm,
                "initialization_receipt": initialization,
            }
            if global_step % checkpoint_every_steps == 0:
                _atomic_torch_save(checkpoint, checkpoint_path)
            if maximum_steps is not None and global_step >= maximum_steps:
                _atomic_torch_save(checkpoint, checkpoint_path)
                stop_requested = True
                break
        elapsed = time.perf_counter() - epoch_began
        if completed_batches:
            history.append(
                {
                    "epoch_index": epoch_index,
                    "completed_batches_this_invocation": completed_batches,
                    "mean_loss": epoch_loss / completed_batches,
                    "mean_center_loss": epoch_center / completed_batches,
                    "mean_duration_loss": epoch_duration / completed_batches,
                    "positive_draws": positive_draws,
                    "background_draws": background_draws,
                    "mean_gradient_global_L2_norm_before_clip": (
                        gradient_norm_sum / completed_batches
                    ),
                    "maximum_gradient_global_L2_norm_before_clip": (
                        gradient_norm_maximum
                    ),
                    "gradient_clip_global_L2_norm": (
                        GRADIENT_CLIP_GLOBAL_L2_NORM
                    ),
                    "gradient_clipped_step_count": gradient_clipped_steps,
                    "round_aware_batch_count": len(batch_indices),
                    "elapsed_seconds": elapsed,
                    "tiles_per_second": (positive_draws + background_draws) / elapsed,
                }
            )
            checkpoint["history"] = history
            _atomic_torch_save(checkpoint, checkpoint_path)
        if stop_requested:
            break
        start_batch = 0
    receipt = _content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "training_invocation",
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "complete_manifest_required": require_complete_manifest,
            "formal_hyperparameter_contract_enforced": require_complete_manifest,
            "common17_channel_order": list(COMMON17_CHANNEL_ORDER),
            "FZ_or_PZ_model_axis_present": False,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_exists": checkpoint_path.is_file(),
            "global_step": global_step,
            "training_complete": bool(
                checkpoint_path.is_file()
                and checkpoint.get("training_complete") is True
            ),
            "requested_epochs": epochs,
            "maximum_steps": maximum_steps,
            "stopped_by_maximum_steps": stop_requested,
            "history": history,
            "hyperparameters": frozen_hyperparameters,
            "epoch_plan_id": EPOCH_PLAN_ID,
            "wall_seconds": time.perf_counter() - began,
            "receipt_sha256": _PENDING,
        }
    )
    (output / "training_receipt.json").write_bytes(_canonical_json_bytes(receipt) + b"\n")
    return receipt


def _load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    *,
    manifest_receipt_sha256: str,
) -> tuple[EventNetCleanroomUNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path.resolve(strict=True), map_location="cpu", weights_only=False)
    _validate_checkpoint_control_plane(
        checkpoint,
        manifest_receipt_sha256=manifest_receipt_sha256,
        expected_hyperparameters=None,
        require_training_complete=True,
    )
    model = EventNetCleanroomUNet(len(COMMON17_CHANNEL_ORDER))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _predict_record_peaks(
    model: EventNetCleanroomUNet,
    path: str | Path,
    *,
    expected_target_sample_count: int,
    device: torch.device,
    inference_batch_size: int,
    minimum_threshold: float,
    smoothing_sigma_samples: int,
    minimum_peak_distance_seconds: int,
) -> tuple[list[dict[str, float]], dict[str, Any], float, float]:
    source = Path(path).resolve(strict=True)
    reader = pyedflib.EdfReader(str(source))
    try:
        layout = _edf_layout(reader)
        up, down = _resample_ratio(layout.sampling_rate)
        target_sample_count = (layout.source_sample_count * up) // down
        if target_sample_count != expected_target_sample_count:
            raise ValueError("manifest/EDF common17 target sample count drifted")
        tile_rows = enumerate_target_tiles(target_sample_count)
    # Model forwards remain tile-bounded, but decoding must operate on one
    # absolute full-record posterior.  Per-tile smoothing/find_peaks would
    # reset the Gaussian kernel and minimum-distance constraint every 120 s,
    # causing both missed boundary peaks and duplicate alarms across a tile
    # boundary.
        center_posterior = np.empty(target_sample_count, dtype=np.float32)
        duration_posterior = np.empty(target_sample_count, dtype=np.float32)
        posterior_coverage = np.zeros(target_sample_count, dtype=np.uint8)
        inference_seconds = 0.0
        io_seconds = 0.0
        for offset in range(0, len(tile_rows), inference_batch_size):
            batch_rows = tile_rows[offset : offset + inference_batch_size]
            arrays = []
            actuals = []
            began_io = time.perf_counter()
            for start, expected_actual in batch_rows:
                tile, actual, observed_target_count = (
                    _read_common17_inference_tile_from_reader(
                        reader,
                        layout,
                        target_start_sample=start,
                    )
                )
                if (
                    actual != expected_actual
                    or observed_target_count != target_sample_count
                ):
                    raise RuntimeError("common17 inference tile roster drifted")
                arrays.append(tile)
                actuals.append(actual)
            io_seconds += time.perf_counter() - began_io
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            began = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                center_logits, duration_logits = model.forward_logits(tensor)
                center_batch = torch.sigmoid(center_logits).float().cpu().numpy()[:, 0]
                duration_batch = torch.sigmoid(duration_logits).float().cpu().numpy()[:, 0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - began
            for row_index, ((start, _actual), actual) in enumerate(
                zip(batch_rows, actuals)
            ):
                stop = start + actual
                if bool(np.any(posterior_coverage[start:stop])):
                    raise RuntimeError("common17 posterior tiles overlap")
                center_posterior[start:stop] = center_batch[row_index, :actual]
                duration_posterior[start:stop] = duration_batch[row_index, :actual]
                posterior_coverage[start:stop] = 1
        _require_exact_posterior_coverage(posterior_coverage)
    finally:
        reader.close()
    peaks_out, candidate_cache = _decode_global_posteriors_to_replayable_candidates(
        center_posterior,
        duration_posterior,
        minimum_threshold=minimum_threshold,
        smoothing_sigma_samples=smoothing_sigma_samples,
        minimum_peak_distance_seconds=minimum_peak_distance_seconds,
    )
    return peaks_out, candidate_cache, inference_seconds, io_seconds


def _extract_global_pre_nms_candidates(
    center_posterior: np.ndarray,
    duration_posterior: np.ndarray,
    *,
    minimum_threshold: float,
    smoothing_sigma_samples: int,
) -> list[dict[str, float | int]]:
    """Preserve reference-free local maxima before any distance suppression.

    The adjacent valley values are sufficient to replay fixed-distance NMS and
    to evaluate a preregistered valley-separation deblender on CPU.  They are
    intentionally measured from the full-record smoothed posterior, before
    labels, annotations, or a record-specific decoder can be consulted.
    """

    center = np.asarray(center_posterior)
    duration = np.asarray(duration_posterior)
    if (
        center.ndim != 1
        or duration.ndim != 1
        or center.shape != duration.shape
        or center.size < 1
        or not np.isfinite(center).all()
        or not np.isfinite(duration).all()
    ):
        raise ValueError("common17 full-record posteriors are malformed")
    if (
        not 0.0 < minimum_threshold < 1.0
        or isinstance(smoothing_sigma_samples, bool)
        or not isinstance(smoothing_sigma_samples, int)
        or smoothing_sigma_samples < 1
    ):
        raise ValueError("common17 pre-NMS candidate parameters are invalid")
    smoothed = gaussian_filter1d(center, smoothing_sigma_samples)
    internal_peaks, properties = find_peaks(smoothed, height=minimum_threshold)
    candidates = {
        int(index): float(height)
        for index, height in zip(internal_peaks, properties["peak_heights"])
    }
    # scipy excludes endpoints; a strict one-sided maximum at the physical
    # record boundary remains a legitimate reference-free candidate.
    if center.size == 1:
        if float(smoothed[0]) >= minimum_threshold:
            candidates[0] = float(smoothed[0])
    else:
        if (
            float(smoothed[0]) >= minimum_threshold
            and float(smoothed[0]) > float(smoothed[1])
        ):
            candidates[0] = float(smoothed[0])
        last = center.size - 1
        if (
            float(smoothed[last]) >= minimum_threshold
            and float(smoothed[last]) > float(smoothed[last - 1])
        ):
            candidates[last] = float(smoothed[last])
    ordered = sorted(candidates.items())
    if not ordered:
        return []
    pair_valleys = [
        float(np.min(smoothed[left_index : right_index + 1]))
        for (left_index, _), (right_index, _) in zip(ordered, ordered[1:])
    ]
    output: list[dict[str, float | int]] = []
    for position, (index, height) in enumerate(ordered):
        left_valley = (
            float(np.min(smoothed[: index + 1]))
            if position == 0
            else pair_valleys[position - 1]
        )
        right_valley = (
            float(np.min(smoothed[index:]))
            if position == len(ordered) - 1
            else pair_valleys[position]
        )
        output.append(
            {
                "center_sample": int(index),
                "center_probability": float(height),
                "duration_fraction": float(duration[index]),
                "left_valley_probability": left_valley,
                "right_valley_probability": right_valley,
            }
        )
    return output


def _apply_minimum_distance_nms(
    candidates: Sequence[Mapping[str, float | int]],
    *,
    minimum_peak_distance_seconds: int,
) -> list[dict[str, float | int]]:
    """Replay probability-priority NMS from a pre-NMS candidate cache."""

    if (
        isinstance(minimum_peak_distance_seconds, bool)
        or not isinstance(minimum_peak_distance_seconds, int)
        or minimum_peak_distance_seconds < 1
    ):
        raise ValueError("common17 NMS distance is invalid")
    normalized: list[dict[str, float | int]] = []
    seen_samples: set[int] = set()
    for row in candidates:
        sample = int(row["center_sample"])
        probability = float(row["center_probability"])
        values = (
            probability,
            float(row["duration_fraction"]),
            float(row["left_valley_probability"]),
            float(row["right_valley_probability"]),
        )
        if sample < 0 or sample in seen_samples or not all(map(math.isfinite, values)):
            raise ValueError("common17 pre-NMS candidate cache is malformed")
        seen_samples.add(sample)
        normalized.append(
            {
                "center_sample": sample,
                "center_probability": probability,
                "duration_fraction": values[1],
                "left_valley_probability": values[2],
                "right_valley_probability": values[3],
            }
        )
    minimum_distance = minimum_peak_distance_seconds * TARGET_FS_HZ
    kept: list[dict[str, float | int]] = []
    for candidate in sorted(
        normalized,
        key=lambda row: (-float(row["center_probability"]), int(row["center_sample"])),
    ):
        if all(
            abs(int(candidate["center_sample"]) - int(previous["center_sample"]))
            >= minimum_distance
            for previous in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda row: int(row["center_sample"]))


def _pre_nms_candidate_cache_payload(
    candidates: Sequence[Mapping[str, float | int]],
    *,
    minimum_threshold: float,
    smoothing_sigma_samples: int,
) -> dict[str, Any]:
    rows = list(candidates)
    return {
        "schema_version": PRE_NMS_CANDIDATE_CACHE_SCHEMA_VERSION,
        "stage": "full_record_smoothed_center_posterior_before_distance_nms",
        "target_fs_hz": TARGET_FS_HZ,
        "minimum_peak_threshold": float(minimum_threshold),
        "smoothing_sigma_samples": int(smoothing_sigma_samples),
        "candidate_count": len(rows),
        "center_sample": [int(row["center_sample"]) for row in rows],
        "center_probability": [float(row["center_probability"]) for row in rows],
        "duration_fraction": [float(row["duration_fraction"]) for row in rows],
        "left_valley_probability": [
            float(row["left_valley_probability"]) for row in rows
        ],
        "right_valley_probability": [
            float(row["right_valley_probability"]) for row in rows
        ],
        "reference_or_annotation_used": False,
        "dense_posterior_preserved": False,
        "replay_scope": (
            "threshold_and_minimum_distance_nms_and_adjacent_valley_deblending"
        ),
    }


def _valid_pre_nms_candidate_cache(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if (
        payload.get("schema_version") != PRE_NMS_CANDIDATE_CACHE_SCHEMA_VERSION
        or payload.get("stage")
        != "full_record_smoothed_center_posterior_before_distance_nms"
        or payload.get("target_fs_hz") != TARGET_FS_HZ
        or payload.get("reference_or_annotation_used") is not False
        or payload.get("dense_posterior_preserved") is not False
        or payload.get("replay_scope")
        != "threshold_and_minimum_distance_nms_and_adjacent_valley_deblending"
    ):
        return False
    cache_threshold = payload.get("minimum_peak_threshold")
    cache_sigma = payload.get("smoothing_sigma_samples")
    if (
        isinstance(cache_threshold, bool)
        or not isinstance(cache_threshold, (int, float))
        or not math.isfinite(float(cache_threshold))
        or not 0.0 < float(cache_threshold) < 1.0
        or isinstance(cache_sigma, bool)
        or not isinstance(cache_sigma, int)
        or cache_sigma < 1
    ):
        return False
    count = payload.get("candidate_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return False
    columns = (
        "center_sample",
        "center_probability",
        "duration_fraction",
        "left_valley_probability",
        "right_valley_probability",
    )
    if not all(
        isinstance(payload.get(key), list) and len(payload[key]) == count
        for key in columns
    ):
        return False
    samples = payload["center_sample"]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in samples
    ) or any(left >= right for left, right in zip(samples, samples[1:])):
        return False
    numeric_columns = columns[1:]
    return all(
        all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in payload[key]
        )
        for key in numeric_columns
    )


def _replay_pre_nms_candidate_cache(
    payload: Mapping[str, Any],
    *,
    threshold: float,
    minimum_peak_distance_seconds: int,
) -> list[dict[str, float]]:
    """CPU-only threshold/distance replay from the versioned v4 cache."""

    if not _valid_pre_nms_candidate_cache(payload):
        raise ValueError("common17 pre-NMS candidate cache is invalid")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("common17 replay threshold is invalid")
    rows = [
        {
            "center_sample": payload["center_sample"][index],
            "center_probability": payload["center_probability"][index],
            "duration_fraction": payload["duration_fraction"][index],
            "left_valley_probability": payload["left_valley_probability"][index],
            "right_valley_probability": payload["right_valley_probability"][index],
        }
        for index in range(int(payload["candidate_count"]))
        if float(payload["center_probability"][index]) >= threshold
    ]
    kept = _apply_minimum_distance_nms(
        rows,
        minimum_peak_distance_seconds=minimum_peak_distance_seconds,
    )
    return [
        {
            "center_seconds": int(row["center_sample"]) / TARGET_FS_HZ,
            "center_probability": float(row["center_probability"]),
            "duration_fraction": float(row["duration_fraction"]),
        }
        for row in kept
    ]


def _decode_global_posteriors_to_replayable_candidates(
    center_posterior: np.ndarray,
    duration_posterior: np.ndarray,
    *,
    minimum_threshold: float,
    smoothing_sigma_samples: int,
    minimum_peak_distance_seconds: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    candidates = _extract_global_pre_nms_candidates(
        center_posterior,
        duration_posterior,
        minimum_threshold=minimum_threshold,
        smoothing_sigma_samples=smoothing_sigma_samples,
    )
    kept = _apply_minimum_distance_nms(
        candidates,
        minimum_peak_distance_seconds=minimum_peak_distance_seconds,
    )
    peaks = [
        {
            "center_seconds": int(row["center_sample"]) / TARGET_FS_HZ,
            "center_probability": float(row["center_probability"]),
            "duration_fraction": float(row["duration_fraction"]),
        }
        for row in kept
    ]
    cache = _pre_nms_candidate_cache_payload(
        candidates,
        minimum_threshold=minimum_threshold,
        smoothing_sigma_samples=smoothing_sigma_samples,
    )
    return peaks, cache


def _decode_global_posteriors_to_peaks(
    center_posterior: np.ndarray,
    duration_posterior: np.ndarray,
    *,
    minimum_threshold: float,
    smoothing_sigma_samples: int,
    minimum_peak_distance_seconds: int,
) -> list[dict[str, float]]:
    """Decode one absolute posterior timeline, never independent tile pieces."""
    peaks, _candidate_cache = _decode_global_posteriors_to_replayable_candidates(
        center_posterior,
        duration_posterior,
        minimum_threshold=minimum_threshold,
        smoothing_sigma_samples=smoothing_sigma_samples,
        minimum_peak_distance_seconds=minimum_peak_distance_seconds,
    )
    return peaks


def _require_exact_posterior_coverage(coverage: np.ndarray) -> None:
    value = np.asarray(coverage)
    if (
        value.ndim != 1
        or value.size < 1
        or value.dtype != np.dtype(np.uint8)
        or not bool(np.all(value == 1))
    ):
        raise RuntimeError("common17 full-record posterior coverage is not exactly one")


def _decode_peaks(
    peaks: Sequence[Mapping[str, float]],
    *,
    threshold: float,
    recording_duration_seconds: float,
) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    for peak in peaks:
        if float(peak["center_probability"]) < threshold:
            continue
        duration = min(
            float(MAXIMUM_DURATION_SECONDS),
            max(1.0 / TARGET_FS_HZ, float(peak["duration_fraction"]) * MAXIMUM_DURATION_SECONDS),
        )
        center = float(peak["center_seconds"])
        start = max(0.0, center - duration / 2.0)
        stop = min(recording_duration_seconds, center + duration / 2.0)
        if stop > start:
            events.append({"start_seconds": start, "stop_seconds": stop})
    merged: list[dict[str, float]] = []
    for event in sorted(events, key=lambda row: (row["start_seconds"], row["stop_seconds"])):
        if not merged or event["start_seconds"] > merged[-1]["stop_seconds"]:
            merged.append(dict(event))
        else:
            merged[-1]["stop_seconds"] = max(merged[-1]["stop_seconds"], event["stop_seconds"])
    return merged


def _seizure_record_recall(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    denominator = 0
    hit_count = 0
    for row in rows:
        references = row["reference_events"]
        if not references:
            continue
        denominator += 1
        predictions = row["predicted_events"]
        if any(
            float(prediction["stop_seconds"]) > float(reference["start_seconds"])
            and float(prediction["start_seconds"]) < float(reference["stop_seconds"])
            for reference in references
            for prediction in predictions
        ):
            hit_count += 1
    return {
        "seizure_recording_denominator": denominator,
        "seizure_recording_hit_count": hit_count,
        "rate": None if denominator == 0 else hit_count / denominator,
    }


def evaluate_common17_source_dev(
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: str,
    inference_batch_size: int,
    thresholds: Sequence[float],
    maximum_records: int | None = None,
    reverse_record_order: bool = False,
    require_complete_manifest: bool = True,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve(strict=True)
    manifest = load_common17_manifest(
        manifest_source, require_complete=require_complete_manifest
    )
    root = Path(manifest["source_bindings"]["tusz_root"])
    records = [row for row in manifest["records"] if row["model_split"] == "source_dev"]
    if maximum_records is not None:
        records = records[:maximum_records]
    if reverse_record_order:
        records.reverse()
    if not records:
        raise ValueError("common17 source-dev evaluation roster is empty")
    candidate_thresholds = sorted({float(value) for value in thresholds})
    if not candidate_thresholds or candidate_thresholds[0] <= 0 or candidate_thresholds[-1] >= 1:
        raise ValueError("common17 thresholds must lie strictly between zero and one")
    run_device = torch.device(device)
    checkpoint_source = Path(checkpoint_path)
    model, checkpoint = _load_model_checkpoint(
        checkpoint_source,
        run_device,
        manifest_receipt_sha256=manifest["receipt_sha256"],
    )
    checkpoint_file_sha256 = _file_sha256(checkpoint_source.resolve(strict=True))
    output = Path(output_dir)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    minimum_threshold = candidate_thresholds[0]
    new_inference_seconds = 0.0
    new_io_seconds = 0.0
    new_pipeline_seconds = 0.0
    completed = 0
    for record in records:
        record_id = str(record["analysis_identity_id"])
        target = prediction_dir / f"{record_id}.json.gz"
        if target.is_file():
            with gzip.open(target, "rt", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("schema_version") == LEGACY_POST_NMS_PREDICTION_SCHEMA_VERSION:
                raise RuntimeError(
                    "legacy post-NMS-only prediction is frozen and cannot be "
                    "upgraded in place; choose a new output directory for the "
                    "replayable pre-NMS schema"
                )
            if (
                existing.get("schema_version") == PREDICTION_SCHEMA_VERSION
                and existing.get("checkpoint_global_step") == checkpoint["global_step"]
                and existing.get("checkpoint_file_sha256") == checkpoint_file_sha256
                and existing.get("minimum_peak_threshold") == minimum_threshold
                and existing.get("smoothing_sigma_samples")
                == DEFAULT_SMOOTHING_SIGMA_SAMPLES
                and existing.get("minimum_peak_distance_seconds")
                == DEFAULT_MINIMUM_PEAK_DISTANCE_SECONDS
                and _valid_pre_nms_candidate_cache(
                    existing.get("pre_nms_candidate_cache")
                )
                and existing["pre_nms_candidate_cache"].get(
                    "minimum_peak_threshold"
                )
                == minimum_threshold
                and existing["pre_nms_candidate_cache"].get(
                    "smoothing_sigma_samples"
                )
                == DEFAULT_SMOOTHING_SIGMA_SAMPLES
            ):
                completed += 1
                continue
        edf_path = _safe_relative_edf(root, str(record["edf_relative_path"]))
        began_pipeline = time.perf_counter()
        peaks, candidate_cache, inference_seconds, io_seconds = _predict_record_peaks(
            model,
            edf_path,
            expected_target_sample_count=int(record["target_sample_count_256hz"]),
            device=run_device,
            inference_batch_size=inference_batch_size,
            minimum_threshold=minimum_threshold,
            smoothing_sigma_samples=DEFAULT_SMOOTHING_SIGMA_SAMPLES,
            minimum_peak_distance_seconds=DEFAULT_MINIMUM_PEAK_DISTANCE_SECONDS,
        )
        pipeline_seconds = time.perf_counter() - began_pipeline
        new_inference_seconds += inference_seconds
        new_io_seconds += io_seconds
        new_pipeline_seconds += pipeline_seconds
        payload = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "analysis_identity_id": record_id,
            "patient_id": record["patient_id"],
            "recording_duration_seconds": float(
                Fraction(*record["recording_duration_seconds_fraction"])
            ),
            "common17_channel_order": list(COMMON17_CHANNEL_ORDER),
            "FZ_or_PZ_model_axis_present": False,
            "checkpoint_global_step": checkpoint["global_step"],
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "minimum_peak_threshold": minimum_threshold,
            "smoothing_sigma_samples": DEFAULT_SMOOTHING_SIGMA_SAMPLES,
            "minimum_peak_distance_seconds": DEFAULT_MINIMUM_PEAK_DISTANCE_SECONDS,
            "pre_nms_candidate_cache": candidate_cache,
            "runtime": {
                "model_inference_seconds": inference_seconds,
                "EEG_IO_and_resample_seconds": io_seconds,
                "end_to_end_pipeline_seconds": pipeline_seconds,
            },
            "peaks": peaks,
        }
        # A PID-qualified staging name keeps independent prediction workers
        # from sharing the same temporary path.  The final record path remains
        # atomic and content-validated on every resume.
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        os.replace(temporary, target)
        completed += 1
    if completed != len(records):
        raise RuntimeError("common17 source-dev prediction denominator is incomplete")
    record_by_id = {str(row["analysis_identity_id"]): row for row in records}
    predictions: dict[str, dict[str, Any]] = {}
    for target in prediction_dir.glob("*.json.gz"):
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        identity = payload.get("analysis_identity_id")
        if (
            identity in record_by_id
            and payload.get("schema_version") == PREDICTION_SCHEMA_VERSION
            and payload.get("checkpoint_global_step") == checkpoint["global_step"]
            and payload.get("checkpoint_file_sha256") == checkpoint_file_sha256
            and payload.get("minimum_peak_threshold") == minimum_threshold
            and payload.get("smoothing_sigma_samples")
            == DEFAULT_SMOOTHING_SIGMA_SAMPLES
            and payload.get("minimum_peak_distance_seconds")
            == DEFAULT_MINIMUM_PEAK_DISTANCE_SECONDS
            and _valid_pre_nms_candidate_cache(
                payload.get("pre_nms_candidate_cache")
            )
            and payload["pre_nms_candidate_cache"].get("minimum_peak_threshold")
            == minimum_threshold
            and payload["pre_nms_candidate_cache"].get(
                "smoothing_sigma_samples"
            )
            == DEFAULT_SMOOTHING_SIGMA_SAMPLES
        ):
            predictions[str(identity)] = payload
    if set(predictions) != set(record_by_id):
        raise RuntimeError("common17 source-dev prediction files do not close the roster")
    total_inference_seconds = 0.0
    total_io_seconds = 0.0
    total_pipeline_seconds = 0.0
    for prediction in predictions.values():
        runtime = prediction.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError("common17 prediction lacks runtime accounting")
        values = [
            float(runtime.get("model_inference_seconds", math.nan)),
            float(runtime.get("EEG_IO_and_resample_seconds", math.nan)),
            float(runtime.get("end_to_end_pipeline_seconds", math.nan)),
        ]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("common17 prediction runtime is invalid")
        total_inference_seconds += values[0]
        total_io_seconds += values[1]
        total_pipeline_seconds += values[2]
    grid: list[dict[str, Any]] = []
    for threshold in candidate_thresholds:
        metric_rows = []
        by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for identity, record in record_by_id.items():
            prediction = predictions[identity]
            duration = float(Fraction(*record["recording_duration_seconds_fraction"]))
            row = {
                "duration_seconds": duration,
                "reference_events": record["seizure_events"],
                "predicted_events": _decode_peaks(
                    prediction["peaks"],
                    threshold=threshold,
                    recording_duration_seconds=duration,
                ),
            }
            metric_rows.append(row)
            by_patient[str(record["patient_id"])].append(row)
        pooled = _aggregate_metrics(metric_rows, tolerances=(1.0, 3.0, 5.0, 10.0))
        pooled["seizure_record_recall"] = _seizure_record_recall(metric_rows)
        patient_metrics = [_aggregate_metrics(rows, tolerances=(1.0, 3.0, 5.0, 10.0)) for rows in by_patient.values()]
        macro_fields = {}
        for field in ("event_sensitivity", "event_precision", "event_f1", "alarm_false_alarms_per_24h"):
            values = [float(metric[field]) for metric in patient_metrics if metric[field] is not None]
            macro_fields[field] = None if not values else sum(values) / len(values)
        grid.append(
            {
                "center_threshold": threshold,
                "pooled": pooled,
                "patient_macro": macro_fields,
            }
        )
    best = max(
        grid,
        key=lambda row: (
            -1.0 if row["pooled"]["event_f1"] is None else row["pooled"]["event_f1"],
            -float("inf") if row["pooled"]["alarm_false_alarms_per_24h"] is None else -row["pooled"]["alarm_false_alarms_per_24h"],
        ),
    )
    evaluated_hours = sum(
        float(Fraction(*row["recording_duration_seconds_fraction"])) for row in records
    ) / 3600.0
    complete_eval = (
        maximum_records is None
        and require_complete_manifest
        and len(records) == manifest["split_summaries"]["source_dev"]["recording_count"]
    )
    evaluated_seconds = evaluated_hours * 3600.0
    warm_model_rtf = total_inference_seconds / evaluated_seconds
    end_to_end_rtf = total_pipeline_seconds / evaluated_seconds
    best_pooled = best["pooled"]
    user_facing_best = {
        "scope": "source_dev_diagnostic_threshold_selection_not_held_out_eval",
        "center_threshold": best["center_threshold"],
        "seizure_record_recall": best_pooled["seizure_record_recall"],
        "pooled_event_sensitivity": best_pooled["event_sensitivity"],
        "pooled_event_precision": best_pooled["event_precision"],
        "pooled_event_f1": best_pooled["event_f1"],
        "false_alarms_per_24h": best_pooled["alarm_false_alarms_per_24h"],
        "onset_hit_at_1s": best_pooled["onset_absolute_hit_rate"]["1s"],
        "onset_hit_at_3s": best_pooled["onset_absolute_hit_rate"]["3s"],
        "onset_hit_at_5s": best_pooled["onset_absolute_hit_rate"]["5s"],
        "onset_hit_at_10s": best_pooled["onset_absolute_hit_rate"]["10s"],
        "absolute_onset_error_median_matched_only_seconds": best_pooled[
            "onset_latency_seconds"
        ]["absolute_median_matched_only"],
        "warm_model_inference_RTF": warm_model_rtf,
        "end_to_end_EEG_IO_resample_inference_decode_RTF": end_to_end_rtf,
    }
    receipt = _content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "source_dev_full_record_evaluation",
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "checkpoint_path": str(checkpoint_source.resolve(strict=True)),
            "checkpoint_global_step": checkpoint["global_step"],
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "common17_channel_order": list(COMMON17_CHANNEL_ORDER),
            "FZ_or_PZ_model_axis_present": False,
            "recording_count": len(records),
            "reference_event_count": sum(len(row["seizure_events"]) for row in records),
            "recording_hours": evaluated_hours,
            "complete_source_dev_denominator": complete_eval,
            "threshold_selection_status": "source_dev_diagnostic_grid_not_source_eval",
            "metric_grid": grid,
            "best_source_dev_diagnostic_operating_point": best,
            "user_facing_best_source_dev_diagnostic_metrics": user_facing_best,
            "runtime": {
                "all_prediction_model_inference_seconds": total_inference_seconds,
                "all_prediction_EEG_IO_and_resample_seconds": total_io_seconds,
                "all_prediction_end_to_end_pipeline_seconds": total_pipeline_seconds,
                "new_prediction_model_inference_seconds": new_inference_seconds,
                "new_prediction_EEG_IO_and_resample_seconds": new_io_seconds,
                "new_prediction_end_to_end_pipeline_seconds": new_pipeline_seconds,
                "warm_model_inference_RTF": warm_model_rtf,
                "end_to_end_EEG_IO_resample_inference_decode_RTF": end_to_end_rtf,
            },
            "scope": {
                "full_record_EEG_used": True,
                "global_TERM_seiz_used_for_evaluation": True,
                "channel_annotation_or_SOZ_label_used": False,
                "EDF_plus_annotation_used": False,
                "clinical_text_or_spreadsheet_used": False,
            },
            "receipt_sha256": _PENDING,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_bytes(_canonical_json_bytes(receipt) + b"\n")
    return receipt


__all__ = [
    "COMMON17_CHANNEL_ORDER",
    "Common17TileDataset",
    "TileDraw",
    "build_epoch_draws",
    "evaluate_common17_source_dev",
    "load_common17_manifest",
    "materialize_common17_manifest",
    "read_common17_full_record",
    "read_common17_training_tile",
    "streaming_full_parity",
    "train_common17",
]
