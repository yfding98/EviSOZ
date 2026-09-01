#!/usr/bin/env python3
"""Materialize EEG-only DeepSOZ OOF one-second posteriors from TUSZ EDFs.

The script reads signal samples and channel labels only.  It never calls an EDF
annotation API and never reads seizure start/stop or SOZ fields from the input
manifest.  Official numeric patient IDs are used solely to select all
published folds which held that patient out.

This command emits posterior artifacts, not seizure alarms.  A source-dev
operating point must be frozen later, after which the model-neutral hysteresis
decoder and patient-level benchmark may be run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pyedflib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.deepsoz_temporal_adapter import (  # noqa: E402
    DEEPSOZ_CHUNK_SECONDS,
    DEEPSOZ_OOF_ENSEMBLE_SCHEMA_VERSION,
    DEEPSOZ_OVERLAP_SECONDS,
    PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256,
    DEEPSOZ_STRIDE_SECONDS,
    PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
    STANDARD_19,
    DeepSOZTemporalResearchAdapter,
    aggregate_deepsoz_oof_fold_posteriors,
    load_published_deepsoz_oof_fold_assignment,
)
from src.clinical_eeg_long_recording.canonical_detector_input_binding import (  # noqa: E402
    build_canonical_detector_input_binding,
    validate_canonical_detector_input_binding,
)
from src.clinical_eeg_long_recording.canonical_edf_materialization import (  # noqa: E402
    load_canonical_edf_record,
)


DEFAULT_MANIFEST = ROOT / "deepsoz_tusz_652_record_manifest.csv"
DEFAULT_WEIGHTS = ROOT / "models/deepsoz_official_weights"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
DEEPSOZ_DETECTOR_INPUT_CHANNEL_SCHEMA_VERSION = (
    "deepsoz_detector_input_channel_observation_v1"
)
PUBLISHED_DEEPSOZ_MISSING_CHANNEL_POLICY = (
    "zero_fill_entire_missing_channel_before_published_preprocessing"
)
PUBLISHED_DEEPSOZ_UTILS_PREPROCESS_SHA256 = (
    "e7c25de98043e02959577efecdee07f54875d3fa126114407c9c9b18fe0b4a9d"
)
DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION = (
    "deepsoz_oof_physical_binding_and_offline_time_support_v2"
)
DEEPSOZ_POSTERIOR_TIME_SUPPORT_SCHEMA_VERSION = (
    "deepsoz_offline_posterior_physical_time_support_v1"
)
DEEPSOZ_RUNTIME_RECEIPT_SCHEMA_VERSION = "deepsoz_offline_runtime_receipt_v1"
DEEPSOZ_BATCH_RUNTIME_RECEIPT_SCHEMA_VERSION = (
    "deepsoz_offline_batch_runtime_receipt_v1"
)
DEEPSOZ_PROVIDER_ID = "deepsoz_temporal_oof_candidate_v1"
DEEPSOZ_DECISION_AVAILABILITY = (
    "offline_after_complete_record_capture_preprocessing_and_all_held_out_fold_inference"
)
DEEPSOZ_TIMESTAMP_SEMANTICS = (
    "recording_relative_navigation_coordinate_not_real_time_decision_latency"
)
DEEPSOZ_PARTIAL_TAIL_POLICY = (
    "emit_unusable_coverage_marker_with_zero_sentinel_not_model_probability"
)
_UNIT_TO_MICROVOLTS = {
    "v": 1_000_000.0,
    "mv": 1_000.0,
    "uv": 1.0,
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _elapsed_seconds(start_ns: int) -> float:
    return max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000_000.0)


def _normalize_label(value: object) -> str:
    label = str(value).strip().upper().replace("EEG ", "")
    label = label.split("-")[0].strip().replace(" ", "")
    return LEGACY_TO_MODERN.get(label, label)


def _safe_edf(root: Path, relative: str) -> Path:
    value = PurePosixPath(str(relative))
    if value.is_absolute() or ".." in value.parts or value.suffix.lower() != ".edf":
        raise ValueError(f"unsafe TUSZ EDF path: {relative!r}")
    path = root.joinpath(*value.parts).resolve(strict=True)
    path.relative_to(root)
    return path


def _normalized_physical_unit(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace(" ", "")
    )


def _read_complete_standard19(
    path: Path,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Read the published 19-channel detector view with explicit zero-fill.

    The official DeepSOZ ``readEDF`` helper initializes every requested
    channel to ``None`` and replaces an absent channel with a full-length zero
    vector before filtering and normalization.  We reproduce that detector
    compatibility policy, while recording which channels were not observed.
    Those zero-filled channels are never eligible for Findings or SOZ facts.
    """

    reader = pyedflib.EdfReader(str(path))
    try:
        bound: dict[str, int] = {}
        for index, label in enumerate(reader.getSignalLabels()):
            normalized = _normalize_label(label)
            if normalized not in STANDARD_19:
                continue
            if normalized in bound:
                raise ValueError(
                    "EDF has duplicate normalized Standard-19 detector channels"
                )
            bound[normalized] = index
        missing = [channel for channel in STANDARD_19 if channel not in bound]
        if STANDARD_19[0] not in bound:
            raise ValueError("EDF lacks FP1 required by the published zero-fill clock")
        reference_index = bound[STANDARD_19[0]]
        sampling_rate = float(reader.getSampleFrequency(reference_index))
        sample_count = int(reader.getNSamples()[reference_index])
        if not math.isfinite(sampling_rate) or sampling_rate <= 0 or sample_count < 2:
            raise ValueError("EDF has an invalid Standard-19 detector sampling grid")
        signals: list[np.ndarray] = []
        unit_by_channel: dict[str, str | None] = {}
        scale_to_microvolts_by_channel: dict[str, float | None] = {}
        for channel in STANDARD_19:
            if channel not in bound:
                signals.append(np.zeros(sample_count, dtype=np.float64))
                unit_by_channel[channel] = None
                scale_to_microvolts_by_channel[channel] = None
                continue
            index = bound[channel]
            if (
                abs(float(reader.getSampleFrequency(index)) - sampling_rate) > 1e-9
                or int(reader.getNSamples()[index]) != sample_count
            ):
                raise ValueError("EDF has mixed Standard-19 sampling grids")
            unit = _normalized_physical_unit(reader.getPhysicalDimension(index))
            if unit not in _UNIT_TO_MICROVOLTS:
                raise ValueError("EDF detector channel has an unsupported physical unit")
            scale = _UNIT_TO_MICROVOLTS[unit]
            signal = np.asarray(reader.readSignal(index), dtype=np.float64) * scale
            if signal.shape != (sample_count,) or not np.isfinite(signal).all():
                raise ValueError("EDF detector channel contains an invalid signal")
            signals.append(signal)
            unit_by_channel[channel] = unit
            scale_to_microvolts_by_channel[channel] = scale
        observed = [channel for channel in STANDARD_19 if channel in bound]
        receipt: dict[str, Any] = {
            "schema_version": DEEPSOZ_DETECTOR_INPUT_CHANNEL_SCHEMA_VERSION,
            "receipt_id": "DEEPSOZ-DETECTOR-INPUT-PENDING",
            "channel_order": list(STANDARD_19),
            "observed_channel_ids": observed,
            "imputed_channel_ids": missing,
            "observed_channel_count": len(observed),
            "imputed_channel_count": len(missing),
            "sampling_rate_hz": sampling_rate,
            "sample_count": sample_count,
            "physical_unit_after_conversion": "uV",
            "source_physical_unit_by_channel": unit_by_channel,
            "scale_to_microvolts_by_channel": scale_to_microvolts_by_channel,
            "missing_channel_imputation": bool(missing),
            "missing_channel_imputation_policy": (
                PUBLISHED_DEEPSOZ_MISSING_CHANNEL_POLICY
            ),
            "published_policy_source_sha256": (
                PUBLISHED_DEEPSOZ_UTILS_PREPROCESS_SHA256
            ),
            "all_observed_channels_share_sampling_grid": True,
            "imputed_channels_clinical_evidence_eligible": False,
            "edf_signal_labels_used": True,
            "edf_annotations_used": False,
        }
        receipt["receipt_id"] = "DSZINPUT-" + _canonical_sha256(receipt)[:24]
        return np.stack(signals), sampling_rate, receipt
    finally:
        reader.close()


def _load_canonical_bound_standard19(
    path: Path,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any], dict[str, float]]:
    """Load one signal-only root and bind the provider carrier before transforms.

    The canonical reader and the published-compatibility reader intentionally
    remain independent.  Their physical samples are compared sample-by-sample
    before DeepSOZ's whole-record resampling, zero-phase filtering, clipping or
    normalization.  Neither reader calls an EDF annotation/header identity API.
    """

    stage_started = time.perf_counter_ns()
    canonical_record = load_canonical_edf_record(path)
    canonical_elapsed = _elapsed_seconds(stage_started)

    stage_started = time.perf_counter_ns()
    eeg, sampling_rate, input_receipt = _read_complete_standard19(path)
    provider_read_elapsed = _elapsed_seconds(stage_started)

    stage_started = time.perf_counter_ns()
    binding = build_canonical_detector_input_binding(
        canonical_record=canonical_record,
        provider_id=DEEPSOZ_PROVIDER_ID,
        detector_input=eeg,
        detector_channel_ids=STANDARD_19,
        detector_sampling_rate_hz=sampling_rate,
        detector_physical_unit="uV",
        observed_channel_ids=input_receipt["observed_channel_ids"],
        imputed_channel_ids=input_receipt["imputed_channel_ids"],
        provider_input_receipt_id=input_receipt["receipt_id"],
        provider_input_receipt_sha256=_canonical_sha256(input_receipt),
    )
    binding_elapsed = _elapsed_seconds(stage_started)
    return (
        eeg,
        sampling_rate,
        input_receipt,
        binding,
        {
            "canonical_physical_root_read": canonical_elapsed,
            "provider_native_physical_carrier_read": provider_read_elapsed,
            "canonical_input_binding": binding_elapsed,
        },
    )


def _bind_detector_input_receipt(
    artifact: Mapping[str, Any],
    *,
    input_channel_receipt: Mapping[str, Any],
    canonical_detector_input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind observed/imputed detector channels without changing posteriors."""

    value = deepcopy(dict(artifact))
    canonical_binding = validate_canonical_detector_input_binding(
        dict(canonical_detector_input_binding)
    )
    if canonical_binding["provider_input_receipt_id"] != input_channel_receipt[
        "receipt_id"
    ] or canonical_binding["provider_input_receipt_sha256"] != _canonical_sha256(
        input_channel_receipt
    ):
        raise ValueError("canonical binding and provider input receipt disagree")
    preprocessing = deepcopy(dict(value["preprocessing_receipt"]))
    missing = list(input_channel_receipt["imputed_channel_ids"])
    preprocessing.update(
        {
            "missing_channel_imputation": bool(missing),
            "missing_channel_ids": missing,
            "observed_channel_ids": list(
                input_channel_receipt["observed_channel_ids"]
            ),
            "missing_channel_imputation_policy": (
                PUBLISHED_DEEPSOZ_MISSING_CHANNEL_POLICY
            ),
            "detector_input_channel_receipt_id": input_channel_receipt[
                "receipt_id"
            ],
            "detector_input_channel_receipt_sha256": _canonical_sha256(
                input_channel_receipt
            ),
            "canonical_detector_input_binding_id": canonical_binding[
                "binding_id"
            ],
            "canonical_detector_input_binding_receipt_sha256": (
                canonical_binding["receipt_sha256"]
            ),
            "imputed_channels_clinical_evidence_eligible": False,
        }
    )
    value["preprocessing_receipt"] = preprocessing
    value["posterior_artifact_id"] = "DEEPSOZ-POSTERIOR-PENDING"
    value["posterior_artifact_id"] = "DSZPOST-" + _canonical_sha256(value)[:24]
    return value


def _selected_manifest_rows(
    path: Path, *, split: str | None, recording_id: str | None, max_records: int
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"deepsoz_patient_id", "local_edf_path", "model_split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("DeepSOZ/TUSZ manifest lacks identity/split fields")
        selected: list[dict[str, str]] = []
        for row in reader:
            row_split = str(row["model_split"]).strip()
            row_recording = str(row["local_edf_path"]).strip()
            if split is not None and row_split != split:
                continue
            if recording_id is not None and row_recording != recording_id:
                continue
            # Project the label-bearing source manifest immediately to the
            # three split/identity fields needed for OOF model selection.  No
            # seizure-time, SOZ or clinical field is retained or passed to a
            # provider.
            selected.append(
                {
                    "deepsoz_patient_id": str(row["deepsoz_patient_id"]).strip(),
                    "local_edf_path": row_recording,
                    "model_split": row_split,
                }
            )
    selected.sort(key=lambda row: str(row["local_edf_path"]))
    if max_records > 0:
        selected = selected[:max_records]
    if not selected:
        raise ValueError("no DeepSOZ/TUSZ records matched the requested scope")
    ids = [str(row["local_edf_path"]).strip() for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("selected DeepSOZ/TUSZ records are not unique")
    return selected


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _artifact_filename(recording_id: str) -> str:
    return hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:24] + ".json"


def _source_signal_tensor_sha256(
    raw: np.ndarray, *, sampling_rate_hz: float
) -> str:
    """Reproduce the adapter's physical-input binding without model execution."""

    metadata = json.dumps(
        {
            "shape": list(raw.shape),
            "dtype": "little_endian_float64",
            "sampling_rate_hz": float(sampling_rate_hz),
            "channel_names": list(STANDARD_19),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical = np.ascontiguousarray(raw, dtype="<f8")
    digest = hashlib.sha256(metadata)
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _interval(start: float, stop: float) -> dict[str, object]:
    return {
        "start_offset_seconds": float(start),
        "stop_offset_seconds": float(stop),
        "closure": "left_closed_right_open",
    }


def _model_context_plan(full_seconds: int) -> list[dict[str, object]]:
    if isinstance(full_seconds, bool) or not isinstance(full_seconds, int):
        raise TypeError("DeepSOZ full-second count must be an integer")
    if full_seconds < 1:
        raise ValueError("DeepSOZ time support needs at least one modeled second")
    contexts: list[dict[str, object]] = []
    start = 0
    while True:
        stop = min(full_seconds, start + DEEPSOZ_CHUNK_SECONDS)
        contexts.append(
            {
                "model_context_id": f"DEEPSOZ-CONTEXT-{len(contexts) + 1:04d}",
                "support_interval": _interval(float(start), float(stop)),
                "modeled_seconds": stop - start,
                "bidirectional_temporal_context": True,
                "temporal_padding_used": False,
                "left_overlap_seconds": (
                    0
                    if start == 0
                    else min(DEEPSOZ_OVERLAP_SECONDS, stop - start)
                ),
                "right_overlap_seconds": (
                    0
                    if stop == full_seconds
                    else min(DEEPSOZ_OVERLAP_SECONDS, stop - start)
                ),
            }
        )
        if stop == full_seconds:
            return contexts
        start += DEEPSOZ_STRIDE_SECONDS


def _build_deepsoz_offline_time_support_receipt(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Project exact target/dependency/availability support for every row.

    DeepSOZ's target coordinate is one second, but its physical dependency is
    not one second: FFT resampling, zero-phase filters, clipping statistics and
    normalization are computed offline, and the temporal head is bidirectional
    within every contributing chunk.  Consequently no row is a causal or
    real-time detector decision, even when its target coordinate is near zero.
    """

    timeline = artifact.get("posterior_timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("DeepSOZ posterior timeline is empty")
    duration = float(artifact["recording_duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("DeepSOZ recording duration is invalid")
    usable_rows = [row for row in timeline if row.get("signal_usable") is True]
    if not usable_rows:
        raise ValueError("DeepSOZ posterior has no modeled full second")
    full_seconds = len(usable_rows)
    if any(
        float(row["start_offset_seconds"]) != float(index)
        or float(row["stop_offset_seconds"]) != float(index + 1)
        for index, row in enumerate(usable_rows)
    ):
        raise ValueError("DeepSOZ modeled rows are not the expected one-second grid")
    contexts = _model_context_plan(full_seconds)
    full_record_support = _interval(0.0, duration)
    support_rows: list[dict[str, Any]] = []
    partial_tail_count = 0
    for row in timeline:
        start = float(row["start_offset_seconds"])
        stop = float(row["stop_offset_seconds"])
        usable = row["signal_usable"] is True
        target = _interval(start, stop)
        if usable:
            contributing = [
                context
                for context in contexts
                if float(context["support_interval"]["start_offset_seconds"])
                <= start + 1e-9
                and float(context["support_interval"]["stop_offset_seconds"])
                >= stop - 1e-9
            ]
            if not contributing:
                raise RuntimeError("DeepSOZ posterior lacks a model context")
            context_ids = [
                str(context["model_context_id"]) for context in contributing
            ]
            model_future = max(
                float(context["support_interval"]["stop_offset_seconds"])
                - stop
                for context in contributing
            )
            effective_support: dict[str, object] | None = deepcopy(
                full_record_support
            )
            future_lookahead: float | None = max(0.0, duration - stop)
            value_semantics = "offline_model_probability_for_navigation_only"
            partial_tail = False
        else:
            partial_tail_count += 1
            context_ids = []
            model_future = None
            effective_support = None
            future_lookahead = None
            value_semantics = (
                "zero_sentinel_for_unmodeled_partial_tail_not_model_probability"
            )
            partial_tail = True
        support_rows.append(
            {
                "window_id": row["window_id"],
                "target_interval": target,
                "actual_preprocessing_support_interval": deepcopy(
                    full_record_support
                ),
                "actual_model_context_ids": context_ids,
                "effective_signal_dependency_support_interval": effective_support,
                "future_lookahead_from_target_stop_seconds": future_lookahead,
                "maximum_bidirectional_model_future_context_seconds": model_future,
                "decision_available_at_recording_offset_seconds": duration,
                "decision_availability_semantics": DEEPSOZ_DECISION_AVAILABILITY,
                "posterior_value_semantics": value_semantics,
                "partial_tail_coverage_marker": partial_tail,
                "temporal_padding_used": False,
            }
        )
    expected_tail = duration - full_seconds > 1e-9
    if partial_tail_count != int(expected_tail):
        raise ValueError("DeepSOZ partial-tail coverage semantics drifted")

    body: dict[str, Any] = {
        "schema_version": DEEPSOZ_POSTERIOR_TIME_SUPPORT_SCHEMA_VERSION,
        "receipt_id": "DEEPSOZ-TIME-SUPPORT-PENDING",
        "provider_id": DEEPSOZ_PROVIDER_ID,
        "recording_id": artifact["recording_id"],
        "recording_duration_seconds": duration,
        "target_clock": "recording_relative_physical_seconds",
        "timestamp_semantics": DEEPSOZ_TIMESTAMP_SEMANTICS,
        "offline_future_dependent": True,
        "causal_or_streaming_decision": False,
        "real_time_latency_metric_authorized": False,
        "preprocessing_dependency": {
            "support_interval": deepcopy(full_record_support),
            "whole_record_fft_resampling": True,
            "whole_record_zero_phase_filtering": True,
            "whole_record_per_channel_clipping_statistics": True,
            "whole_modeled_record_global_normalization": True,
            "finite_causal_receptive_field": False,
            "dependency_interpretation": (
                "conservative_full_record_physical_dependency_due_to_offline_global_and_zero_phase_operations"
            ),
        },
        "model_context_policy": {
            "chunk_seconds": DEEPSOZ_CHUNK_SECONDS,
            "overlap_seconds": DEEPSOZ_OVERLAP_SECONDS,
            "stride_seconds": DEEPSOZ_STRIDE_SECONDS,
            "bidirectional_chunk_context": True,
            "overlap_fusion": "linear_edge_ramp_weighted_probability_mean",
            "all_held_out_folds_share_context_plan": True,
        },
        "model_contexts": contexts,
        "padding_and_tail_semantics": {
            "temporal_padding_used": False,
            "silent_time_padding_used": False,
            "missing_channel_zero_fill_is_spatial_imputation_not_time_padding": True,
            "modeled_full_second_count": full_seconds,
            "partial_tail_present": expected_tail,
            "partial_tail_interval": (
                _interval(float(full_seconds), duration) if expected_tail else None
            ),
            "partial_tail_policy": DEEPSOZ_PARTIAL_TAIL_POLICY,
            "partial_tail_zero_is_negative_evidence": False,
        },
        "posterior_support_rows": support_rows,
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotations_used": False,
            "spreadsheet_used": False,
            "doctor_labels_used": False,
            "reference_event_times_used": False,
            "navigation_only": True,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_id"] = "DSZTIME-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _validate_deepsoz_offline_time_support_receipt(
    payload: object,
    *,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("DeepSOZ time-support receipt must be an object")
    expected = _build_deepsoz_offline_time_support_receipt(artifact)
    if payload != expected:
        raise ValueError("DeepSOZ posterior time-support receipt drifted")
    return deepcopy(payload)


def _build_runtime_receipt(
    *,
    recording_id: str,
    recording_duration_seconds: float,
    execution_mode: str,
    requested_device: str,
    stage_wall_seconds: Mapping[str, float],
    fold_wall_seconds: list[dict[str, Any]],
    total_compute_wall_seconds: float,
) -> dict[str, Any]:
    if execution_mode not in {"new_oof_inference", "resume_validation_only"}:
        raise ValueError("DeepSOZ runtime execution mode is invalid")
    duration = float(recording_duration_seconds)
    total = float(total_compute_wall_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("DeepSOZ runtime recording duration is invalid")
    if not math.isfinite(total) or total < 0:
        raise ValueError("DeepSOZ runtime duration is invalid")
    stages = {str(key): float(value) for key, value in stage_wall_seconds.items()}
    if not stages or any(
        not key or not math.isfinite(value) or value < 0
        for key, value in stages.items()
    ):
        raise ValueError("DeepSOZ runtime stage timings are invalid")
    fold_rows = deepcopy(fold_wall_seconds)
    body: dict[str, Any] = {
        "schema_version": DEEPSOZ_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "DEEPSOZ-RUNTIME-PENDING",
        "recording_id": recording_id,
        "execution_mode": execution_mode,
        "requested_device": str(requested_device),
        "monotonic_clock": "python_time_perf_counter_ns",
        "recording_duration_seconds": duration,
        "stage_wall_seconds": stages,
        "held_out_fold_wall_seconds": fold_rows,
        "total_compute_wall_seconds": total,
        "compute_real_time_factor": total / duration,
        "recording_seconds_per_compute_wall_second": (
            duration / total if total > 0 else None
        ),
        "runtime_semantics": (
            "offline_batch_compute_wall_time_not_signal_timestamp_and_not_real_time_decision_latency"
        ),
        "decision_availability_semantics": DEEPSOZ_DECISION_AVAILABILITY,
        "real_time_latency_metric_authorized": False,
        "onset_latency_claim_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_id"] = "DSZRUNTIME-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return _validate_runtime_receipt(body, recording_id=recording_id)


def _validate_runtime_receipt(
    payload: object,
    *,
    recording_id: str,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("DeepSOZ runtime receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_id",
        "recording_id",
        "execution_mode",
        "requested_device",
        "monotonic_clock",
        "recording_duration_seconds",
        "stage_wall_seconds",
        "held_out_fold_wall_seconds",
        "total_compute_wall_seconds",
        "compute_real_time_factor",
        "recording_seconds_per_compute_wall_second",
        "runtime_semantics",
        "decision_availability_semantics",
        "real_time_latency_metric_authorized",
        "onset_latency_claim_authorized",
        "receipt_sha256",
    }
    if set(data) != required:
        raise ValueError("DeepSOZ runtime receipt fields drifted")
    if data["schema_version"] != DEEPSOZ_RUNTIME_RECEIPT_SCHEMA_VERSION:
        raise ValueError("DeepSOZ runtime receipt schema drifted")
    if data["recording_id"] != recording_id:
        raise ValueError("DeepSOZ runtime recording binding drifted")
    if data["execution_mode"] not in {
        "new_oof_inference",
        "resume_validation_only",
    }:
        raise ValueError("DeepSOZ runtime execution mode drifted")
    if data["monotonic_clock"] != "python_time_perf_counter_ns":
        raise ValueError("DeepSOZ runtime clock drifted")
    if not isinstance(data["requested_device"], str) or not data[
        "requested_device"
    ]:
        raise ValueError("DeepSOZ runtime device is invalid")
    duration = float(data["recording_duration_seconds"])
    total = float(data["total_compute_wall_seconds"])
    if (
        not math.isfinite(duration)
        or duration <= 0
        or not math.isfinite(total)
        or total < 0
    ):
        raise ValueError("DeepSOZ runtime values are invalid")
    stages = data["stage_wall_seconds"]
    if not isinstance(stages, dict) or not stages or any(
        not isinstance(key, str)
        or not key
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for key, value in stages.items()
    ):
        raise ValueError("DeepSOZ runtime stages are invalid")
    folds = data["held_out_fold_wall_seconds"]
    if not isinstance(folds, list):
        raise ValueError("DeepSOZ runtime fold timings are invalid")
    observed_folds: list[int] = []
    for row in folds:
        if type(row) is not dict or set(row) != {
            "fold_index",
            "adapter_load_and_inference_wall_seconds",
        }:
            raise ValueError("DeepSOZ runtime fold row drifted")
        fold = row["fold_index"]
        elapsed = row["adapter_load_and_inference_wall_seconds"]
        if (
            isinstance(fold, bool)
            or not isinstance(fold, int)
            or not 0 <= fold < 15
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise ValueError("DeepSOZ runtime fold value is invalid")
        observed_folds.append(fold)
    if observed_folds != sorted(set(observed_folds)):
        raise ValueError("DeepSOZ runtime folds are duplicated or unordered")
    if data["execution_mode"] == "new_oof_inference" and not folds:
        raise ValueError("new DeepSOZ runtime receipt lacks fold timings")
    if data["execution_mode"] == "resume_validation_only" and folds:
        raise ValueError("resume runtime receipt cannot claim fold inference")
    expected_rtf = total / duration
    if abs(float(data["compute_real_time_factor"]) - expected_rtf) > 1e-12:
        raise ValueError("DeepSOZ runtime real-time factor drifted")
    expected_throughput = duration / total if total > 0 else None
    if data["recording_seconds_per_compute_wall_second"] != expected_throughput:
        raise ValueError("DeepSOZ runtime throughput drifted")
    if data["runtime_semantics"] != (
        "offline_batch_compute_wall_time_not_signal_timestamp_and_not_real_time_decision_latency"
    ):
        raise ValueError("DeepSOZ runtime semantics drifted")
    if (
        data["decision_availability_semantics"] != DEEPSOZ_DECISION_AVAILABILITY
        or data["real_time_latency_metric_authorized"] is not False
        or data["onset_latency_claim_authorized"] is not False
    ):
        raise ValueError("DeepSOZ runtime improperly claims causal latency")
    if not _is_sha256(data["receipt_sha256"]):
        raise ValueError("DeepSOZ runtime hash is invalid")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("DeepSOZ runtime receipt is not content-bound")
    id_source = deepcopy(data)
    id_source["receipt_id"] = "DEEPSOZ-RUNTIME-PENDING"
    id_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_id"] != "DSZRUNTIME-" + _canonical_sha256(id_source)[:24]:
        raise ValueError("DeepSOZ runtime receipt ID is not content-bound")
    return data


def _bind_oof_materialization_receipts(
    artifact: Mapping[str, Any],
    *,
    canonical_detector_input_binding: Mapping[str, Any],
    time_support_receipt: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = deepcopy(dict(artifact))
    canonical_binding = validate_canonical_detector_input_binding(
        dict(canonical_detector_input_binding)
    )
    time_support = _validate_deepsoz_offline_time_support_receipt(
        dict(time_support_receipt), artifact=value
    )
    runtime = _validate_runtime_receipt(
        dict(runtime_receipt), recording_id=str(value["recording_id"])
    )
    preprocessing = value.get("preprocessing_receipt")
    if not isinstance(preprocessing, Mapping) or preprocessing.get(
        "canonical_detector_input_binding_id"
    ) != canonical_binding["binding_id"] or preprocessing.get(
        "canonical_detector_input_binding_receipt_sha256"
    ) != canonical_binding["receipt_sha256"]:
        raise ValueError("OOF preprocessing does not bind the canonical carrier")
    value.update(
        {
            "materialization_schema_version": (
                DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION
            ),
            "canonical_detector_input_binding": canonical_binding,
            "posterior_time_support_receipt": time_support,
            "posterior_runtime_receipt": runtime,
        }
    )
    value["posterior_artifact_id"] = "DEEPSOZ-OOF-POSTERIOR-PENDING"
    value["posterior_artifact_id"] = "DSZOOF-" + _canonical_sha256(value)[:24]
    return value


def _validate_resumable_oof_artifact(
    value: object,
    *,
    recording_id: str,
    patient_id: str,
    expected_fold_indices: tuple[int, ...],
    fold_assignment_receipt: Mapping[str, Any],
    source_signal_tensor_sha256: str,
    input_channel_receipt: Mapping[str, Any],
    canonical_detector_input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed before reusing a posterior left by an interrupted batch."""

    if type(value) is not dict:
        raise TypeError("resumed DeepSOZ posterior artifact must be an object")
    artifact = deepcopy(value)
    required = {
        "schema_version",
        "materialization_schema_version",
        "posterior_artifact_id",
        "provider_id",
        "recording_id",
        "deepsoz_patient_id",
        "held_out_fold_indices",
        "held_out_repeat_count",
        "fold_assignment_receipt_sha256",
        "patient_fold_binding_sha256",
        "weights_manifest_sha256",
        "adapter_code_sha256",
        "fold_posterior_artifact_ids",
        "source_signal_tensor_sha256",
        "recording_duration_seconds",
        "preprocessing_receipt",
        "canonical_detector_input_binding",
        "posterior_time_support_receipt",
        "posterior_runtime_receipt",
        "fold_fusion",
        "posterior_timeline",
        "scope_receipt",
    }
    if set(artifact) != required:
        raise ValueError("resumed DeepSOZ posterior artifact schema drifted")
    if artifact["schema_version"] != DEEPSOZ_OOF_ENSEMBLE_SCHEMA_VERSION:
        raise ValueError("resumed DeepSOZ posterior schema version drifted")
    if artifact["materialization_schema_version"] != (
        DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION
    ):
        raise ValueError("resumed DeepSOZ materialization schema drifted")
    if artifact["provider_id"] != DEEPSOZ_PROVIDER_ID:
        raise ValueError("resumed DeepSOZ posterior provider drifted")
    if artifact["recording_id"] != recording_id:
        raise ValueError("resumed DeepSOZ posterior recording binding drifted")
    normalized_patient = str(int(str(patient_id).strip()))
    if artifact["deepsoz_patient_id"] != normalized_patient:
        raise ValueError("resumed DeepSOZ posterior patient binding drifted")
    expected_folds = tuple(sorted(expected_fold_indices))
    if artifact["held_out_fold_indices"] != list(expected_folds):
        raise ValueError("resumed DeepSOZ posterior held-out folds drifted")
    if artifact["held_out_repeat_count"] != len(expected_folds):
        raise ValueError("resumed DeepSOZ posterior held-out repeat count drifted")
    assignment_sha256 = _canonical_sha256(fold_assignment_receipt)
    if artifact["fold_assignment_receipt_sha256"] != assignment_sha256:
        raise ValueError("resumed DeepSOZ posterior fold receipt drifted")
    expected_patient_binding = _canonical_sha256(
        [normalized_patient, list(expected_folds), fold_assignment_receipt["receipt_id"]]
    )
    if artifact["patient_fold_binding_sha256"] != expected_patient_binding:
        raise ValueError("resumed DeepSOZ posterior fold-patient binding drifted")
    if artifact["weights_manifest_sha256"] != PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256:
        raise ValueError("resumed DeepSOZ posterior weight manifest drifted")
    adapter_sha256 = hashlib.sha256(
        (ROOT / "src/clinical_eeg_long_recording/deepsoz_temporal_adapter.py").read_bytes()
    ).hexdigest()
    if artifact["adapter_code_sha256"] != adapter_sha256:
        raise ValueError("resumed DeepSOZ posterior adapter code drifted")
    fold_artifact_ids = artifact["fold_posterior_artifact_ids"]
    if (
        not isinstance(fold_artifact_ids, list)
        or len(fold_artifact_ids) != len(expected_folds)
        or any(
            not isinstance(identifier, str) or not identifier.startswith("DSZPOST-")
            for identifier in fold_artifact_ids
        )
    ):
        raise ValueError("resumed DeepSOZ posterior fold artifacts drifted")
    if artifact["source_signal_tensor_sha256"] != source_signal_tensor_sha256:
        raise ValueError("resumed DeepSOZ posterior physical signal binding drifted")
    preprocessing = artifact["preprocessing_receipt"]
    if not isinstance(preprocessing, Mapping):
        raise TypeError("resumed DeepSOZ preprocessing receipt must be an object")
    missing = list(input_channel_receipt["imputed_channel_ids"])
    if preprocessing.get("missing_channel_imputation") is not bool(missing):
        raise ValueError("resumed DeepSOZ missing-channel state drifted")
    expected_input_sha256 = _canonical_sha256(input_channel_receipt)
    if preprocessing.get("missing_channel_ids") != missing:
        raise ValueError("resumed DeepSOZ imputed-channel set drifted")
    if preprocessing.get("observed_channel_ids") != list(
        input_channel_receipt["observed_channel_ids"]
    ):
        raise ValueError("resumed DeepSOZ observed-channel set drifted")
    if preprocessing.get("missing_channel_imputation_policy") != (
        PUBLISHED_DEEPSOZ_MISSING_CHANNEL_POLICY
    ):
        raise ValueError("resumed DeepSOZ missing-channel policy drifted")
    if preprocessing.get("detector_input_channel_receipt_id") != (
        input_channel_receipt["receipt_id"]
    ) or preprocessing.get("detector_input_channel_receipt_sha256") != (
        expected_input_sha256
    ):
        raise ValueError("resumed DeepSOZ detector-input binding drifted")
    if preprocessing.get("imputed_channels_clinical_evidence_eligible") is not False:
        raise ValueError("resumed DeepSOZ imputed channel became evidence eligible")
    expected_canonical_binding = validate_canonical_detector_input_binding(
        dict(canonical_detector_input_binding)
    )
    if artifact["canonical_detector_input_binding"] != expected_canonical_binding:
        raise ValueError("resumed DeepSOZ canonical physical binding drifted")
    if preprocessing.get("canonical_detector_input_binding_id") != (
        expected_canonical_binding["binding_id"]
    ) or preprocessing.get(
        "canonical_detector_input_binding_receipt_sha256"
    ) != expected_canonical_binding["receipt_sha256"]:
        raise ValueError("resumed DeepSOZ preprocessing canonical binding drifted")
    duration = artifact["recording_duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise TypeError("resumed DeepSOZ posterior duration is not numeric")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("resumed DeepSOZ posterior duration is invalid")
    timeline = artifact["posterior_timeline"]
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("resumed DeepSOZ posterior timeline is empty")
    cursor = 0.0
    for index, row in enumerate(timeline):
        if type(row) is not dict or set(row) != {
            "window_id",
            "start_offset_seconds",
            "stop_offset_seconds",
            "seizure_probability",
            "signal_usable",
        }:
            raise ValueError(f"resumed DeepSOZ posterior row {index} schema drifted")
        start = float(row["start_offset_seconds"])
        stop = float(row["stop_offset_seconds"])
        probability = float(row["seizure_probability"])
        usable = row["signal_usable"]
        if (
            not math.isfinite(start)
            or not math.isfinite(stop)
            or abs(start - cursor) > 1e-9
            or stop <= start
            or stop - start > 1.0 + 1e-9
        ):
            raise ValueError("resumed DeepSOZ posterior time grid drifted")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("resumed DeepSOZ posterior probability is invalid")
        if type(usable) is not bool or (not usable and probability != 0.0):
            raise ValueError("resumed DeepSOZ posterior usability semantics drifted")
        cursor = stop
    if abs(cursor - duration) > 1e-9:
        raise ValueError("resumed DeepSOZ posterior does not cover the recording")
    _validate_deepsoz_offline_time_support_receipt(
        artifact["posterior_time_support_receipt"], artifact=artifact
    )
    runtime = _validate_runtime_receipt(
        artifact["posterior_runtime_receipt"], recording_id=recording_id
    )
    if runtime["execution_mode"] != "new_oof_inference":
        raise ValueError("resumed artifact lacks its original inference runtime")
    if runtime["recording_duration_seconds"] != duration:
        raise ValueError("resumed DeepSOZ runtime duration drifted")
    runtime_folds = [
        row["fold_index"] for row in runtime["held_out_fold_wall_seconds"]
    ]
    if runtime_folds != list(expected_folds):
        raise ValueError("resumed DeepSOZ runtime fold set drifted")
    scope = artifact["scope_receipt"]
    required_scope = {
        "eeg_signal_only": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "clinical_context_used": False,
        "reference_seizure_times_used_for_inference": False,
        "fold_assignment_uses_patient_split_metadata_only": True,
        "research_only": True,
        "posterior_is_confirmed_seizure_or_onset": False,
        "sota_claim_authorized": False,
    }
    if scope != required_scope:
        raise ValueError("resumed DeepSOZ posterior inference scope drifted")
    content = deepcopy(artifact)
    content["posterior_artifact_id"] = "DEEPSOZ-OOF-POSTERIOR-PENDING"
    expected_artifact_id = "DSZOOF-" + _canonical_sha256(content)[:24]
    if artifact["posterior_artifact_id"] != expected_artifact_id:
        raise ValueError("resumed DeepSOZ posterior content binding failed")
    return artifact


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    batch_started_ns = time.perf_counter_ns()
    if args.max_records < 0:
        raise ValueError("max-records must be non-negative")
    tusz_root = args.tusz_root.resolve(strict=True)
    assignment, assignment_receipt = load_published_deepsoz_oof_fold_assignment(
        args.fold_directory
    )
    assignment_sha256 = _canonical_sha256(assignment_receipt)
    rows = _selected_manifest_rows(
        args.manifest,
        split=args.split,
        recording_id=args.recording_id,
        max_records=args.max_records,
    )
    adapters: dict[int, DeepSOZTemporalResearchAdapter] = {}

    def adapter(fold: int) -> DeepSOZTemporalResearchAdapter:
        if fold not in adapters:
            adapters[fold] = DeepSOZTemporalResearchAdapter(
                checkpoint_path=args.weights_directory / f"fold{fold}.pth.tar",
                expected_checkpoint_sha256=PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256[fold],
                weights_manifest_sha256=PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
                fold_index=fold,
                inference_mode="tusz_patient_oof",
                fold_assignment_receipt_sha256=assignment_sha256,
                device=args.device,
            )
        return adapters[fold]

    index_rows: list[dict[str, Any]] = []
    resumed_artifact_count = 0
    newly_materialized_artifact_count = 0
    for ordinal, row in enumerate(rows, start=1):
        record_started_ns = time.perf_counter_ns()
        patient_id = str(int(str(row["deepsoz_patient_id"]).strip()))
        if patient_id not in assignment:
            raise ValueError(f"patient {patient_id} has no official held-out fold")
        recording_id = str(row["local_edf_path"]).strip()
        filename = _artifact_filename(recording_id)
        relative = Path("posteriors") / filename
        artifact_path = args.output_directory / relative
        (
            eeg,
            sampling_rate,
            input_channel_receipt,
            canonical_detector_input_binding,
            stage_wall_seconds,
        ) = _load_canonical_bound_standard19(
            _safe_edf(tusz_root, recording_id)
        )
        source_tensor_sha256 = _source_signal_tensor_sha256(
            eeg, sampling_rate_hz=sampling_rate
        )
        if args.resume and (
            artifact_path.is_symlink()
            or (artifact_path.exists() and not artifact_path.is_file())
        ):
            raise ValueError(
                f"resumed DeepSOZ posterior path is not a regular file: {artifact_path}"
            )
        if args.resume and artifact_path.is_file():
            stage_started_ns = time.perf_counter_ns()
            text = artifact_path.read_text(encoding="utf-8")
            artifact = _validate_resumable_oof_artifact(
                json.loads(text),
                recording_id=recording_id,
                patient_id=patient_id,
                expected_fold_indices=assignment[patient_id],
                fold_assignment_receipt=assignment_receipt,
                source_signal_tensor_sha256=source_tensor_sha256,
                input_channel_receipt=input_channel_receipt,
                canonical_detector_input_binding=(
                    canonical_detector_input_binding
                ),
            )
            stage_wall_seconds["resume_artifact_read_and_validation"] = (
                _elapsed_seconds(stage_started_ns)
            )
            current_run_runtime = _build_runtime_receipt(
                recording_id=recording_id,
                recording_duration_seconds=artifact[
                    "recording_duration_seconds"
                ],
                execution_mode="resume_validation_only",
                requested_device=args.device,
                stage_wall_seconds=stage_wall_seconds,
                fold_wall_seconds=[],
                total_compute_wall_seconds=_elapsed_seconds(record_started_ns),
            )
            resumed_artifact_count += 1
        else:
            fold_artifacts: list[dict[str, Any]] = []
            fold_wall_seconds: list[dict[str, Any]] = []
            for fold in sorted(assignment[patient_id]):
                stage_started_ns = time.perf_counter_ns()
                fold_artifact = _bind_detector_input_receipt(
                    adapter(fold).materialize_dense_posterior(
                        recording_id=recording_id,
                        standardized_eeg=eeg,
                        sampling_rate_hz=sampling_rate,
                        channel_names=STANDARD_19,
                    ),
                    input_channel_receipt=input_channel_receipt,
                    canonical_detector_input_binding=(
                        canonical_detector_input_binding
                    ),
                )
                fold_artifacts.append(fold_artifact)
                fold_wall_seconds.append(
                    {
                        "fold_index": fold,
                        "adapter_load_and_inference_wall_seconds": (
                            _elapsed_seconds(stage_started_ns)
                        ),
                    }
                )
            stage_wall_seconds["held_out_fold_inference"] = sum(
                float(value["adapter_load_and_inference_wall_seconds"])
                for value in fold_wall_seconds
            )
            stage_started_ns = time.perf_counter_ns()
            base_artifact = aggregate_deepsoz_oof_fold_posteriors(
                patient_id=patient_id,
                expected_fold_indices=assignment[patient_id],
                fold_assignment_receipt=assignment_receipt,
                fold_artifacts=fold_artifacts,
            )
            stage_wall_seconds["oof_fold_aggregation"] = _elapsed_seconds(
                stage_started_ns
            )
            stage_started_ns = time.perf_counter_ns()
            time_support_receipt = _build_deepsoz_offline_time_support_receipt(
                base_artifact
            )
            stage_wall_seconds["posterior_time_support_projection"] = (
                _elapsed_seconds(stage_started_ns)
            )
            current_run_runtime = _build_runtime_receipt(
                recording_id=recording_id,
                recording_duration_seconds=base_artifact[
                    "recording_duration_seconds"
                ],
                execution_mode="new_oof_inference",
                requested_device=args.device,
                stage_wall_seconds=stage_wall_seconds,
                fold_wall_seconds=fold_wall_seconds,
                total_compute_wall_seconds=_elapsed_seconds(record_started_ns),
            )
            artifact = _bind_oof_materialization_receipts(
                base_artifact,
                canonical_detector_input_binding=(
                    canonical_detector_input_binding
                ),
                time_support_receipt=time_support_receipt,
                runtime_receipt=current_run_runtime,
            )
            text = json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n"
            _atomic_text(artifact_path, text)
            newly_materialized_artifact_count += 1
        index_rows.append(
            {
                "ordinal": ordinal,
                "recording_id": recording_id,
                "deepsoz_patient_id": patient_id,
                "model_split": str(row["model_split"]).strip(),
                "held_out_fold_indices": list(assignment[patient_id]),
                "posterior_artifact_id": artifact["posterior_artifact_id"],
                "adapter_code_sha256": artifact["adapter_code_sha256"],
                "posterior_relative_path": relative.as_posix(),
                "posterior_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "recording_duration_seconds": artifact["recording_duration_seconds"],
                "timeline_window_count": len(artifact["posterior_timeline"]),
                "canonical_signal_id": canonical_detector_input_binding[
                    "canonical_signal_id"
                ],
                "canonical_detector_input_binding_id": (
                    canonical_detector_input_binding["binding_id"]
                ),
                "canonical_detector_input_binding_receipt_sha256": (
                    canonical_detector_input_binding["receipt_sha256"]
                ),
                "detector_input_channel_receipt": input_channel_receipt,
                "detector_input_channel_receipt_sha256": _canonical_sha256(
                    input_channel_receipt
                ),
                "detector_imputed_channel_count": input_channel_receipt[
                    "imputed_channel_count"
                ],
                "posterior_time_support_receipt_id": artifact[
                    "posterior_time_support_receipt"
                ]["receipt_id"],
                "posterior_time_support_receipt_sha256": artifact[
                    "posterior_time_support_receipt"
                ]["receipt_sha256"],
                "posterior_runtime_receipt_id": artifact[
                    "posterior_runtime_receipt"
                ]["receipt_id"],
                "posterior_runtime_receipt_sha256": artifact[
                    "posterior_runtime_receipt"
                ]["receipt_sha256"],
                "current_run_runtime_receipt": current_run_runtime,
                "current_run_runtime_receipt_sha256": current_run_runtime[
                    "receipt_sha256"
                ],
                "offline_future_dependent": True,
                "posterior_timestamp_is_real_time_latency": False,
                "decision_available_at_recording_end": True,
                "partial_tail_present": artifact[
                    "posterior_time_support_receipt"
                ]["padding_and_tail_semantics"]["partial_tail_present"],
                "partial_tail_policy": DEEPSOZ_PARTIAL_TAIL_POLICY,
            }
        )

    index_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in index_rows
    )
    _atomic_text(args.output_directory / "posterior_index.jsonl", index_text)
    adapter_code_hashes = {row["adapter_code_sha256"] for row in index_rows}
    if len(adapter_code_hashes) != 1:
        raise RuntimeError("DeepSOZ adapter code changed within one batch")
    batch_elapsed_seconds = _elapsed_seconds(batch_started_ns)
    total_recording_seconds = sum(
        float(row["recording_duration_seconds"]) for row in index_rows
    )
    batch_runtime_receipt: dict[str, Any] = {
        "schema_version": DEEPSOZ_BATCH_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "DEEPSOZ-BATCH-RUNTIME-PENDING",
        "monotonic_clock": "python_time_perf_counter_ns",
        "batch_compute_wall_seconds": batch_elapsed_seconds,
        "total_recording_duration_seconds": total_recording_seconds,
        "batch_compute_real_time_factor": (
            batch_elapsed_seconds / total_recording_seconds
        ),
        "recording_seconds_per_batch_compute_wall_second": (
            total_recording_seconds / batch_elapsed_seconds
            if batch_elapsed_seconds > 0
            else None
        ),
        "runtime_semantics": (
            "offline_batch_materialization_or_resume_validation_wall_time_not_real_time_decision_latency"
        ),
        "real_time_latency_metric_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    batch_runtime_receipt["receipt_id"] = "DSZBATCHRUNTIME-" + _canonical_sha256(
        batch_runtime_receipt
    )[:24]
    batch_runtime_receipt["receipt_sha256"] = _canonical_sha256(
        batch_runtime_receipt
    )
    receipt: dict[str, Any] = {
        "schema_version": "deepsoz_tusz_continuous_posterior_batch_v2",
        "receipt_id": "DEEPSOZ-BATCH-PENDING",
        "provider_id": "deepsoz_temporal_oof_candidate_v1",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "fold_assignment_receipt": assignment_receipt,
        "weights_manifest_sha256": PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
        "adapter_code_sha256": next(iter(adapter_code_hashes)),
        "materialized_oof_schema_version": (
            DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION
        ),
        "posterior_time_support_schema_version": (
            DEEPSOZ_POSTERIOR_TIME_SUPPORT_SCHEMA_VERSION
        ),
        "selected_split": args.split,
        "selected_recording_id": args.recording_id,
        "max_records": args.max_records,
        "inventory_scope": (
            "full_selected_split" if args.max_records == 0 and args.recording_id is None
            else "explicit_research_smoke_subset"
        ),
        "recording_count": len(index_rows),
        "resume_requested": bool(args.resume),
        "resumed_artifact_count": resumed_artifact_count,
        "newly_materialized_artifact_count": newly_materialized_artifact_count,
        "index_sha256": hashlib.sha256(index_text.encode("utf-8")).hexdigest(),
        "batch_runtime_receipt": batch_runtime_receipt,
        "all_selected_records_materialized": True,
        "records_with_detector_channel_imputation": sum(
            int(row["detector_imputed_channel_count"] > 0) for row in index_rows
        ),
        "total_detector_imputed_channels": sum(
            int(row["detector_imputed_channel_count"]) for row in index_rows
        ),
        "published_missing_channel_policy": (
            PUBLISHED_DEEPSOZ_MISSING_CHANNEL_POLICY
        ),
        "published_missing_channel_policy_source_sha256": (
            PUBLISHED_DEEPSOZ_UTILS_PREPROCESS_SHA256
        ),
        "detector_imputed_channels_clinical_evidence_eligible": False,
        "canonical_physical_input_bindings_verified": len(index_rows),
        "all_posteriors_have_explicit_physical_time_support": True,
        "all_posteriors_offline_future_dependent": True,
        "posterior_timestamp_semantics": DEEPSOZ_TIMESTAMP_SEMANTICS,
        "decision_availability_semantics": DEEPSOZ_DECISION_AVAILABILITY,
        "real_time_latency_metric_authorized": False,
        "partial_tail_policy": DEEPSOZ_PARTIAL_TAIL_POLICY,
        "silent_time_padding_used": False,
        "edf_annotations_used": False,
        "label_bearing_manifest_fields_retained_for_inference": False,
        "seizure_or_soz_labels_used_for_inference": False,
        "posterior_only_operating_point_not_applied": True,
        "production_qualified": False,
        "sota_claim_authorized": False,
    }
    receipt["receipt_id"] = "DSZBATCH-" + _canonical_sha256(receipt)[:24]
    _atomic_text(
        args.output_directory / "batch_receipt.json",
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize patient-OOF DeepSOZ dense posteriors for TUSZ EDFs"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--weights-directory", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--fold-directory", type=Path, default=Path("/tmp"))
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="source_dev",
        help=(
            "manifest split to materialize; defaults to source_dev so operating "
            "point development never opens source_eval implicitly"
        ),
    )
    parser.add_argument("--recording-id")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse only content-bound artifacts whose signal, patient, fold, "
            "weight, adapter, scope and complete timeline bindings validate"
        ),
    )
    return parser.parse_args()


def main() -> None:
    receipt = materialize(_parse_args())
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
