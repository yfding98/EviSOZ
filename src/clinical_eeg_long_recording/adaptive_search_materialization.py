"""EEG-only materialization of adaptive transition-search envelopes.

This module deliberately uses only physical EDF signal methods.  The loader
never calls ``readAnnotations`` and the public API has no annotation, Excel,
clinical-context, label, or ground-truth argument.  A coarse detector anchor is
used only to plan an envelope; the signal search itself may refine or reject it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import torch

from src.soz.geometry import STANDARD_19, TCP_20_EDGES

from .canonical_adaptive_binding import (
    build_canonical_adaptive_signal_binding,
    validate_canonical_adaptive_signal_binding,
)
from .canonical_edf_materialization import (
    CanonicalEDFConfig,
    CanonicalEDFViewBundle,
    load_canonical_edf_views,
    validate_canonical_edf_materialization,
)
from .canonical_signal_views import validate_transform_spec
from .montage_reference_observability import (
    build_reference_matrix_observability,
    require_reference_materialization_authorized,
    validate_reference_matrix_observability,
)

from .adaptive_search import (
    analyze_adaptive_eeg_envelope,
    generalized_signal_tensor_sha256,
    plan_adaptive_search_envelopes,
    validate_adaptive_search_envelope_plan,
    validate_adaptive_search_receipt,
)
from .schema import (
    canonical_payload_sha256,
    validate_long_term_seizure_detection_manifest,
)


ADAPTIVE_PREPROCESSING_SCHEMA_VERSION = "adaptive_eeg_envelope_preprocessing_v4"
ADAPTIVE_WHOLE_RECORD_TRANSFORM_SCHEMA_VERSION = (
    "adaptive_eeg_whole_record_navigation_transform_v2"
)
ADAPTIVE_MATERIALIZATION_SCHEMA_VERSION = "adaptive_eeg_search_materialization_v4"
ADAPTIVE_PREPROCESSING_METHOD = (
    "canonical_context_offline_whole_record_then_montage_qualified_observed_car_crop_v4"
)
ADAPTIVE_WHOLE_RECORD_TRANSFORM_METHOD = (
    "canonical_context_offline_whole_record_montage_qualified_observed_car_navigation_v2"
)
_ADAPTIVE_PHASE_POLICY = "offline_zero_phase_whole_record_then_crop"
_ADAPTIVE_CROP_POLICY = "global_output_sample_edges_no_recompute_v1"

@dataclass(frozen=True)
class LoadedAdaptiveEnvelope:
    signal: torch.Tensor
    preprocessing_receipt: dict[str, Any]


@dataclass(frozen=True)
class _WholeRecordingAdaptiveNavigationView:
    """One immutable navigation tensor shared by all event crops."""

    signal: torch.Tensor
    transform_receipt: dict[str, Any]


AdaptiveEnvelopeLoader = Callable[..., LoadedAdaptiveEnvelope]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_edf(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("adaptive-search EDF must not be a symbolic link")
    resolved = source.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file() or resolved.suffix.lower() != ".edf":
        raise ValueError("adaptive-search source must be a regular EDF file")
    return resolved


def _require_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} is not SHA-256")
    return value


def _aligned_sample_edge(seconds: object, rate_hz: float, *, context: str) -> int:
    value = float(seconds)
    position = value * rate_hz
    sample = int(round(position))
    if not math.isfinite(value) or abs(position - sample) > 1e-8:
        raise ValueError(f"{context} is not aligned to its global sample clock")
    return sample


def _validate_sample_intervals(
    payload: object,
    *,
    context: str,
    upper: int,
) -> list[list[int]]:
    if not isinstance(payload, list):
        raise ValueError(f"{context} must be a list")
    result: list[list[int]] = []
    previous_stop = 0
    for index, raw in enumerate(payload):
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or isinstance(raw[0], bool)
            or isinstance(raw[1], bool)
            or not isinstance(raw[0], int)
            or not isinstance(raw[1], int)
            or raw[0] < previous_stop
            or raw[1] <= raw[0]
            or raw[1] > upper
        ):
            raise ValueError(f"{context}[{index}] is invalid")
        result.append([raw[0], raw[1]])
        previous_stop = raw[1]
    return result


def validate_adaptive_whole_record_transform_receipt(
    payload: object,
) -> dict[str, Any]:
    """Validate the immutable whole-record navigation transform.

    The parent offline view is transformed exactly once on the recording-global
    clock.  Event expansion is therefore a pure crop operation and cannot
    create a second zero-phase boundary condition for the same physical sample.
    This receipt is navigation-only and is never onset-authorized.
    """

    if type(payload) is not dict:
        raise TypeError("adaptive whole-record transform receipt must be an object")
    required = {
        "schema_version",
        "method",
        "role",
        "source_signal_sha256",
        "source_identity_semantics",
        "canonical_signal_binding",
        "canonical_materialization_receipt_sha256",
        "montage_reference_observability_receipt_sha256",
        "acquisition_montage_class",
        "observed_car_reference_matrix_observability",
        "parent_context_offline_view",
        "semantic_channels",
        "raw_channel_names",
        "raw_units",
        "source_sampling_rate_hz",
        "output_sampling_rate_hz",
        "whole_record_interval_recording_seconds",
        "whole_record_source_sample_interval",
        "whole_record_output_sample_interval",
        "whole_record_output_sample_count",
        "highpass_hz",
        "lowpass_hz",
        "butterworth_order",
        "phase_policy",
        "resample_up",
        "resample_down",
        "reference_policy",
        "eligible_bipolar_derivations",
        "global_edge_invalid_output_sample_intervals",
        "whole_record_processed_navigation_sha256",
        "crop_policy",
        "scope_receipt",
        "receipt_sha256",
    }
    if set(payload) != required:
        raise ValueError(
            "adaptive whole-record transform receipt has missing or unknown fields"
        )
    data = deepcopy(payload)
    if data["schema_version"] != ADAPTIVE_WHOLE_RECORD_TRANSFORM_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive whole-record transform schema")
    if data["method"] != ADAPTIVE_WHOLE_RECORD_TRANSFORM_METHOD:
        raise ValueError("adaptive whole-record transform method drifted")
    if data["role"] != "adaptive_boundary_navigation_only":
        raise ValueError("adaptive whole-record transform role drifted")
    if data["source_identity_semantics"] != (
        "detector_manifest_edf_container_sha256"
    ):
        raise ValueError("adaptive whole-record source hash semantics drifted")
    for key in (
        "source_signal_sha256",
        "canonical_materialization_receipt_sha256",
        "montage_reference_observability_receipt_sha256",
        "whole_record_processed_navigation_sha256",
        "receipt_sha256",
    ):
        _require_sha256(data[key], context=f"adaptive whole-record {key}")

    binding = validate_canonical_adaptive_signal_binding(
        data["canonical_signal_binding"]
    )
    data["canonical_signal_binding"] = binding
    if data["acquisition_montage_class"] != "common_compatible_referential":
        raise ValueError("adaptive CAR acquisition montage is not compatible")
    if data["semantic_channels"] != list(STANDARD_19):
        raise ValueError("adaptive whole-record channel order drifted")
    observed = list(binding["observed_channel_ids"])
    if len(observed) < 2:
        raise ValueError("adaptive observed-channel CAR requires at least two electrodes")
    car_matrix = validate_reference_matrix_observability(
        data["observed_car_reference_matrix_observability"]
    )
    scale = 1.0 / len(observed)
    expected_car_matrix = []
    for target_index in range(len(observed)):
        row = [-scale] * len(observed)
        row[target_index] += 1.0
        expected_car_matrix.append(row)
    if (
        car_matrix["row_unit_ids"] != [f"{channel}-CAR-NAV" for channel in observed]
        or car_matrix["column_unit_ids"] != observed
        or car_matrix["matrix"] != expected_car_matrix
    ):
        raise ValueError("adaptive observed-channel CAR matrix drifted")
    data["observed_car_reference_matrix_observability"] = car_matrix
    if (
        not isinstance(data["raw_channel_names"], list)
        or len(data["raw_channel_names"]) != len(observed)
        or not all(isinstance(item, str) and item for item in data["raw_channel_names"])
        or not isinstance(data["raw_units"], list)
        or len(data["raw_units"]) != len(observed)
        or not all(isinstance(item, str) and item for item in data["raw_units"])
    ):
        raise ValueError("adaptive whole-record raw channel ledger is invalid")

    parent_raw = data["parent_context_offline_view"]
    parent_required = {
        "view_id",
        "receipt_sha256",
        "processed_view_sha256",
        "transform_spec",
        "selected_global_output_sample_interval",
        "selected_recording_seconds",
        "edge_invalid_output_sample_intervals",
    }
    if type(parent_raw) is not dict or set(parent_raw) != parent_required:
        raise ValueError("adaptive whole-record parent view binding is invalid")
    parent = deepcopy(parent_raw)
    if not isinstance(parent["view_id"], str) or not parent["view_id"]:
        raise ValueError("adaptive whole-record parent view ID is invalid")
    for key in ("receipt_sha256", "processed_view_sha256"):
        _require_sha256(parent[key], context=f"adaptive parent view {key}")
    transform = validate_transform_spec(parent["transform_spec"])
    parent["transform_spec"] = transform
    data["parent_context_offline_view"] = parent
    if (
        transform["input_unit_ids"] != list(STANDARD_19)
        or transform["output_unit_ids"] != list(STANDARD_19)
        or transform["filter"]["family"] != "butterworth"
        or transform["filter"]["phase_policy"] != "offline_zero_phase"
        or transform["reference"]["reference_type"]
        != "source_header_reference_preserved_per_channel_v1"
        or transform["output_clock"]["global_origin_recording_seconds"] != 0.0
    ):
        raise ValueError("adaptive parent is not the canonical whole-record offline view")

    source_sfreq = float(data["source_sampling_rate_hz"])
    output_sfreq = float(data["output_sampling_rate_hz"])
    transform_source_sfreq = (
        int(transform["source_clock"]["sampling_rate_numerator"])
        / int(transform["source_clock"]["sampling_rate_denominator"])
    )
    transform_output_sfreq = (
        int(transform["output_clock"]["sampling_rate_numerator"])
        / int(transform["output_clock"]["sampling_rate_denominator"])
    )
    if (
        not math.isfinite(source_sfreq)
        or not math.isfinite(output_sfreq)
        or min(source_sfreq, output_sfreq) <= 0
        or abs(source_sfreq - transform_source_sfreq) > 1e-9
        or abs(output_sfreq - transform_output_sfreq) > 1e-9
    ):
        raise ValueError("adaptive whole-record sampling rates drifted")
    if (
        float(data["highpass_hz"]) != float(transform["filter"]["highpass_hz"])
        or float(data["lowpass_hz"]) != float(transform["filter"]["lowpass_hz"])
        or int(data["butterworth_order"]) != int(transform["filter"]["order"])
        or data["phase_policy"] != _ADAPTIVE_PHASE_POLICY
        or int(data["resample_up"]) != int(transform["resampler"]["up"])
        or int(data["resample_down"]) != int(transform["resampler"]["down"])
    ):
        raise ValueError("adaptive whole-record transform parameters drifted")

    duration = float(binding["recording_duration_seconds"])
    interval = data["whole_record_interval_recording_seconds"]
    source_samples = data["whole_record_source_sample_interval"]
    output_samples = data["whole_record_output_sample_interval"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or [float(item) for item in interval] != [0.0, duration]
        or not isinstance(source_samples, list)
        or len(source_samples) != 2
        or not isinstance(output_samples, list)
        or len(output_samples) != 2
    ):
        raise ValueError("adaptive whole-record interval is invalid")
    expected_source = [0, _aligned_sample_edge(duration, source_sfreq, context="recording stop")]
    expected_output = [0, _aligned_sample_edge(duration, output_sfreq, context="recording stop")]
    if list(map(int, source_samples)) != expected_source:
        raise ValueError("adaptive whole-record source clock drifted")
    if (
        list(map(int, output_samples)) != expected_output
        or data["whole_record_output_sample_count"] != expected_output[1]
        or parent["selected_global_output_sample_interval"] != expected_output
        or [float(item) for item in parent["selected_recording_seconds"]]
        != [0.0, duration]
    ):
        raise ValueError("adaptive whole-record output clock drifted")

    edge_intervals = _validate_sample_intervals(
        data["global_edge_invalid_output_sample_intervals"],
        context="adaptive whole-record edge intervals",
        upper=expected_output[1],
    )
    parent_edges = _validate_sample_intervals(
        parent["edge_invalid_output_sample_intervals"],
        context="adaptive parent edge intervals",
        upper=expected_output[1],
    )
    left = int(transform["edge_handling"]["left_invalid_samples"])
    right = int(transform["edge_handling"]["right_invalid_samples"])
    expected_edges: list[list[int]] = []
    if left:
        expected_edges.append([0, left])
    if right:
        expected_edges.append([expected_output[1] - right, expected_output[1]])
    if edge_intervals != expected_edges or parent_edges != expected_edges:
        raise ValueError("adaptive whole-record global edge mask drifted")

    if data["reference_policy"] != (
        "common_average_directly_observed_standard19_missing_zero_masked_v1"
    ):
        raise ValueError("adaptive whole-record reference policy drifted")
    observed_set = set(observed)
    expected_derivations = [
        f"{left_name}-{right_name}"
        for left_name, right_name in TCP_20_EDGES
        if left_name in observed_set and right_name in observed_set
    ]
    if data["eligible_bipolar_derivations"] != expected_derivations:
        raise ValueError("adaptive whole-record eligible derivations drifted")
    if data["crop_policy"] != _ADAPTIVE_CROP_POLICY:
        raise ValueError("adaptive whole-record crop policy drifted")
    if data["scope_receipt"] != {
        "eeg_samples_used": True,
        "edf_annotation_api_called": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "navigation_only": True,
        "onset_evidence_authorized": False,
        "findings_evidence_authorized": False,
        "detector_provider_native_transform_modified": False,
        "event_local_filter_or_resample_used": False,
    }:
        raise ValueError("adaptive whole-record transform violates its scope")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("adaptive whole-record receipt hash does not bind content")
    return data


def validate_adaptive_preprocessing_receipt(payload: object) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("adaptive preprocessing receipt must be an object")
    required = {
        "schema_version",
        "method",
        "source_signal_sha256",
        "source_identity_semantics",
        "canonical_signal_binding",
        "montage_reference_observability_receipt_sha256",
        "acquisition_montage_class",
        "semantic_channels",
        "raw_channel_names",
        "raw_units",
        "source_sampling_rate_hz",
        "output_sampling_rate_hz",
        "requested_interval_recording_seconds",
        "source_sample_interval",
        "output_sample_interval",
        "output_sample_count",
        "highpass_hz",
        "lowpass_hz",
        "butterworth_order",
        "phase_policy",
        "resample_up",
        "resample_down",
        "reference_policy",
        "eligible_bipolar_derivations",
        "whole_record_transform_receipt",
        "crop_policy",
        "crop_intersecting_global_edge_invalid_output_sample_intervals",
        "processed_envelope_sha256",
        "scope_receipt",
        "receipt_sha256",
    }
    if set(payload) != required:
        raise ValueError("adaptive preprocessing receipt has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != ADAPTIVE_PREPROCESSING_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive preprocessing schema")
    if data["method"] != ADAPTIVE_PREPROCESSING_METHOD:
        raise ValueError("adaptive preprocessing method drifted")
    if data["source_identity_semantics"] != (
        "detector_manifest_edf_container_sha256"
    ):
        raise ValueError("adaptive preprocessing source hash semantics drifted")
    binding = validate_canonical_adaptive_signal_binding(
        data["canonical_signal_binding"]
    )
    whole = validate_adaptive_whole_record_transform_receipt(
        data["whole_record_transform_receipt"]
    )
    data["canonical_signal_binding"] = binding
    data["whole_record_transform_receipt"] = whole
    if (
        binding != whole["canonical_signal_binding"]
        or data["montage_reference_observability_receipt_sha256"]
        != whole["montage_reference_observability_receipt_sha256"]
        or data["acquisition_montage_class"] != whole["acquisition_montage_class"]
        or data["source_signal_sha256"] != whole["source_signal_sha256"]
        or data["semantic_channels"] != list(STANDARD_19)
        or data["semantic_channels"] != whole["semantic_channels"]
        or data["raw_channel_names"] != whole["raw_channel_names"]
        or data["raw_units"] != whole["raw_units"]
    ):
        raise ValueError("adaptive event/whole-record identity binding drifted")
    for key in (
        "source_signal_sha256",
        "montage_reference_observability_receipt_sha256",
        "processed_envelope_sha256",
        "receipt_sha256",
    ):
        _require_sha256(data[key], context=f"adaptive preprocessing {key}")

    source_sfreq = float(data["source_sampling_rate_hz"])
    output_sfreq = float(data["output_sampling_rate_hz"])
    if (
        source_sfreq != float(whole["source_sampling_rate_hz"])
        or output_sfreq != float(whole["output_sampling_rate_hz"])
        or not math.isfinite(source_sfreq)
        or not math.isfinite(output_sfreq)
        or min(source_sfreq, output_sfreq) <= 0
    ):
        raise ValueError("adaptive preprocessing sampling rates are invalid")
    interval = data["requested_interval_recording_seconds"]
    source_samples = data["source_sample_interval"]
    output_samples = data["output_sample_interval"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not isinstance(source_samples, list)
        or len(source_samples) != 2
        or not isinstance(output_samples, list)
        or len(output_samples) != 2
        or float(interval[1]) <= float(interval[0])
    ):
        raise ValueError("adaptive preprocessing interval is invalid")
    expected_source = [
        _aligned_sample_edge(interval[0], source_sfreq, context="event source start"),
        _aligned_sample_edge(interval[1], source_sfreq, context="event source stop"),
    ]
    expected_output = [
        _aligned_sample_edge(interval[0], output_sfreq, context="event output start"),
        _aligned_sample_edge(interval[1], output_sfreq, context="event output stop"),
    ]
    if list(map(int, source_samples)) != expected_source:
        raise ValueError("adaptive preprocessing source sample clock drifted")
    if (
        list(map(int, output_samples)) != expected_output
        or data["output_sample_count"] != expected_output[1] - expected_output[0]
        or expected_output[0] < whole["whole_record_output_sample_interval"][0]
        or expected_output[1] > whole["whole_record_output_sample_interval"][1]
    ):
        raise ValueError("adaptive preprocessing output sample clock drifted")
    if float(interval[1]) > float(binding["recording_duration_seconds"]) + 1e-8:
        raise ValueError("adaptive preprocessing interval exceeds canonical EEG")
    if (
        float(data["highpass_hz"]) != float(whole["highpass_hz"])
        or float(data["lowpass_hz"]) != float(whole["lowpass_hz"])
        or int(data["butterworth_order"]) != int(whole["butterworth_order"])
        or data["phase_policy"] != _ADAPTIVE_PHASE_POLICY
        or int(data["resample_up"]) != int(whole["resample_up"])
        or int(data["resample_down"]) != int(whole["resample_down"])
        or data["reference_policy"] != whole["reference_policy"]
        or data["eligible_bipolar_derivations"]
        != whole["eligible_bipolar_derivations"]
        or data["crop_policy"] != _ADAPTIVE_CROP_POLICY
    ):
        raise ValueError("adaptive event transform drifted from its whole-record parent")

    expected_edge_overlap = [
        [max(edge[0], expected_output[0]), min(edge[1], expected_output[1])]
        for edge in whole["global_edge_invalid_output_sample_intervals"]
        if max(edge[0], expected_output[0]) < min(edge[1], expected_output[1])
    ]
    edge_overlap = _validate_sample_intervals(
        data["crop_intersecting_global_edge_invalid_output_sample_intervals"],
        context="adaptive crop edge overlap",
        upper=whole["whole_record_output_sample_count"],
    )
    if edge_overlap != expected_edge_overlap:
        raise ValueError("adaptive crop edge overlap drifted")
    if data["scope_receipt"] != {
        "eeg_samples_used": True,
        "edf_annotation_api_called": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
    }:
        raise ValueError("adaptive preprocessing violates the EEG-only scope")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("adaptive preprocessing receipt hash does not bind content")
    return data


def _adaptive_canonical_config(
    *,
    output_sampling_rate_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    butterworth_order: int,
) -> CanonicalEDFConfig:
    values = (
        float(output_sampling_rate_hz),
        float(highpass_hz),
        float(lowpass_hz),
    )
    if (
        not all(math.isfinite(value) and value > 0 for value in values)
        or values[1] >= values[2]
        or isinstance(butterworth_order, bool)
        or not isinstance(butterworth_order, int)
        or butterworth_order < 1
    ):
        raise ValueError("adaptive EDF filter configuration is invalid")
    return CanonicalEDFConfig(
        output_sampling_rate_hz=values[0],
        findings_highpass_hz=values[1],
        findings_lowpass_hz=values[2],
        butterworth_order=butterworth_order,
    )


def _materialize_whole_record_adaptive_navigation(
    edf_path: str | Path,
    *,
    canonical_bundle: CanonicalEDFViewBundle,
    source_signal_sha256: str | None,
    output_sampling_rate_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    butterworth_order: int,
) -> _WholeRecordingAdaptiveNavigationView:
    """Build one immutable, navigation-only full-record tensor.

    The zero-phase filter and resampler live exclusively in the canonical
    ``context_offline`` parent.  This function adds a recording-global common
    average across directly observed channels and zero carriers for missing
    Standard-19 channels.  It never filters or resamples an event interval.
    """

    path = _source_edf(edf_path)
    actual_source_hash = _file_sha256(path)
    if source_signal_sha256 is not None and source_signal_sha256 != actual_source_hash:
        raise ValueError("adaptive EDF hash does not match the detection manifest")
    bundle = canonical_bundle
    validate_canonical_edf_materialization(bundle)
    binding = build_canonical_adaptive_signal_binding(bundle)
    record = bundle.canonical_record
    # The adaptive CAR remains navigation-only, but it still must not be
    # fabricated from an already-bipolar, mixed, or unknown acquisition
    # montage.  The canonical bundle carries the signal-label-only receipt.
    montage = require_reference_materialization_authorized(
        record.montage_reference_observability_receipt,
        reference_kinds=("car",),
    )
    header_rows = record.source_header_receipt["channel_signal_headers"]
    if not header_rows:
        raise ValueError("adaptive canonical EEG has no directly observed channels")
    rate_pairs = {
        (
            int(row["sampling_rate_numerator"]),
            int(row["sampling_rate_denominator"]),
        )
        for row in header_rows
    }
    if len(rate_pairs) != 1:
        raise ValueError("adaptive canonical channels must share one sampling clock")
    source_num, source_den = next(iter(rate_pairs))
    source_sfreq = source_num / source_den

    parent_view = bundle.context_offline
    parent_receipt = parent_view.receipt
    transform = validate_transform_spec(parent_receipt["transform_spec"])
    output_sfreq = (
        int(transform["output_clock"]["sampling_rate_numerator"])
        / int(transform["output_clock"]["sampling_rate_denominator"])
    )
    if (
        abs(output_sfreq - float(output_sampling_rate_hz)) > 1e-9
        or float(transform["filter"]["highpass_hz"]) != float(highpass_hz)
        or float(transform["filter"]["lowpass_hz"]) != float(lowpass_hz)
        or int(transform["filter"]["order"]) != int(butterworth_order)
    ):
        raise ValueError(
            "canonical context_offline cannot represent the requested adaptive bandwidth"
        )
    if float(lowpass_hz) >= 0.5 * min(source_sfreq, output_sfreq):
        raise ValueError("adaptive lowpass must be below source and output Nyquist")

    observed_ids = tuple(binding["observed_channel_ids"])
    if len(observed_ids) < 2:
        raise ValueError("adaptive observed-channel CAR requires at least two electrodes")
    if tuple(record.observed_channel_ids) != observed_ids:
        raise ValueError("adaptive canonical observed-channel order drifted")
    parent = parent_view.tensor.detach().cpu().to(torch.float64).numpy()
    if (
        parent.shape[0] != len(STANDARD_19)
        or not np.isfinite(parent).all()
        or parent.shape[1] < 1
    ):
        raise ValueError("adaptive canonical whole-record parent tensor is invalid")
    observed_rows = [STANDARD_19.index(channel_id) for channel_id in observed_ids]
    observed_values = parent[observed_rows].copy()
    observed_values -= np.mean(observed_values, axis=0, keepdims=True)
    carrier = np.zeros(parent.shape, dtype=np.float64)
    carrier[observed_rows] = observed_values
    signal = torch.from_numpy(carrier.astype(np.float32, copy=False)).contiguous()
    if not torch.isfinite(signal).all():
        raise ValueError("adaptive whole-record navigation tensor is non-finite")

    duration = float(binding["recording_duration_seconds"])
    source_count = int(record.observed_signal_volts.shape[1])
    output_count = int(signal.shape[1])
    expected_source_count = _aligned_sample_edge(
        duration, source_sfreq, context="whole recording source stop"
    )
    expected_output_count = _aligned_sample_edge(
        duration, output_sfreq, context="whole recording output stop"
    )
    if source_count != expected_source_count or output_count != expected_output_count:
        raise ValueError("adaptive whole-record tensor drifted from the global clock")
    coordinates = parent_receipt["coordinates"]
    parent_edges = parent_receipt["masks"]["edge_invalid_intervals"]
    observed_set = set(observed_ids)
    raw_names = [str(row["raw_label"]) for row in header_rows]
    raw_units = [str(row["raw_physical_dimension"]) for row in header_rows]
    whole_hash = generalized_signal_tensor_sha256(signal)
    observed_car_scale = 1.0 / len(observed_ids)
    observed_car_matrix: list[list[float]] = []
    for target_index in range(len(observed_ids)):
        row = [-observed_car_scale] * len(observed_ids)
        row[target_index] += 1.0
        observed_car_matrix.append(row)
    observed_car_observability = build_reference_matrix_observability(
        row_unit_ids=[f"{channel}-CAR-NAV" for channel in observed_ids],
        column_unit_ids=observed_ids,
        matrix=observed_car_matrix,
    )
    parent_binding = {
        "view_id": parent_receipt["view_id"],
        "receipt_sha256": parent_receipt["receipt_sha256"],
        "processed_view_sha256": parent_receipt["processed_view_sha256"],
        "transform_spec": transform,
        "selected_global_output_sample_interval": deepcopy(
            coordinates["selected_global_output_sample_interval"]
        ),
        "selected_recording_seconds": deepcopy(
            coordinates["selected_recording_seconds"]
        ),
        "edge_invalid_output_sample_intervals": deepcopy(parent_edges),
    }
    body = {
        "schema_version": ADAPTIVE_WHOLE_RECORD_TRANSFORM_SCHEMA_VERSION,
        "method": ADAPTIVE_WHOLE_RECORD_TRANSFORM_METHOD,
        "role": "adaptive_boundary_navigation_only",
        # Container identity remains distinct from canonical physical-signal
        # identity, which is carried in canonical_signal_binding.
        "source_signal_sha256": actual_source_hash,
        "source_identity_semantics": "detector_manifest_edf_container_sha256",
        "canonical_signal_binding": binding,
        "canonical_materialization_receipt_sha256": bundle.materialization_receipt[
            "receipt_sha256"
        ],
        "montage_reference_observability_receipt_sha256": montage[
            "receipt_sha256"
        ],
        "acquisition_montage_class": montage["montage_class"],
        "observed_car_reference_matrix_observability": observed_car_observability,
        "parent_context_offline_view": parent_binding,
        "semantic_channels": list(STANDARD_19),
        "raw_channel_names": raw_names,
        "raw_units": raw_units,
        "source_sampling_rate_hz": source_sfreq,
        "output_sampling_rate_hz": output_sfreq,
        "whole_record_interval_recording_seconds": [0.0, duration],
        "whole_record_source_sample_interval": [0, source_count],
        "whole_record_output_sample_interval": [0, output_count],
        "whole_record_output_sample_count": output_count,
        "highpass_hz": float(highpass_hz),
        "lowpass_hz": float(lowpass_hz),
        "butterworth_order": int(butterworth_order),
        "phase_policy": _ADAPTIVE_PHASE_POLICY,
        "resample_up": int(transform["resampler"]["up"]),
        "resample_down": int(transform["resampler"]["down"]),
        "reference_policy": (
            "common_average_directly_observed_standard19_missing_zero_masked_v1"
        ),
        "eligible_bipolar_derivations": [
            f"{left}-{right}"
            for left, right in TCP_20_EDGES
            if left in observed_set and right in observed_set
        ],
        "global_edge_invalid_output_sample_intervals": deepcopy(parent_edges),
        "whole_record_processed_navigation_sha256": whole_hash,
        "crop_policy": _ADAPTIVE_CROP_POLICY,
        "scope_receipt": {
            "eeg_samples_used": True,
            "edf_annotation_api_called": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
            "navigation_only": True,
            "onset_evidence_authorized": False,
            "findings_evidence_authorized": False,
            "detector_provider_native_transform_modified": False,
            "event_local_filter_or_resample_used": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    receipt = validate_adaptive_whole_record_transform_receipt(body)
    if generalized_signal_tensor_sha256(signal) != receipt[
        "whole_record_processed_navigation_sha256"
    ]:
        raise ValueError("adaptive whole-record receipt does not bind its tensor")
    return _WholeRecordingAdaptiveNavigationView(
        signal=signal,
        transform_receipt=receipt,
    )


def _crop_adaptive_navigation_envelope(
    whole_view: _WholeRecordingAdaptiveNavigationView,
    *,
    start_recording_seconds: float,
    stop_recording_seconds: float,
) -> LoadedAdaptiveEnvelope:
    if not isinstance(whole_view, _WholeRecordingAdaptiveNavigationView):
        raise TypeError("adaptive whole-record view is invalid")
    whole = validate_adaptive_whole_record_transform_receipt(
        whole_view.transform_receipt
    )
    if generalized_signal_tensor_sha256(whole_view.signal) != whole[
        "whole_record_processed_navigation_sha256"
    ]:
        raise ValueError("adaptive whole-record transform receipt does not bind its tensor")
    start = float(start_recording_seconds)
    stop = float(stop_recording_seconds)
    duration = float(whole["canonical_signal_binding"]["recording_duration_seconds"])
    if (
        not math.isfinite(start)
        or not math.isfinite(stop)
        or start < 0
        or stop <= start
        or stop > duration + 1e-9
    ):
        raise ValueError("adaptive EDF interval is invalid")
    source_sfreq = float(whole["source_sampling_rate_hz"])
    output_sfreq = float(whole["output_sampling_rate_hz"])
    source_interval = [
        _aligned_sample_edge(start, source_sfreq, context="adaptive source start"),
        _aligned_sample_edge(stop, source_sfreq, context="adaptive source stop"),
    ]
    output_interval = [
        _aligned_sample_edge(start, output_sfreq, context="adaptive output start"),
        _aligned_sample_edge(stop, output_sfreq, context="adaptive output stop"),
    ]
    if output_interval[1] > int(whole_view.signal.shape[1]):
        raise ValueError("adaptive output crop extends beyond the whole-record tensor")
    signal = whole_view.signal[
        :, output_interval[0] : output_interval[1]
    ].clone().contiguous()
    if signal.shape != (
        len(STANDARD_19),
        output_interval[1] - output_interval[0],
    ):
        raise ValueError("adaptive whole-record crop has an invalid shape")
    processed_hash = generalized_signal_tensor_sha256(signal)
    edge_overlap = [
        [max(edge[0], output_interval[0]), min(edge[1], output_interval[1])]
        for edge in whole["global_edge_invalid_output_sample_intervals"]
        if max(edge[0], output_interval[0]) < min(edge[1], output_interval[1])
    ]
    body = {
        "schema_version": ADAPTIVE_PREPROCESSING_SCHEMA_VERSION,
        "method": ADAPTIVE_PREPROCESSING_METHOD,
        "source_signal_sha256": whole["source_signal_sha256"],
        "source_identity_semantics": whole["source_identity_semantics"],
        "canonical_signal_binding": whole["canonical_signal_binding"],
        "montage_reference_observability_receipt_sha256": whole[
            "montage_reference_observability_receipt_sha256"
        ],
        "acquisition_montage_class": whole["acquisition_montage_class"],
        "semantic_channels": whole["semantic_channels"],
        "raw_channel_names": whole["raw_channel_names"],
        "raw_units": whole["raw_units"],
        "source_sampling_rate_hz": source_sfreq,
        "output_sampling_rate_hz": output_sfreq,
        "requested_interval_recording_seconds": [start, stop],
        "source_sample_interval": source_interval,
        "output_sample_interval": output_interval,
        "output_sample_count": int(signal.shape[1]),
        "highpass_hz": whole["highpass_hz"],
        "lowpass_hz": whole["lowpass_hz"],
        "butterworth_order": whole["butterworth_order"],
        "phase_policy": whole["phase_policy"],
        "resample_up": whole["resample_up"],
        "resample_down": whole["resample_down"],
        "reference_policy": whole["reference_policy"],
        "eligible_bipolar_derivations": whole["eligible_bipolar_derivations"],
        "whole_record_transform_receipt": whole,
        "crop_policy": _ADAPTIVE_CROP_POLICY,
        "crop_intersecting_global_edge_invalid_output_sample_intervals": edge_overlap,
        "processed_envelope_sha256": processed_hash,
        "scope_receipt": {
            "eeg_samples_used": True,
            "edf_annotation_api_called": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    receipt = validate_adaptive_preprocessing_receipt(body)
    return LoadedAdaptiveEnvelope(signal=signal, preprocessing_receipt=receipt)


def _load_standard19_adaptive_envelope_from_canonical(
    edf_path: str | Path,
    *,
    start_recording_seconds: float,
    stop_recording_seconds: float,
    source_signal_sha256: str | None = None,
    output_sampling_rate_hz: float = 100.0,
    highpass_hz: float = 0.5,
    lowpass_hz: float = 40.0,
    butterworth_order: int = 4,
    canonical_bundle: CanonicalEDFViewBundle,
) -> LoadedAdaptiveEnvelope:
    whole = _materialize_whole_record_adaptive_navigation(
        edf_path,
        canonical_bundle=canonical_bundle,
        source_signal_sha256=source_signal_sha256,
        output_sampling_rate_hz=output_sampling_rate_hz,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        butterworth_order=butterworth_order,
    )
    return _crop_adaptive_navigation_envelope(
        whole,
        start_recording_seconds=start_recording_seconds,
        stop_recording_seconds=stop_recording_seconds,
    )


def load_standard19_adaptive_envelope(
    edf_path: str | Path,
    *,
    start_recording_seconds: float,
    stop_recording_seconds: float,
    source_signal_sha256: str | None = None,
    output_sampling_rate_hz: float = 100.0,
    highpass_hz: float = 0.5,
    lowpass_hz: float = 40.0,
    butterworth_order: int = 4,
    reader_factory: Callable[[str], object] | None = None,
) -> LoadedAdaptiveEnvelope:
    """Load one canonical root from ``edf_path`` and derive its search view.

    The public function intentionally does not accept a prebuilt canonical
    bundle: without a co-produced container attestation, a caller could pair a
    detector manifest for EDF A with a canonical tensor from EDF B.  The batch
    materializer uses the private helper only after it has itself loaded the
    canonical bundle from this exact path once.
    """

    path = _source_edf(edf_path)
    config = _adaptive_canonical_config(
        output_sampling_rate_hz=output_sampling_rate_hz,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        butterworth_order=butterworth_order,
    )
    bundle = load_canonical_edf_views(
        path,
        config=config,
        reader_factory=reader_factory,
    )
    return _load_standard19_adaptive_envelope_from_canonical(
        path,
        start_recording_seconds=start_recording_seconds,
        stop_recording_seconds=stop_recording_seconds,
        source_signal_sha256=source_signal_sha256,
        output_sampling_rate_hz=output_sampling_rate_hz,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        butterworth_order=butterworth_order,
        canonical_bundle=bundle,
    )


def _atomic_json(path: Path, value: object) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, target)
        os.chmod(target, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def validate_adaptive_materialization_artifact(payload: object) -> dict[str, Any]:
    """Strictly validate all plan/preprocessing/search bindings in an artifact."""

    if type(payload) is not dict:
        raise TypeError("adaptive materialization artifact must be an object")
    required = {
        "schema_version",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "canonical_signal_binding",
        "recording_duration_seconds",
        "detection_manifest_sha256",
        "event_count",
        "events",
        "scope_receipt",
        "artifact_sha256",
    }
    if set(payload) != required:
        raise ValueError("adaptive materialization has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != ADAPTIVE_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive materialization schema")
    for field in ("recording_id", "patient_pseudonym"):
        if not isinstance(data[field], str) or not data[field] or data[field] != data[field].strip():
            raise ValueError(f"adaptive materialization {field} is invalid")
    for field in (
        "source_signal_sha256",
        "detection_manifest_sha256",
        "artifact_sha256",
    ):
        value = data[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"adaptive materialization {field} is not SHA-256")
    canonical_binding = (
        None
        if data["canonical_signal_binding"] is None
        else validate_canonical_adaptive_signal_binding(
            data["canonical_signal_binding"]
        )
    )
    data["canonical_signal_binding"] = canonical_binding
    duration = float(data["recording_duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("adaptive materialization duration is invalid")
    if canonical_binding is not None and abs(
        float(canonical_binding["recording_duration_seconds"]) - duration
    ) > 1e-6:
        raise ValueError("adaptive materialization canonical clock drifted")
    events = data["events"]
    if not isinstance(events, list) or data["event_count"] != len(events):
        raise ValueError("adaptive materialization event count is invalid")
    if events and canonical_binding is None:
        raise ValueError(
            "adaptive materialization with candidates needs canonical binding"
        )
    seen_candidates: set[str] = set()
    seen_events: set[str] = set()
    previous_anchor = -math.inf
    shared_whole_record_transform_sha256: str | None = None
    for index, raw_event in enumerate(events):
        if type(raw_event) is not dict or set(raw_event) != {
            "candidate_id",
            "eeg_event_id",
            "status",
            "plan",
            "preprocessing_receipt",
            "adaptive_search_receipt",
        }:
            raise ValueError(f"adaptive materialization event {index} is invalid")
        candidate_id = raw_event["candidate_id"]
        event_id = raw_event["eeg_event_id"]
        if (
            not isinstance(candidate_id, str)
            or not isinstance(event_id, str)
            or candidate_id in seen_candidates
            or event_id in seen_events
        ):
            raise ValueError("adaptive materialization event identities are invalid")
        seen_candidates.add(candidate_id)
        seen_events.add(event_id)
        plan = validate_adaptive_search_envelope_plan(raw_event["plan"])
        if (
            plan["candidate_id"] != candidate_id
            or plan["eeg_event_id"] != event_id
            or plan["recording_id"] != data["recording_id"]
            or plan["patient_pseudonym"] != data["patient_pseudonym"]
            or plan["source_signal_sha256"] != data["source_signal_sha256"]
            or abs(plan["recording_duration_seconds"] - duration) > 1e-6
        ):
            raise ValueError("adaptive materialization plan binding drifted")
        anchor = float(plan["coarse_anchor_recording_seconds"])
        if anchor < previous_anchor:
            raise ValueError("adaptive materialization events are out of recording order")
        previous_anchor = anchor
        preprocessing_raw = raw_event["preprocessing_receipt"]
        search_raw = raw_event["adaptive_search_receipt"]
        if raw_event["status"] == "abstained_plan_context_unavailable":
            if (
                "anchor_context_unavailable"
                not in plan["boundary_truncation_reasons"]
                or preprocessing_raw is not None
                or search_raw is not None
            ):
                raise ValueError("adaptive plan-context abstention is inconsistent")
            continue
        preprocessing = validate_adaptive_preprocessing_receipt(preprocessing_raw)
        search = validate_adaptive_search_receipt(search_raw)
        event_whole_hash = preprocessing["whole_record_transform_receipt"][
            "receipt_sha256"
        ]
        if shared_whole_record_transform_sha256 is None:
            shared_whole_record_transform_sha256 = event_whole_hash
        elif event_whole_hash != shared_whole_record_transform_sha256:
            raise ValueError(
                "adaptive events do not share one whole-record transform"
            )
        if raw_event["status"] != search["status"]:
            raise ValueError("adaptive materialization status differs from search")
        if (
            preprocessing["source_signal_sha256"] != data["source_signal_sha256"]
            or preprocessing["canonical_signal_binding"] != canonical_binding
            or preprocessing["requested_interval_recording_seconds"]
            != plan["effective_interval_recording_seconds"]
            or search["processed_envelope_sha256"]
            != preprocessing["processed_envelope_sha256"]
            or search["preprocessing_receipt_sha256"]
            != preprocessing["receipt_sha256"]
            or search["envelope_interval_recording_seconds"]
            != plan["effective_interval_recording_seconds"]
            or abs(search["coarse_anchor_recording_seconds"] - anchor) > 1e-6
            or abs(search["recording_duration_seconds"] - duration) > 1e-6
            or search["canonical_signal_binding"] != canonical_binding
        ):
            raise ValueError("adaptive materialization signal/search binding drifted")
    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotation_api_called": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "coarse_anchor_used_for_navigation_only": True,
        "fixed_v29_window_used_as_search_range": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("adaptive materialization violates the EEG-only scope")
    digest_source = deepcopy(data)
    digest_source["artifact_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["artifact_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("adaptive materialization hash does not bind its content")
    return data


def materialize_adaptive_eeg_search(
    *,
    detection_manifest_path: Path,
    edf_path: Path,
    output_path: Path,
    event_id_by_candidate: Mapping[str, str] | None = None,
    envelope_loader: AdaptiveEnvelopeLoader = load_standard19_adaptive_envelope,
) -> dict[str, Any]:
    """Materialize one content-bound adaptive search artifact atomically."""

    manifest_path = detection_manifest_path.resolve(strict=True)
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = validate_long_term_seizure_detection_manifest(manifest_raw)
    source = _source_edf(edf_path)
    if _file_sha256(source) != manifest["source_signal_sha256"]:
        raise ValueError("adaptive-search EDF does not match the detection manifest")
    resolved_bundle: CanonicalEDFViewBundle | None = None
    resolved_whole_view: _WholeRecordingAdaptiveNavigationView | None = None
    if envelope_loader is load_standard19_adaptive_envelope:
        # The production path reads/validates the canonical physical EEG and
        # applies the offline transform exactly once.  Every event, including
        # later expansions of the same event, is a crop on this frozen global
        # output clock.  Custom loaders must carry the equivalent v3 receipt.
        adaptive_config = _adaptive_canonical_config(
            output_sampling_rate_hz=100.0,
            highpass_hz=0.5,
            lowpass_hz=40.0,
            butterworth_order=4,
        )
        resolved_bundle = load_canonical_edf_views(
            source,
            config=adaptive_config,
        )
        resolved_whole_view = _materialize_whole_record_adaptive_navigation(
            source,
            canonical_bundle=resolved_bundle,
            source_signal_sha256=manifest["source_signal_sha256"],
            output_sampling_rate_hz=100.0,
            highpass_hz=0.5,
            lowpass_hz=40.0,
            butterworth_order=4,
        )
    canonical_binding = (
        build_canonical_adaptive_signal_binding(resolved_bundle)
        if resolved_bundle is not None
        else None
    )
    plans = plan_adaptive_search_envelopes(
        manifest, event_id_by_candidate=event_id_by_candidate
    )
    events: list[dict[str, Any]] = []
    for plan_raw in plans:
        plan = validate_adaptive_search_envelope_plan(plan_raw)
        start, stop = plan["effective_interval_recording_seconds"]
        if (
            "anchor_context_unavailable" in plan["boundary_truncation_reasons"]
            or not start < plan["coarse_anchor_recording_seconds"] < stop
        ):
            events.append(
                {
                    "candidate_id": plan["candidate_id"],
                    "eeg_event_id": plan["eeg_event_id"],
                    "status": "abstained_plan_context_unavailable",
                    "plan": plan,
                    "preprocessing_receipt": None,
                    "adaptive_search_receipt": None,
                }
            )
            continue
        loader_kwargs: dict[str, object] = {
            "start_recording_seconds": start,
            "stop_recording_seconds": stop,
            "source_signal_sha256": manifest["source_signal_sha256"],
        }
        if resolved_whole_view is not None:
            loaded = _crop_adaptive_navigation_envelope(
                resolved_whole_view,
                start_recording_seconds=start,
                stop_recording_seconds=stop,
            )
        else:
            loaded = envelope_loader(source, **loader_kwargs)
        if not isinstance(loaded, LoadedAdaptiveEnvelope):
            raise TypeError("adaptive envelope loader returned an invalid object")
        preprocessing = validate_adaptive_preprocessing_receipt(
            loaded.preprocessing_receipt
        )
        event_binding = preprocessing["canonical_signal_binding"]
        if canonical_binding is None:
            canonical_binding = event_binding
        elif event_binding != canonical_binding:
            raise ValueError(
                "adaptive events do not share one canonical signal binding"
            )
        if preprocessing["requested_interval_recording_seconds"] != [start, stop]:
            raise ValueError("adaptive preprocessing interval does not match its plan")
        signal_hash = generalized_signal_tensor_sha256(loaded.signal)
        if signal_hash != preprocessing["processed_envelope_sha256"]:
            raise ValueError("adaptive preprocessing receipt does not bind its tensor")
        search = analyze_adaptive_eeg_envelope(
            loaded.signal,
            sampling_rate_hz=preprocessing["output_sampling_rate_hz"],
            envelope_start_recording_seconds=start,
            candidate_anchor_recording_seconds=plan[
                "coarse_anchor_recording_seconds"
            ],
            recording_duration_seconds=manifest["recording_duration_seconds"],
            processed_envelope_sha256=signal_hash,
            preprocessing_receipt_sha256=preprocessing["receipt_sha256"],
            canonical_signal_binding=canonical_binding,
        )
        search = validate_adaptive_search_receipt(search)
        events.append(
            {
                "candidate_id": plan["candidate_id"],
                "eeg_event_id": plan["eeg_event_id"],
                "status": search["status"],
                "plan": plan,
                "preprocessing_receipt": preprocessing,
                "adaptive_search_receipt": search,
            }
        )
    body = {
        "schema_version": ADAPTIVE_MATERIALIZATION_SCHEMA_VERSION,
        "recording_id": manifest["recording_id"],
        "patient_pseudonym": manifest["patient_pseudonym"],
        "source_signal_sha256": manifest["source_signal_sha256"],
        "canonical_signal_binding": canonical_binding,
        "recording_duration_seconds": manifest["recording_duration_seconds"],
        "detection_manifest_sha256": canonical_payload_sha256(manifest),
        "event_count": len(events),
        "events": events,
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotation_api_called": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
            "coarse_anchor_used_for_navigation_only": True,
            "fixed_v29_window_used_as_search_range": False,
        },
        "artifact_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["artifact_sha256"] = _canonical_sha256(body)
    body = validate_adaptive_materialization_artifact(body)
    _atomic_json(output_path, body)
    return body


__all__ = [
    "ADAPTIVE_MATERIALIZATION_SCHEMA_VERSION",
    "ADAPTIVE_PREPROCESSING_METHOD",
    "ADAPTIVE_PREPROCESSING_SCHEMA_VERSION",
    "ADAPTIVE_WHOLE_RECORD_TRANSFORM_METHOD",
    "ADAPTIVE_WHOLE_RECORD_TRANSFORM_SCHEMA_VERSION",
    "LoadedAdaptiveEnvelope",
    "load_standard19_adaptive_envelope",
    "materialize_adaptive_eeg_search",
    "validate_adaptive_materialization_artifact",
    "validate_adaptive_preprocessing_receipt",
    "validate_adaptive_whole_record_transform_receipt",
]
