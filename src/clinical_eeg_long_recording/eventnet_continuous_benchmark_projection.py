"""Project frozen EventNet outputs into the model-neutral benchmark contract.

The first projection is prediction-only and therefore remains EEG-only.  A
second, explicit post-inference join may add public reference intervals and a
patient/split key before calling the shared continuous-detection scorer.  This
separation prevents reference events from reaching the detector provider.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .continuous_detection_benchmark import validate_continuous_benchmark_rows
from .eventnet_full_record_adapter import (
    EVENTNET_PROVIDER_ID,
    validate_eventnet_prediction_receipt,
)


EVENTNET_BENCHMARK_PROJECTION_SCHEMA_VERSION = (
    "eventnet_continuous_benchmark_prediction_projection_v1"
)
EVENTNET_BENCHMARK_PROJECTION_METHOD_ID = (
    "frozen_direct_event_intervals_to_benchmark_v5_prediction_sidecar_v1"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_nonnegative(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be finite and non-negative")
    return result


def validate_eventnet_benchmark_prediction_projection(
    payload: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "projection_id",
        "method_id",
        "provider_id",
        "prediction_id",
        "prediction_receipt_sha256",
        "recording_id",
        "duration_seconds",
        "predicted_events",
        "execution_receipt",
        "join_permissions",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("EventNet benchmark projection fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != EVENTNET_BENCHMARK_PROJECTION_SCHEMA_VERSION
        or data["method_id"] != EVENTNET_BENCHMARK_PROJECTION_METHOD_ID
        or data["provider_id"] != EVENTNET_PROVIDER_ID
    ):
        raise ValueError("EventNet benchmark projection identity drifted")
    for name in ("projection_id", "prediction_id", "recording_id"):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError(f"EventNet benchmark projection {name} is invalid")
    receipt_sha = data["prediction_receipt_sha256"]
    if (
        not isinstance(receipt_sha, str)
        or len(receipt_sha) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha)
    ):
        raise ValueError("EventNet prediction receipt SHA-256 is invalid")
    duration = _finite_nonnegative(data["duration_seconds"], "duration")
    if duration <= 0:
        raise ValueError("EventNet benchmark duration must be positive")
    predicted = data["predicted_events"]
    if not isinstance(predicted, list):
        raise TypeError("EventNet predicted events must be an array")
    previous_stop = 0.0
    for index, event in enumerate(predicted):
        if type(event) is not dict or set(event) != {
            "start_seconds",
            "stop_seconds",
        }:
            raise ValueError(f"EventNet predicted event {index} fields drifted")
        start = _finite_nonnegative(event["start_seconds"], "predicted start")
        stop = _finite_nonnegative(event["stop_seconds"], "predicted stop")
        if stop <= start or stop > duration + 1e-9 or (index and start < previous_stop):
            raise ValueError("EventNet predicted events are invalid or overlapping")
        previous_stop = stop
    execution = data["execution_receipt"]
    expected_execution_fields = {
        "edf_io_seconds",
        "preprocessing_seconds",
        "inference_seconds",
        "postprocessing_seconds",
        "total_wall_seconds",
        "gpu_active_seconds",
        "gpu_measurement_status",
        "peak_gpu_memory_bytes",
        "peak_host_memory_bytes",
        "service_state",
        "device_type",
        "native_preprocessing_receipt_id",
        "native_preprocessing_receipt_sha256",
        "complete_recording_coverage",
    }
    if type(execution) is not dict or set(execution) != expected_execution_fields:
        raise ValueError("EventNet benchmark execution receipt fields drifted")
    # Reuse the authoritative generic row validator without inventing a
    # reference fact: a synthetic empty reference array checks only execution
    # and prediction shape here and is never serialized as benchmark truth.
    validate_continuous_benchmark_rows(
        [
            {
                "patient_id": "PREDICTION-PROJECTION-SHAPE-CHECK",
                "recording_id": data["recording_id"],
                "split": "prediction_only_not_scored",
                "duration_seconds": duration,
                "reference_events": [],
                "predicted_events": predicted,
                "execution_receipt": execution,
            }
        ]
    )
    if data["join_permissions"] != {
        "prediction_frozen_before_reference_join": True,
        "public_reference_join_allowed_post_inference": True,
        "private_annotations_or_excel_allowed": False,
        "source_eval_requires_separate_admission_receipt": True,
        "projection_itself_is_accuracy_evidence": False,
    }:
        raise ValueError("EventNet benchmark join permissions were widened")
    digest = deepcopy(data)
    digest["projection_id"] = "EVENTNET-BENCHMARK-PROJECTION-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["projection_id"] != "EVNBENCHPROJ-" + _canonical_sha256(digest)[:24]:
        raise ValueError("EventNet benchmark projection ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet benchmark projection hash is invalid")
    return data


def project_eventnet_prediction_to_benchmark(
    prediction_receipt: Mapping[str, Any],
    *,
    edf_io_seconds: float,
    service_state: str = "cold",
    peak_host_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a prediction-only sidecar accepted by benchmark-v5 after join."""

    prediction = validate_eventnet_prediction_receipt(dict(prediction_receipt))
    edf_seconds = _finite_nonnegative(edf_io_seconds, "EDF I/O seconds")
    if service_state not in {"cold", "warm"}:
        raise ValueError("EventNet service state must be cold or warm")
    if peak_host_memory_bytes is not None and (
        isinstance(peak_host_memory_bytes, bool)
        or not isinstance(peak_host_memory_bytes, int)
        or peak_host_memory_bytes < 0
    ):
        raise ValueError("EventNet peak host memory must be non-negative or null")
    runtime = prediction["runtime_receipt"]
    preprocessing_seconds = sum(
        float(runtime[name])
        for name in (
            "checkpoint_static_audit_and_safe_load_seconds",
            "canonical_carrier_binding_seconds",
            "provider_preprocessing_and_tiling_seconds",
        )
    )
    inference_seconds = float(runtime["model_inference_seconds"])
    postprocessing_seconds = float(runtime["direct_event_decoding_seconds"])
    total_wall = edf_seconds + float(runtime["full_adapter_wall_seconds"])
    preprocessing = prediction["preprocessing_receipt"]
    preprocessing_sha = str(preprocessing["receipt_sha256"])
    device = str(runtime["device"])
    cpu_only = device == "cpu"
    execution = {
        "edf_io_seconds": edf_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": inference_seconds,
        "postprocessing_seconds": postprocessing_seconds,
        "total_wall_seconds": total_wall,
        "gpu_active_seconds": None,
        "gpu_measurement_status": (
            "not_applicable_cpu_only" if cpu_only else "not_measured"
        ),
        "peak_gpu_memory_bytes": None,
        "peak_host_memory_bytes": peak_host_memory_bytes,
        "service_state": service_state,
        "device_type": device,
        "native_preprocessing_receipt_id": ("EVNPREPROC-" + preprocessing_sha[:24]),
        "native_preprocessing_receipt_sha256": preprocessing_sha,
        "complete_recording_coverage": prediction["generic_full_record_result"][
            "coverage_receipt"
        ]["posterior_target_coverage_complete"],
    }
    predicted_events = [
        {
            "start_seconds": float(event["start_offset_seconds"]),
            "stop_seconds": float(event["stop_offset_seconds"]),
        }
        for event in prediction["decoder_receipt"]["event_proposals"]
    ]
    body: dict[str, Any] = {
        "schema_version": EVENTNET_BENCHMARK_PROJECTION_SCHEMA_VERSION,
        "projection_id": "EVENTNET-BENCHMARK-PROJECTION-PENDING",
        "method_id": EVENTNET_BENCHMARK_PROJECTION_METHOD_ID,
        "provider_id": EVENTNET_PROVIDER_ID,
        "prediction_id": prediction["prediction_id"],
        "prediction_receipt_sha256": prediction["receipt_sha256"],
        "recording_id": prediction["recording_id"],
        "duration_seconds": prediction["generic_full_record_result"][
            "recording_duration_seconds"
        ],
        "predicted_events": predicted_events,
        "execution_receipt": execution,
        "join_permissions": {
            "prediction_frozen_before_reference_join": True,
            "public_reference_join_allowed_post_inference": True,
            "private_annotations_or_excel_allowed": False,
            "source_eval_requires_separate_admission_receipt": True,
            "projection_itself_is_accuracy_evidence": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["projection_id"] = "EVNBENCHPROJ-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_eventnet_benchmark_prediction_projection(body)


def join_eventnet_projection_with_public_references(
    projection: Mapping[str, Any],
    *,
    patient_id: str,
    split: str,
    reference_events: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Join frozen public truth after inference and return one benchmark-v5 row.

    This function is not a provider input.  Source-eval callers must still use
    the repository's separate source-eval admission workflow before scoring.
    """

    value = validate_eventnet_benchmark_prediction_projection(dict(projection))
    row = {
        "patient_id": patient_id,
        "recording_id": value["recording_id"],
        "split": split,
        "duration_seconds": value["duration_seconds"],
        "reference_events": [dict(event) for event in reference_events],
        "predicted_events": deepcopy(value["predicted_events"]),
        "execution_receipt": deepcopy(value["execution_receipt"]),
    }
    return validate_continuous_benchmark_rows([row])[0]


__all__ = [
    "EVENTNET_BENCHMARK_PROJECTION_METHOD_ID",
    "EVENTNET_BENCHMARK_PROJECTION_SCHEMA_VERSION",
    "join_eventnet_projection_with_public_references",
    "project_eventnet_prediction_to_benchmark",
    "validate_eventnet_benchmark_prediction_projection",
]
