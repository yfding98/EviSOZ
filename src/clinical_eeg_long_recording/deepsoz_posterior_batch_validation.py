"""Reference-free closure validation for a DeepSOZ posterior batch.

This module validates either a patient-OOF ``source_train`` batch or the
prediction-only ``source_dev`` batch used by the Stage-1 side of calibration.
Its public API deliberately accepts no seizure-reference, annotation, Excel,
clinical-text or source-evaluation path.  It validates the frozen posterior
batch, index and every referenced artifact before any caller is allowed to
open development references or fit a train-OOF fusion policy.

The returned dataclasses contain only immutable strings, numbers and tuples.
Provider receipts and posterior timelines are stored as canonical JSON and
decoded to fresh copies on request, so a downstream join cannot mutate the
validated carrier in place.  This is an engineering evidence boundary; it
does not qualify DeepSOZ for private, production or clinical use.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .canonical_detector_input_binding import (
    validate_canonical_detector_input_binding,
)
from .deepsoz_temporal_adapter import (
    DEEPSOZ_CHUNK_SECONDS,
    DEEPSOZ_FOLD_ASSIGNMENT_SCHEMA_VERSION,
    DEEPSOZ_OOF_ENSEMBLE_SCHEMA_VERSION,
    DEEPSOZ_OVERLAP_SECONDS,
    DEEPSOZ_STRIDE_SECONDS,
    PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
)
from .detector_provider_contract import validate_provider_registry


DEEPSOZ_POSTERIOR_BATCH_VALIDATION_SCHEMA_VERSION = (
    "deepsoz_posterior_batch_reference_free_validation_v2"
)
DEEPSOZ_POSTERIOR_BATCH_VALIDATION_METHOD_ID = (
    "deepsoz_split_bound_batch_index_artifact_reference_free_closure_v2"
)
DEEPSOZ_BATCH_SCHEMA_VERSION = "deepsoz_tusz_continuous_posterior_batch_v2"
DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION = (
    "deepsoz_oof_physical_binding_and_offline_time_support_v2"
)
DEEPSOZ_TIME_SUPPORT_SCHEMA_VERSION = (
    "deepsoz_offline_posterior_physical_time_support_v1"
)
DEEPSOZ_RUNTIME_SCHEMA_VERSION = "deepsoz_offline_runtime_receipt_v1"
DEEPSOZ_BATCH_RUNTIME_SCHEMA_VERSION = "deepsoz_offline_batch_runtime_receipt_v1"
DEEPSOZ_PROVIDER_ID = "deepsoz_temporal_oof_candidate_v1"
DEEPSOZ_DECISION_AVAILABILITY = "offline_after_complete_record_capture_preprocessing_and_all_held_out_fold_inference"
DEEPSOZ_TIMESTAMP_SEMANTICS = (
    "recording_relative_navigation_coordinate_not_real_time_decision_latency"
)
DEEPSOZ_PARTIAL_TAIL_POLICY = (
    "emit_unusable_coverage_marker_with_zero_sentinel_not_model_probability"
)
DEEPSOZ_652_OVERLAY_MANIFEST_SHA256 = (
    "e83ef89037bbbefd7e226bdb9b2ae103170b8768b63729a81c891fd593ec8eef"
)
DEEPSOZ_OFFICIAL_FOLD_ASSIGNMENT_RECEIPT_SHA256 = (
    "90786e77d20217fb871b99d470ef01c1da904f82e1fc3b6d0267faa91b56a7b2"
)
DEEPSOZ_MATERIALIZER_CODE_SHA256 = (
    "7213c1bbd2d2ba1d2f41785ac107ec4c501eda6ddd39dc04888f8a5f9279a7a1"
)
DEEPSOZ_ADAPTER_CODE_SHA256 = (
    "16386dd8fb8f8f11853ae087e8d865ecec113c415491a11680557a4fcc0dc9ee"
)


@dataclass(frozen=True)
class _DeepSOZSplitProfile:
    split: str
    recording_namespace: str
    official_recording_count: int
    official_patient_count: int


_SPLIT_PROFILES = {
    "source_train": _DeepSOZSplitProfile(
        split="source_train",
        recording_namespace="train",
        official_recording_count=318,
        official_patient_count=70,
    ),
    "source_dev": _DeepSOZSplitProfile(
        split="source_dev",
        recording_namespace="dev",
        official_recording_count=141,
        official_patient_count=16,
    ),
}

_ARTIFACT_FIELDS = {
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
_INDEX_FIELDS = {
    "ordinal",
    "recording_id",
    "deepsoz_patient_id",
    "model_split",
    "held_out_fold_indices",
    "posterior_artifact_id",
    "adapter_code_sha256",
    "posterior_relative_path",
    "posterior_file_sha256",
    "recording_duration_seconds",
    "timeline_window_count",
    "canonical_signal_id",
    "canonical_detector_input_binding_id",
    "canonical_detector_input_binding_receipt_sha256",
    "detector_input_channel_receipt",
    "detector_input_channel_receipt_sha256",
    "detector_imputed_channel_count",
    "posterior_time_support_receipt_id",
    "posterior_time_support_receipt_sha256",
    "posterior_runtime_receipt_id",
    "posterior_runtime_receipt_sha256",
    "current_run_runtime_receipt",
    "current_run_runtime_receipt_sha256",
    "offline_future_dependent",
    "posterior_timestamp_is_real_time_latency",
    "decision_available_at_recording_end",
    "partial_tail_present",
    "partial_tail_policy",
}
_BATCH_FIELDS = {
    "schema_version",
    "receipt_id",
    "provider_id",
    "manifest_sha256",
    "fold_assignment_receipt",
    "weights_manifest_sha256",
    "adapter_code_sha256",
    "materialized_oof_schema_version",
    "posterior_time_support_schema_version",
    "selected_split",
    "selected_recording_id",
    "max_records",
    "inventory_scope",
    "recording_count",
    "resume_requested",
    "resumed_artifact_count",
    "newly_materialized_artifact_count",
    "index_sha256",
    "batch_runtime_receipt",
    "all_selected_records_materialized",
    "records_with_detector_channel_imputation",
    "total_detector_imputed_channels",
    "published_missing_channel_policy",
    "published_missing_channel_policy_source_sha256",
    "detector_imputed_channels_clinical_evidence_eligible",
    "canonical_physical_input_bindings_verified",
    "all_posteriors_have_explicit_physical_time_support",
    "all_posteriors_offline_future_dependent",
    "posterior_timestamp_semantics",
    "decision_availability_semantics",
    "real_time_latency_metric_authorized",
    "partial_tail_policy",
    "silent_time_padding_used",
    "edf_annotations_used",
    "label_bearing_manifest_fields_retained_for_inference",
    "seizure_or_soz_labels_used_for_inference",
    "posterior_only_operating_point_not_applied",
    "production_qualified",
    "sota_claim_authorized",
}
_POSTERIOR_ROW_FIELDS = {
    "window_id",
    "start_offset_seconds",
    "stop_offset_seconds",
    "seizure_probability",
    "signal_usable",
}
_SCOPE = {
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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return str(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{context} is invalid")
    return result


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{context} must be an integer >= {minimum}")
    return value


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _loads_json(payload: bytes | str, context: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error


def _regular_file(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return path


def _split_profile(value: object) -> _DeepSOZSplitProfile:
    split = _identifier(value, "expected split")
    profile = _SPLIT_PROFILES.get(split)
    if profile is None:
        raise ValueError(
            "DeepSOZ posterior validation only permits source_train or source_dev"
        )
    return profile


def _recording_id_for_profile(
    value: object,
    *,
    profile: _DeepSOZSplitProfile,
) -> str:
    recording_id = _identifier(value, "recording_id")
    if "\\" in recording_id:
        raise ValueError("recording_id must use POSIX separators")
    path = PurePosixPath(recording_id)
    forbidden_nested_split_parts = {"train", "dev", "eval", "quarantine"}
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != profile.recording_namespace
        or path.as_posix() != recording_id
        or path.suffix != ".edf"
        or any(part in {".", "..", ""} for part in path.parts)
        or any(part.lower() in forbidden_nested_split_parts for part in path.parts[1:])
    ):
        raise ValueError(
            f"recording_id is not a safe {profile.split}/{profile.recording_namespace} EDF path"
        )
    return recording_id


def _verified_materializer_code_sha256(
    expected_materializer_code_sha256: str | None,
) -> str:
    path = _regular_file(
        Path(__file__).resolve().parents[2]
        / "scripts/materialize_deepsoz_continuous_posteriors.py",
        "DeepSOZ materializer code",
    )
    observed = _file_sha256(path.read_bytes())
    if observed != DEEPSOZ_MATERIALIZER_CODE_SHA256:
        raise ValueError(
            "DeepSOZ materializer code drifted from the audited batch-v2 "
            "implementation"
        )
    if expected_materializer_code_sha256 is not None and observed != _require_sha256(
        expected_materializer_code_sha256,
        "expected materializer code hash",
    ):
        raise ValueError("DeepSOZ materializer code differs from expectation")
    return observed


def _safe_relative_posterior(root: Path, value: object) -> Path:
    text = _identifier(value, "posterior_relative_path")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "posteriors"
        or relative.suffix != ".json"
    ):
        raise ValueError("posterior_relative_path is unsafe")
    path = _regular_file(root.joinpath(*relative.parts), "posterior artifact")
    path.resolve(strict=True).relative_to(root)
    return path


def _interval(value: object, context: str) -> tuple[float, float]:
    if type(value) is not dict or set(value) != {
        "start_offset_seconds",
        "stop_offset_seconds",
        "closure",
    }:
        raise ValueError(f"{context} interval schema drifted")
    if value["closure"] != "left_closed_right_open":
        raise ValueError(f"{context} interval closure drifted")
    start = _finite(value["start_offset_seconds"], f"{context}.start", minimum=0)
    stop = _finite(value["stop_offset_seconds"], f"{context}.stop", minimum=0)
    if stop <= start:
        raise ValueError(f"{context} interval is empty")
    return start, stop


def _validate_fold_assignment(
    payload: object,
) -> tuple[dict[str, tuple[int, ...]], str]:
    if type(payload) is not dict:
        raise TypeError("fold-assignment receipt must be an object")
    receipt = deepcopy(payload)
    if receipt.get("schema_version") != DEEPSOZ_FOLD_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError("fold-assignment schema drifted")
    digest = deepcopy(receipt)
    digest["receipt_id"] = "DEEPSOZ-FOLD-ASSIGNMENT-PENDING"
    if receipt.get("receipt_id") != "DSZFOLD-" + _sha256(digest)[:24]:
        raise ValueError("fold-assignment receipt is not content-bound")
    assignments = receipt.get("patient_fold_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("fold-assignment receipt has no patient inventory")
    lookup: dict[str, tuple[int, ...]] = {}
    normalized_rows: list[list[Any]] = []
    for index, row in enumerate(assignments):
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], list):
            raise ValueError(f"fold assignment {index} is malformed")
        patient_id = str(int(_identifier(str(row[0]), "fold patient ID")))
        folds = tuple(row[1])
        if (
            not folds
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in folds
            )
            or tuple(sorted(set(folds))) != folds
            or any(not 0 <= item < 15 for item in folds)
            or patient_id in lookup
        ):
            raise ValueError("fold-assignment patient binding is invalid")
        lookup[patient_id] = folds
        normalized_rows.append([patient_id, list(folds)])
    if receipt.get("patient_assignment_sha256") != _sha256(normalized_rows):
        raise ValueError("fold-assignment patient inventory hash drifted")
    if receipt.get("unique_patient_count") != len(lookup):
        raise ValueError("fold-assignment patient count drifted")
    return lookup, _sha256(receipt)


def _validate_runtime_receipt(
    payload: object,
    *,
    recording_id: str,
    expected_execution_modes: set[str],
) -> dict[str, Any]:
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
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("posterior runtime receipt schema drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != DEEPSOZ_RUNTIME_SCHEMA_VERSION
        or data["recording_id"] != recording_id
        or data["execution_mode"] not in expected_execution_modes
        or data["monotonic_clock"] != "python_time_perf_counter_ns"
    ):
        raise ValueError("posterior runtime receipt identity drifted")
    duration = _finite(
        data["recording_duration_seconds"], "runtime duration", minimum=1e-12
    )
    total = _finite(data["total_compute_wall_seconds"], "runtime wall time", minimum=0)
    stages = data["stage_wall_seconds"]
    if not isinstance(stages, dict) or not stages:
        raise ValueError("posterior runtime stages are empty")
    for key, value in stages.items():
        _identifier(key, "runtime stage")
        _finite(value, "runtime stage wall time", minimum=0)
    folds = data["held_out_fold_wall_seconds"]
    if not isinstance(folds, list):
        raise ValueError("posterior runtime fold timings are invalid")
    observed_folds: list[int] = []
    for row in folds:
        if type(row) is not dict or set(row) != {
            "fold_index",
            "adapter_load_and_inference_wall_seconds",
        }:
            raise ValueError("posterior runtime fold row drifted")
        observed_folds.append(_integer(row["fold_index"], "runtime fold"))
        _finite(
            row["adapter_load_and_inference_wall_seconds"], "fold wall time", minimum=0
        )
    if observed_folds != sorted(set(observed_folds)) or any(
        item >= 15 for item in observed_folds
    ):
        raise ValueError("posterior runtime folds are invalid")
    if data["execution_mode"] == "new_oof_inference" and not folds:
        raise ValueError("original posterior runtime lacks fold inference")
    if data["execution_mode"] == "resume_validation_only" and folds:
        raise ValueError("resume runtime improperly claims fold inference")
    if abs(float(data["compute_real_time_factor"]) - total / duration) > 1e-12:
        raise ValueError("posterior runtime RTF drifted")
    throughput = duration / total if total > 0 else None
    if data["recording_seconds_per_compute_wall_second"] != throughput:
        raise ValueError("posterior runtime throughput drifted")
    if (
        data["runtime_semantics"]
        != "offline_batch_compute_wall_time_not_signal_timestamp_and_not_real_time_decision_latency"
        or data["decision_availability_semantics"] != DEEPSOZ_DECISION_AVAILABILITY
        or data["real_time_latency_metric_authorized"] is not False
        or data["onset_latency_claim_authorized"] is not False
    ):
        raise ValueError("posterior runtime overclaims causal latency")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _sha256(digest):
        raise ValueError("posterior runtime receipt hash drifted")
    id_source = deepcopy(digest)
    id_source["receipt_id"] = "DEEPSOZ-RUNTIME-PENDING"
    if data["receipt_id"] != "DSZRUNTIME-" + _sha256(id_source)[:24]:
        raise ValueError("posterior runtime receipt ID drifted")
    return data


def _validate_timeline(value: object, *, duration: float) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("posterior timeline is empty")
    rows: list[dict[str, Any]] = []
    cursor = 0.0
    unusable_seen = False
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != _POSTERIOR_ROW_FIELDS:
            raise ValueError(f"posterior row {index} schema drifted")
        row = deepcopy(raw)
        start = _finite(row["start_offset_seconds"], "posterior start", minimum=0)
        stop = _finite(row["stop_offset_seconds"], "posterior stop", minimum=0)
        probability = _finite(
            row["seizure_probability"], "posterior probability", minimum=0
        )
        if (
            probability > 1
            or abs(start - cursor) > 1e-9
            or stop <= start
            or stop - start > 1 + 1e-9
        ):
            raise ValueError("posterior timeline physical grid drifted")
        if row["window_id"] != f"DEEPSOZ-OOF-SEC-{index:07d}":
            raise ValueError("posterior window ID drifted")
        if type(row["signal_usable"]) is not bool:
            raise TypeError("posterior signal_usable must be boolean")
        if not row["signal_usable"]:
            unusable_seen = True
            if probability != 0 or index != len(value) - 1:
                raise ValueError("unusable posterior may only be the zero partial tail")
        elif unusable_seen or abs(stop - start - 1.0) > 1e-9:
            raise ValueError("usable posterior rows must be contiguous full seconds")
        cursor = stop
        rows.append(row)
    if abs(cursor - duration) > 1e-9:
        raise ValueError("posterior timeline does not cover the recording")
    return rows


def _validate_time_support(
    payload: object,
    *,
    recording_id: str,
    duration: float,
    timeline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "receipt_id",
        "provider_id",
        "recording_id",
        "recording_duration_seconds",
        "target_clock",
        "timestamp_semantics",
        "offline_future_dependent",
        "causal_or_streaming_decision",
        "real_time_latency_metric_authorized",
        "preprocessing_dependency",
        "model_context_policy",
        "model_contexts",
        "padding_and_tail_semantics",
        "posterior_support_rows",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("posterior time-support receipt schema drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != DEEPSOZ_TIME_SUPPORT_SCHEMA_VERSION
        or data["provider_id"] != DEEPSOZ_PROVIDER_ID
        or data["recording_id"] != recording_id
        or float(data["recording_duration_seconds"]) != duration
        or data["target_clock"] != "recording_relative_physical_seconds"
        or data["timestamp_semantics"] != DEEPSOZ_TIMESTAMP_SEMANTICS
        or data["offline_future_dependent"] is not True
        or data["causal_or_streaming_decision"] is not False
        or data["real_time_latency_metric_authorized"] is not False
    ):
        raise ValueError("posterior time-support identity or permissions drifted")
    dependency = data["preprocessing_dependency"]
    expected_dependency_flags = {
        "whole_record_fft_resampling": True,
        "whole_record_zero_phase_filtering": True,
        "whole_record_per_channel_clipping_statistics": True,
        "whole_modeled_record_global_normalization": True,
        "finite_causal_receptive_field": False,
        "dependency_interpretation": "conservative_full_record_physical_dependency_due_to_offline_global_and_zero_phase_operations",
    }
    if not isinstance(dependency, dict) or any(
        dependency.get(k) != v for k, v in expected_dependency_flags.items()
    ):
        raise ValueError("posterior preprocessing dependency drifted")
    if _interval(dependency.get("support_interval"), "preprocessing support") != (
        0.0,
        duration,
    ):
        raise ValueError("posterior preprocessing does not bind the full recording")
    policy = data["model_context_policy"]
    expected_policy = {
        "chunk_seconds": DEEPSOZ_CHUNK_SECONDS,
        "overlap_seconds": DEEPSOZ_OVERLAP_SECONDS,
        "stride_seconds": DEEPSOZ_STRIDE_SECONDS,
        "bidirectional_chunk_context": True,
        "overlap_fusion": "linear_edge_ramp_weighted_probability_mean",
        "all_held_out_folds_share_context_plan": True,
    }
    if policy != expected_policy:
        raise ValueError("posterior model-context policy drifted")
    contexts = data["model_contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("posterior model contexts are empty")
    full_seconds = sum(row["signal_usable"] is True for row in timeline)
    expected_contexts: list[dict[str, Any]] = []
    context_start = 0
    while context_start < full_seconds:
        context_stop = min(full_seconds, context_start + DEEPSOZ_CHUNK_SECONDS)
        expected_contexts.append(
            {
                "model_context_id": (
                    f"DEEPSOZ-CONTEXT-{len(expected_contexts) + 1:04d}"
                ),
                "support_interval": {
                    "start_offset_seconds": float(context_start),
                    "stop_offset_seconds": float(context_stop),
                    "closure": "left_closed_right_open",
                },
                "modeled_seconds": context_stop - context_start,
                "left_overlap_seconds": (
                    0
                    if context_start == 0
                    else min(
                        DEEPSOZ_OVERLAP_SECONDS,
                        context_stop - context_start,
                    )
                ),
                "right_overlap_seconds": (
                    0
                    if context_stop == full_seconds
                    else min(
                        DEEPSOZ_OVERLAP_SECONDS,
                        context_stop - context_start,
                    )
                ),
                "bidirectional_temporal_context": True,
                "temporal_padding_used": False,
            }
        )
        if context_stop == full_seconds:
            break
        context_start += DEEPSOZ_STRIDE_SECONDS
    if contexts != expected_contexts:
        raise ValueError("posterior model context plan is not replayable")
    context_intervals: dict[str, tuple[float, float]] = {}
    for index, context in enumerate(contexts, start=1):
        if type(context) is not dict or set(context) != {
            "model_context_id",
            "support_interval",
            "modeled_seconds",
            "left_overlap_seconds",
            "right_overlap_seconds",
            "bidirectional_temporal_context",
            "temporal_padding_used",
        }:
            raise ValueError("posterior model context schema drifted")
        context_id = f"DEEPSOZ-CONTEXT-{index:04d}"
        if context["model_context_id"] != context_id:
            raise ValueError("posterior model context ID drifted")
        span = _interval(context["support_interval"], "model context")
        if (
            span[1] > duration + 1e-9
            or abs(float(context["modeled_seconds"]) - (span[1] - span[0])) > 1e-9
        ):
            raise ValueError("posterior model context support drifted")
        if (
            context["bidirectional_temporal_context"] is not True
            or context["temporal_padding_used"] is not False
        ):
            raise ValueError("posterior model context permissions drifted")
        context_intervals[context_id] = span
    support_rows = data["posterior_support_rows"]
    if not isinstance(support_rows, list) or len(support_rows) != len(timeline):
        raise ValueError("posterior support-row inventory drifted")
    full_support = (0.0, duration)
    for row, support in zip(timeline, support_rows):
        if (
            type(support) is not dict
            or set(support)
            != {
                "window_id",
                "target_interval",
                "actual_preprocessing_support_interval",
                "actual_model_context_ids",
                "effective_signal_dependency_support_interval",
                "future_lookahead_from_target_stop_seconds",
                "maximum_bidirectional_model_future_context_seconds",
                "decision_available_at_recording_offset_seconds",
                "decision_availability_semantics",
                "posterior_value_semantics",
                "partial_tail_coverage_marker",
                "temporal_padding_used",
            }
            or support.get("window_id") != row["window_id"]
        ):
            raise ValueError("posterior support row binding drifted")
        target = _interval(support.get("target_interval"), "posterior target")
        expected_target = (
            float(row["start_offset_seconds"]),
            float(row["stop_offset_seconds"]),
        )
        if (
            target != expected_target
            or _interval(
                support.get("actual_preprocessing_support_interval"),
                "actual preprocessing support",
            )
            != full_support
        ):
            raise ValueError("posterior support physical interval drifted")
        if (
            support.get("decision_available_at_recording_offset_seconds") != duration
            or support.get("decision_availability_semantics")
            != DEEPSOZ_DECISION_AVAILABILITY
            or support.get("temporal_padding_used") is not False
        ):
            raise ValueError("posterior decision availability drifted")
        usable = row["signal_usable"] is True
        if usable:
            context_ids = support.get("actual_model_context_ids")
            if not isinstance(context_ids, list) or not context_ids:
                raise ValueError("usable posterior lacks model context")
            spans = [context_intervals.get(str(item)) for item in context_ids]
            if any(
                span is None or span[0] > target[0] + 1e-9 or span[1] < target[1] - 1e-9
                for span in spans
            ):
                raise ValueError("posterior model context does not cover target")
            expected_model_future = max(
                float(span[1]) - target[1] for span in spans if span is not None
            )
            if (
                _interval(
                    support.get("effective_signal_dependency_support_interval"),
                    "effective signal support",
                )
                != full_support
            ):
                raise ValueError(
                    "usable posterior does not bind full-record dependency"
                )
            if (
                support.get("future_lookahead_from_target_stop_seconds")
                != duration - target[1]
                or support.get("maximum_bidirectional_model_future_context_seconds")
                != expected_model_future
                or support.get("posterior_value_semantics")
                != "offline_model_probability_for_navigation_only"
                or support.get("partial_tail_coverage_marker") is not False
            ):
                raise ValueError("usable posterior offline semantics drifted")
        else:
            if (
                support.get("actual_model_context_ids") != []
                or support.get("effective_signal_dependency_support_interval")
                is not None
                or support.get("future_lookahead_from_target_stop_seconds") is not None
                or support.get("maximum_bidirectional_model_future_context_seconds")
                is not None
                or support.get("posterior_value_semantics")
                != "zero_sentinel_for_unmodeled_partial_tail_not_model_probability"
                or support.get("partial_tail_coverage_marker") is not True
            ):
                raise ValueError("partial-tail posterior semantics drifted")
    tail = data["padding_and_tail_semantics"]
    partial_tail = full_seconds < len(timeline)
    if (
        not isinstance(tail, dict)
        or set(tail)
        != {
            "temporal_padding_used",
            "silent_time_padding_used",
            "missing_channel_zero_fill_is_spatial_imputation_not_time_padding",
            "modeled_full_second_count",
            "partial_tail_present",
            "partial_tail_interval",
            "partial_tail_policy",
            "partial_tail_zero_is_negative_evidence",
        }
        or tail.get("temporal_padding_used") is not False
        or tail.get("silent_time_padding_used") is not False
        or tail.get("missing_channel_zero_fill_is_spatial_imputation_not_time_padding")
        is not True
        or tail.get("modeled_full_second_count") != full_seconds
        or tail.get("partial_tail_present") is not partial_tail
        or tail.get("partial_tail_policy") != DEEPSOZ_PARTIAL_TAIL_POLICY
        or tail.get("partial_tail_zero_is_negative_evidence") is not False
    ):
        raise ValueError("posterior tail semantics drifted")
    if partial_tail:
        if _interval(tail.get("partial_tail_interval"), "partial tail") != (
            float(full_seconds),
            duration,
        ):
            raise ValueError("posterior partial-tail interval drifted")
    elif tail.get("partial_tail_interval") is not None:
        raise ValueError("posterior unexpectedly declares a partial-tail interval")
    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotations_used": False,
        "spreadsheet_used": False,
        "doctor_labels_used": False,
        "reference_event_times_used": False,
        "navigation_only": True,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("posterior time-support scope drifted")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _sha256(digest):
        raise ValueError("posterior time-support hash drifted")
    id_source = deepcopy(digest)
    id_source["receipt_id"] = "DEEPSOZ-TIME-SUPPORT-PENDING"
    if data["receipt_id"] != "DSZTIME-" + _sha256(id_source)[:24]:
        raise ValueError("posterior time-support ID drifted")
    return data


def _validate_batch_runtime(
    payload: object, *, total_duration: float
) -> dict[str, Any]:
    required = {
        "schema_version",
        "receipt_id",
        "monotonic_clock",
        "batch_compute_wall_seconds",
        "total_recording_duration_seconds",
        "batch_compute_real_time_factor",
        "recording_seconds_per_batch_compute_wall_second",
        "runtime_semantics",
        "real_time_latency_metric_authorized",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("batch runtime receipt schema drifted")
    data = deepcopy(payload)
    wall = _finite(data["batch_compute_wall_seconds"], "batch wall time", minimum=0)
    if (
        data["schema_version"] != DEEPSOZ_BATCH_RUNTIME_SCHEMA_VERSION
        or data["monotonic_clock"] != "python_time_perf_counter_ns"
        or abs(float(data["total_recording_duration_seconds"]) - total_duration) > 1e-9
    ):
        raise ValueError("batch runtime identity drifted")
    if (
        abs(float(data["batch_compute_real_time_factor"]) - wall / total_duration)
        > 1e-12
    ):
        raise ValueError("batch runtime RTF drifted")
    throughput = total_duration / wall if wall > 0 else None
    if (
        data["recording_seconds_per_batch_compute_wall_second"] != throughput
        or data["runtime_semantics"]
        != "offline_batch_materialization_or_resume_validation_wall_time_not_real_time_decision_latency"
        or data["real_time_latency_metric_authorized"] is not False
    ):
        raise ValueError("batch runtime semantics drifted")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _sha256(digest):
        raise ValueError("batch runtime hash drifted")
    id_source = deepcopy(digest)
    id_source["receipt_id"] = "DEEPSOZ-BATCH-RUNTIME-PENDING"
    if data["receipt_id"] != "DSZBATCHRUNTIME-" + _sha256(id_source)[:24]:
        raise ValueError("batch runtime ID drifted")
    return data


def _normalized_expected(
    values: Iterable[str] | None, context: str
) -> list[str] | None:
    if values is None:
        return None
    raw = [_identifier(str(value), context) for value in values]
    if len(raw) != len(set(raw)):
        raise ValueError(f"{context} inventory contains duplicates")
    result = sorted(raw)
    if not result:
        raise ValueError(f"{context} inventory is empty")
    return result


@dataclass(frozen=True)
class ValidatedDeepSOZPosteriorRecording:
    patient_id: str
    recording_id: str
    duration_seconds: float
    canonical_source_signal_sha256: str
    posterior_artifact_id: str
    posterior_file_sha256: str
    record_binding_sha256: str
    provider_receipt_json: str
    posterior_timeline_json: str

    def provider_receipt(self) -> dict[str, Any]:
        return _loads_json(self.provider_receipt_json, "provider receipt")

    def posterior_timeline(self) -> list[dict[str, Any]]:
        return _loads_json(self.posterior_timeline_json, "posterior timeline")


@dataclass(frozen=True)
class ValidatedDeepSOZPosteriorBatch:
    batch_root: str
    recordings: tuple[ValidatedDeepSOZPosteriorRecording, ...]
    validation_receipt_json: str

    def validation_receipt(self) -> dict[str, Any]:
        return _loads_json(self.validation_receipt_json, "batch validation receipt")


def revalidate_deepsoz_posterior_batch_without_references(
    value: object,
) -> ValidatedDeepSOZPosteriorBatch:
    """Revalidate the sealed in-memory carrier without opening any file.

    Stage-2 joins must call this function and require the exact frozen type;
    accepting a duck-typed object with a convenient ``validated=True`` flag
    would defeat the prediction-before-reference boundary.
    """

    if type(value) is not ValidatedDeepSOZPosteriorBatch:
        raise TypeError("posterior consumer requires the sealed posterior batch type")
    receipt = value.validation_receipt()
    required_receipt = {
        "schema_version",
        "validation_id",
        "method_id",
        "batch_root",
        "provider_id",
        "selected_split",
        "recording_namespace",
        "inventory_scope",
        "inventory_completeness_verified",
        "official_manifest_inventory_counts_verified",
        "recording_count",
        "patient_count",
        "recording_ids_sha256",
        "patient_ids_sha256",
        "manifest_sha256",
        "batch_receipt_file_sha256",
        "posterior_index_file_sha256",
        "posterior_artifact_inventory_sha256",
        "fold_assignment_receipt_sha256",
        "weights_manifest_sha256",
        "adapter_code_sha256",
        "materializer_code_sha256",
        "provider_receipt_sha256",
        "total_recording_duration_seconds",
        "original_inference_wall_seconds",
        "all_original_runtime_receipts_verified",
        "all_canonical_physical_bindings_verified",
        "all_time_support_receipts_verified",
        "all_posterior_artifact_content_ids_verified",
        "reference_access",
        "research_only",
        "production_qualified",
        "sota_claim_authorized",
        "receipt_sha256",
    }
    if type(receipt) is not dict or set(receipt) != required_receipt:
        raise ValueError("sealed posterior validation receipt schema drifted")
    if value.validation_receipt_json != _canonical_json(receipt):
        raise ValueError("sealed posterior validation JSON is not canonical")
    profile = _split_profile(receipt.get("selected_split"))
    if (
        receipt["schema_version"] != DEEPSOZ_POSTERIOR_BATCH_VALIDATION_SCHEMA_VERSION
        or receipt["method_id"] != DEEPSOZ_POSTERIOR_BATCH_VALIDATION_METHOD_ID
        or receipt["batch_root"] != value.batch_root
        or receipt["provider_id"] != DEEPSOZ_PROVIDER_ID
        or receipt["recording_namespace"] != profile.recording_namespace
        or receipt["inventory_scope"] != "full_selected_split"
        or receipt["inventory_completeness_verified"] is not True
        or receipt["research_only"] is not True
        or receipt["production_qualified"] is not False
        or receipt["sota_claim_authorized"] is not False
    ):
        raise ValueError("sealed posterior validation scope drifted")
    manifest_sha256 = _require_sha256(
        receipt["manifest_sha256"],
        "sealed manifest hash",
    )
    fold_assignment_sha256 = _require_sha256(
        receipt["fold_assignment_receipt_sha256"],
        "sealed fold-assignment hash",
    )
    if (
        receipt["weights_manifest_sha256"] != PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256
        or receipt["adapter_code_sha256"] != DEEPSOZ_ADAPTER_CODE_SHA256
        or receipt["materializer_code_sha256"] != DEEPSOZ_MATERIALIZER_CODE_SHA256
    ):
        raise ValueError("sealed DeepSOZ code or weight provenance drifted")
    official_profile = manifest_sha256 == DEEPSOZ_652_OVERLAY_MANIFEST_SHA256
    if receipt["official_manifest_inventory_counts_verified"] is not official_profile:
        raise ValueError("sealed official-manifest qualification state drifted")
    if official_profile and fold_assignment_sha256 != (
        DEEPSOZ_OFFICIAL_FOLD_ASSIGNMENT_RECEIPT_SHA256
    ):
        raise ValueError("sealed official fold assignment drifted")
    expected_reference_access = {
        "reference_path_argument_accepted": False,
        "reference_files_opened": 0,
        "edf_annotations_opened": 0,
        "excel_files_opened": 0,
        "clinical_text_opened": 0,
        "source_eval_opened": 0,
    }
    if receipt["reference_access"] != expected_reference_access:
        raise ValueError("sealed posterior validation accessed a reference source")
    for field in (
        "all_original_runtime_receipts_verified",
        "all_canonical_physical_bindings_verified",
        "all_time_support_receipts_verified",
        "all_posterior_artifact_content_ids_verified",
    ):
        if receipt[field] is not True:
            raise ValueError("sealed posterior validation lost a closure gate")
    digest = deepcopy(receipt)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt["receipt_sha256"] != _sha256(digest):
        raise ValueError("sealed posterior validation hash drifted")
    id_source = deepcopy(digest)
    id_source["validation_id"] = "DEEPSOZ-BATCH-VALIDATION-PENDING"
    if receipt["validation_id"] != "DSZBATCHVALID-" + _sha256(id_source)[:24]:
        raise ValueError("sealed posterior validation ID drifted")

    if not value.recordings:
        raise ValueError("sealed posterior batch contains no recordings")
    recording_ids: list[str] = []
    patient_ids: set[str] = set()
    artifact_rows: list[list[str]] = []
    total_duration = 0.0
    provider_sha256: str | None = None
    seen_artifacts: set[str] = set()
    for row in value.recordings:
        if type(row) is not ValidatedDeepSOZPosteriorRecording:
            raise TypeError("sealed posterior batch contains an unsealed recording")
        patient_id = str(int(_identifier(row.patient_id, "sealed patient_id")))
        recording_id = _recording_id_for_profile(
            row.recording_id,
            profile=profile,
        )
        duration = _finite(row.duration_seconds, "sealed duration", minimum=1e-12)
        canonical_sha256 = _require_sha256(
            row.canonical_source_signal_sha256,
            "sealed canonical source signal hash",
        )
        file_sha256 = _require_sha256(
            row.posterior_file_sha256, "sealed posterior file hash"
        )
        artifact_id = _identifier(
            row.posterior_artifact_id, "sealed posterior artifact ID"
        )
        if artifact_id in seen_artifacts:
            raise ValueError("sealed posterior artifact is duplicated")
        seen_artifacts.add(artifact_id)
        provider = row.provider_receipt()
        provider_required = {
            "provider_id",
            "model_family",
            "checkpoint_sha256",
            "code_sha256",
            "training_corpus",
            "posterior_calibration_status",
            "deployment_qualification_status",
            "annotations_used_for_current_recording",
            "labels_used_for_current_recording",
        }
        if (
            type(provider) is not dict
            or set(provider) != provider_required
            or provider["provider_id"] != DEEPSOZ_PROVIDER_ID
            or provider["checkpoint_sha256"] != receipt["weights_manifest_sha256"]
            or provider["code_sha256"] != receipt["adapter_code_sha256"]
            or provider["deployment_qualification_status"] != "research_candidate"
            or provider["annotations_used_for_current_recording"] is not False
            or provider["labels_used_for_current_recording"] is not False
        ):
            raise ValueError("sealed detector provider receipt drifted")
        if row.provider_receipt_json != _canonical_json(provider):
            raise ValueError("sealed provider receipt JSON is not canonical")
        current_provider_sha256 = _sha256(provider)
        if provider_sha256 is None:
            provider_sha256 = current_provider_sha256
        elif provider_sha256 != current_provider_sha256:
            raise ValueError("sealed batch mixes detector provider receipts")
        timeline = row.posterior_timeline()
        timeline = _validate_timeline(timeline, duration=duration)
        if row.posterior_timeline_json != _canonical_json(timeline):
            raise ValueError("sealed posterior timeline JSON is not canonical")
        record_body = {
            "patient_id": patient_id,
            "recording_id": recording_id,
            "duration_seconds": duration,
            "canonical_source_signal_sha256": canonical_sha256,
            "posterior_artifact_id": artifact_id,
            "posterior_file_sha256": file_sha256,
            "provider_receipt": provider,
            "posterior_timeline_sha256": hashlib.sha256(
                row.posterior_timeline_json.encode("utf-8")
            ).hexdigest(),
        }
        if row.record_binding_sha256 != _sha256(record_body):
            raise ValueError("sealed posterior recording binding drifted")
        recording_ids.append(recording_id)
        patient_ids.add(patient_id)
        artifact_rows.append([recording_id, file_sha256, artifact_id])
        total_duration += duration
    if recording_ids != sorted(set(recording_ids)):
        # The validator sorts by patient then recording.  Require uniqueness,
        # but do not rely on lexical recording order across patient IDs.
        if len(recording_ids) != len(set(recording_ids)):
            raise ValueError("sealed posterior recording is duplicated")
    sorted_recording_ids = sorted(recording_ids)
    sorted_patient_ids = sorted(patient_ids)
    if (
        receipt["recording_count"] != len(value.recordings)
        or receipt["patient_count"] != len(patient_ids)
        or receipt["recording_ids_sha256"] != _sha256(sorted_recording_ids)
        or receipt["patient_ids_sha256"] != _sha256(sorted_patient_ids)
        or receipt["posterior_artifact_inventory_sha256"]
        != _sha256(sorted(artifact_rows))
        or receipt["provider_receipt_sha256"] != provider_sha256
        or abs(float(receipt["total_recording_duration_seconds"]) - total_duration)
        > 1e-9
    ):
        raise ValueError("sealed posterior batch inventory binding drifted")
    if official_profile and (
        len(value.recordings) != profile.official_recording_count
        or len(patient_ids) != profile.official_patient_count
    ):
        raise ValueError("sealed official split inventory cardinality drifted")
    return value


def validate_deepsoz_posterior_batch_without_references(
    batch_root: str | Path,
    *,
    expected_split: str = "source_dev",
    expected_manifest_sha256: str | None = None,
    expected_recording_ids: Iterable[str] | None = None,
    expected_patient_ids: Iterable[str] | None = None,
    expected_materializer_code_sha256: str | None = None,
    require_complete_inventory: bool = True,
    provider_registry_path: str | Path | None = None,
) -> ValidatedDeepSOZPosteriorBatch:
    """Validate a complete train-OOF or development batch without references.

    ``expected_split`` is a closed enum: ``source_train`` binds recording IDs
    to ``train/`` while ``source_dev`` binds them to ``dev/``.  No source-eval
    profile exists.  The function has no argument through which a seizure
    reference, EDF annotation, spreadsheet or clinical text can be supplied.
    """

    profile = _split_profile(expected_split)
    materializer_code_sha256 = _verified_materializer_code_sha256(
        expected_materializer_code_sha256
    )
    raw_root = Path(batch_root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("posterior batch root must be a regular directory")
    root = raw_root.resolve(strict=True)
    root_entries = {path.name: path for path in root.iterdir()}
    if set(root_entries) != {
        "batch_receipt.json",
        "posterior_index.jsonl",
        "posteriors",
    }:
        raise ValueError("posterior batch root contains an unexpected or missing entry")
    posterior_root = root_entries["posteriors"]
    if posterior_root.is_symlink() or not posterior_root.is_dir():
        raise ValueError("posterior artifact inventory must be a regular directory")
    batch_path = _regular_file(root / "batch_receipt.json", "batch receipt")
    index_path = _regular_file(root / "posterior_index.jsonl", "posterior index")
    batch_bytes = batch_path.read_bytes()
    index_bytes = index_path.read_bytes()
    batch = _loads_json(batch_bytes, "batch receipt")
    if type(batch) is not dict or set(batch) != _BATCH_FIELDS:
        raise ValueError("posterior batch receipt schema drifted")
    if (
        batch["schema_version"] != DEEPSOZ_BATCH_SCHEMA_VERSION
        or batch["provider_id"] != DEEPSOZ_PROVIDER_ID
        or batch["selected_split"] != profile.split
        or batch["selected_recording_id"] is not None
        or batch["max_records"] != 0
        or batch["inventory_scope"] != "full_selected_split"
        or batch["all_selected_records_materialized"] is not True
        or batch["materialized_oof_schema_version"]
        != DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION
        or batch["posterior_time_support_schema_version"]
        != DEEPSOZ_TIME_SUPPORT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"posterior batch is not a completed full {profile.split} split"
        )
    required_false = (
        "detector_imputed_channels_clinical_evidence_eligible",
        "real_time_latency_metric_authorized",
        "silent_time_padding_used",
        "edf_annotations_used",
        "label_bearing_manifest_fields_retained_for_inference",
        "seizure_or_soz_labels_used_for_inference",
        "production_qualified",
        "sota_claim_authorized",
    )
    if (
        any(batch[field] is not False for field in required_false)
        or batch["posterior_only_operating_point_not_applied"] is not True
        or batch["all_posteriors_have_explicit_physical_time_support"] is not True
        or batch["all_posteriors_offline_future_dependent"] is not True
    ):
        raise ValueError(
            "posterior batch inference firewall or promotion state drifted"
        )
    if (
        batch["posterior_timestamp_semantics"] != DEEPSOZ_TIMESTAMP_SEMANTICS
        or batch["decision_availability_semantics"] != DEEPSOZ_DECISION_AVAILABILITY
        or batch["partial_tail_policy"] != DEEPSOZ_PARTIAL_TAIL_POLICY
    ):
        raise ValueError("posterior batch time semantics drifted")
    manifest_sha256 = _require_sha256(batch["manifest_sha256"], "batch manifest hash")
    if expected_manifest_sha256 is not None and manifest_sha256 != _require_sha256(
        expected_manifest_sha256, "expected manifest hash"
    ):
        raise ValueError("posterior batch manifest hash differs from expectation")
    official_manifest_profile = manifest_sha256 == DEEPSOZ_652_OVERLAY_MANIFEST_SHA256
    recording_count = _integer(
        batch["recording_count"], "batch recording count", minimum=1
    )
    if (
        official_manifest_profile
        and recording_count != profile.official_recording_count
    ):
        raise ValueError(
            f"official {profile.split} inventory must contain exactly "
            f"{profile.official_recording_count} recordings"
        )
    if batch["index_sha256"] != _file_sha256(index_bytes):
        raise ValueError("posterior index file hash drifted")
    batch_id_source = deepcopy(batch)
    batch_id_source["receipt_id"] = "DEEPSOZ-BATCH-PENDING"
    if batch["receipt_id"] != "DSZBATCH-" + _sha256(batch_id_source)[:24]:
        raise ValueError("posterior batch receipt ID drifted")

    fold_lookup, fold_receipt_sha256 = _validate_fold_assignment(
        batch["fold_assignment_receipt"]
    )
    if official_manifest_profile and fold_receipt_sha256 != (
        DEEPSOZ_OFFICIAL_FOLD_ASSIGNMENT_RECEIPT_SHA256
    ):
        raise ValueError("official DeepSOZ fold assignment receipt drifted")
    registry_path = (
        Path(provider_registry_path)
        if provider_registry_path is not None
        else Path(__file__).resolve().parents[2]
        / "configs/continuous_detector_providers_v1.json"
    )
    registry_payload = _loads_json(
        _regular_file(registry_path, "provider registry").read_bytes(),
        "provider registry",
    )
    registry = validate_provider_registry(registry_payload)
    definitions = {
        row["execution_definition"]["provider_id"]: row["execution_definition"]
        for row in registry["providers"]
    }
    if DEEPSOZ_PROVIDER_ID not in definitions:
        raise ValueError("provider registry lacks the DeepSOZ provider")
    definition = definitions[DEEPSOZ_PROVIDER_ID]
    adapter_code_sha256 = _file_sha256(
        _regular_file(
            Path(__file__).resolve().with_name("deepsoz_temporal_adapter.py"),
            "DeepSOZ temporal adapter code",
        ).read_bytes()
    )
    if (
        definition["weights_manifest_sha256"] != batch["weights_manifest_sha256"]
        or definition["adapter_code_sha256"] != batch["adapter_code_sha256"]
        or batch["weights_manifest_sha256"] != PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256
        or batch["adapter_code_sha256"] != DEEPSOZ_ADAPTER_CODE_SHA256
        or adapter_code_sha256 != DEEPSOZ_ADAPTER_CODE_SHA256
    ):
        raise ValueError(
            "posterior batch code or weights drifted from provider registry"
        )
    provider_receipt = {
        "provider_id": DEEPSOZ_PROVIDER_ID,
        "model_family": definition["model_family"],
        "checkpoint_sha256": definition["weights_manifest_sha256"],
        "code_sha256": definition["adapter_code_sha256"],
        "training_corpus": definition["training_corpus"],
        "posterior_calibration_status": definition["posterior_calibration_status"],
        "deployment_qualification_status": "research_candidate",
        "annotations_used_for_current_recording": False,
        "labels_used_for_current_recording": False,
    }
    provider_receipt_json = _canonical_json(provider_receipt)

    lines = [line for line in index_bytes.splitlines() if line.strip()]
    if len(lines) != recording_count:
        raise ValueError("posterior index line count differs from batch receipt")
    if (
        _integer(batch["resumed_artifact_count"], "resumed count")
        + _integer(batch["newly_materialized_artifact_count"], "new count")
        != recording_count
        or batch["canonical_physical_input_bindings_verified"] != recording_count
    ):
        raise ValueError("posterior batch completion counters drifted")

    records: list[ValidatedDeepSOZPosteriorRecording] = []
    seen_recordings: set[str] = set()
    seen_artifacts: set[str] = set()
    seen_paths: set[str] = set()
    imputed_recordings = 0
    imputed_channels = 0
    total_duration = 0.0
    original_runtime_total = 0.0
    artifact_hash_rows: list[list[str]] = []
    for ordinal, line in enumerate(lines, start=1):
        index = _loads_json(line, f"posterior index line {ordinal}")
        if (
            type(index) is not dict
            or set(index) != _INDEX_FIELDS
            or index["ordinal"] != ordinal
        ):
            raise ValueError("posterior index schema or ordinal drifted")
        recording_id = _recording_id_for_profile(
            index["recording_id"],
            profile=profile,
        )
        if recording_id in seen_recordings:
            raise ValueError("posterior index contains a duplicate recording")
        seen_recordings.add(recording_id)
        patient_id = str(
            int(_identifier(str(index["deepsoz_patient_id"]), "patient_id"))
        )
        if index["model_split"] != profile.split:
            raise ValueError(
                "posterior index split disagrees with the requested profile"
            )
        folds = tuple(index["held_out_fold_indices"])
        if fold_lookup.get(patient_id) != folds:
            raise ValueError(
                "posterior index folds disagree with official patient assignment"
            )
        relative_text = _identifier(index["posterior_relative_path"], "posterior path")
        if relative_text in seen_paths:
            raise ValueError("posterior index path is duplicated")
        seen_paths.add(relative_text)
        artifact_path = _safe_relative_posterior(root, relative_text)
        artifact_bytes = artifact_path.read_bytes()
        file_sha256 = _file_sha256(artifact_bytes)
        if file_sha256 != index["posterior_file_sha256"]:
            raise ValueError("posterior artifact file hash drifted")
        artifact = _loads_json(artifact_bytes, "posterior artifact")
        if type(artifact) is not dict or set(artifact) != _ARTIFACT_FIELDS:
            raise ValueError("posterior artifact schema drifted")
        artifact_id = _identifier(
            artifact["posterior_artifact_id"], "posterior artifact ID"
        )
        if (
            artifact_id in seen_artifacts
            or artifact_id != index["posterior_artifact_id"]
        ):
            raise ValueError("posterior artifact ID is duplicate or index-mismatched")
        seen_artifacts.add(artifact_id)
        if (
            artifact["schema_version"] != DEEPSOZ_OOF_ENSEMBLE_SCHEMA_VERSION
            or artifact["materialization_schema_version"]
            != DEEPSOZ_MATERIALIZED_OOF_SCHEMA_VERSION
            or artifact["provider_id"] != DEEPSOZ_PROVIDER_ID
            or artifact["recording_id"] != recording_id
            or artifact["deepsoz_patient_id"] != patient_id
            or tuple(artifact["held_out_fold_indices"]) != folds
            or artifact["held_out_repeat_count"] != len(folds)
        ):
            raise ValueError("posterior artifact identity drifted")
        if (
            artifact["fold_assignment_receipt_sha256"] != fold_receipt_sha256
            or artifact["weights_manifest_sha256"] != batch["weights_manifest_sha256"]
            or artifact["adapter_code_sha256"] != batch["adapter_code_sha256"]
            or index["adapter_code_sha256"] != batch["adapter_code_sha256"]
            or artifact["scope_receipt"] != _SCOPE
        ):
            raise ValueError("posterior artifact provenance or scope drifted")
        artifact_id_source = deepcopy(artifact)
        artifact_id_source["posterior_artifact_id"] = "DEEPSOZ-OOF-POSTERIOR-PENDING"
        if artifact_id != "DSZOOF-" + _sha256(artifact_id_source)[:24]:
            raise ValueError("posterior artifact content binding failed")
        duration = _finite(
            artifact["recording_duration_seconds"], "recording duration", minimum=1e-12
        )
        if float(index["recording_duration_seconds"]) != duration:
            raise ValueError("posterior duration disagrees with index")
        timeline = _validate_timeline(artifact["posterior_timeline"], duration=duration)
        if index["timeline_window_count"] != len(timeline):
            raise ValueError("posterior timeline count disagrees with index")
        binding = validate_canonical_detector_input_binding(
            artifact["canonical_detector_input_binding"]
        )
        if (
            binding["provider_id"] != DEEPSOZ_PROVIDER_ID
            or binding["binding_id"] != index["canonical_detector_input_binding_id"]
            or binding["receipt_sha256"]
            != index["canonical_detector_input_binding_receipt_sha256"]
            or binding["canonical_signal_id"] != index["canonical_signal_id"]
        ):
            raise ValueError(
                "posterior canonical physical binding disagrees with index"
            )
        preprocessing = artifact["preprocessing_receipt"]
        if (
            not isinstance(preprocessing, dict)
            or preprocessing.get("canonical_detector_input_binding_id")
            != binding["binding_id"]
            or preprocessing.get("canonical_detector_input_binding_receipt_sha256")
            != binding["receipt_sha256"]
            or preprocessing.get("imputed_channels_clinical_evidence_eligible")
            is not False
            or preprocessing.get("silent_time_padding") is not False
        ):
            raise ValueError("posterior preprocessing physical permissions drifted")
        input_receipt = index["detector_input_channel_receipt"]
        if (
            type(input_receipt) is not dict
            or index["detector_input_channel_receipt_sha256"] != _sha256(input_receipt)
            or preprocessing.get("detector_input_channel_receipt_id")
            != input_receipt.get("receipt_id")
            or preprocessing.get("detector_input_channel_receipt_sha256")
            != index["detector_input_channel_receipt_sha256"]
            or input_receipt.get("edf_annotations_used") is not False
            or input_receipt.get("imputed_channels_clinical_evidence_eligible")
            is not False
        ):
            raise ValueError("posterior detector input receipt drifted")
        missing = input_receipt.get("imputed_channel_ids")
        if (
            not isinstance(missing, list)
            or len(missing) != index["detector_imputed_channel_count"]
            or missing != preprocessing.get("missing_channel_ids")
            or missing != binding["imputed_channel_ids"]
        ):
            raise ValueError("posterior imputed-channel inventory drifted")
        if missing:
            imputed_recordings += 1
            imputed_channels += len(missing)
        time_support = _validate_time_support(
            artifact["posterior_time_support_receipt"],
            recording_id=recording_id,
            duration=duration,
            timeline=timeline,
        )
        if (
            time_support["receipt_id"] != index["posterior_time_support_receipt_id"]
            or time_support["receipt_sha256"]
            != index["posterior_time_support_receipt_sha256"]
            or index["offline_future_dependent"] is not True
            or index["posterior_timestamp_is_real_time_latency"] is not False
            or index["decision_available_at_recording_end"] is not True
            or index["partial_tail_present"]
            is not time_support["padding_and_tail_semantics"]["partial_tail_present"]
            or index["partial_tail_policy"] != DEEPSOZ_PARTIAL_TAIL_POLICY
        ):
            raise ValueError("posterior time-support index projection drifted")
        runtime = _validate_runtime_receipt(
            artifact["posterior_runtime_receipt"],
            recording_id=recording_id,
            expected_execution_modes={"new_oof_inference"},
        )
        if (
            runtime["receipt_id"] != index["posterior_runtime_receipt_id"]
            or runtime["receipt_sha256"] != index["posterior_runtime_receipt_sha256"]
            or tuple(row["fold_index"] for row in runtime["held_out_fold_wall_seconds"])
            != folds
        ):
            raise ValueError("posterior original runtime index projection drifted")
        current_runtime = _validate_runtime_receipt(
            index["current_run_runtime_receipt"],
            recording_id=recording_id,
            expected_execution_modes={"new_oof_inference", "resume_validation_only"},
        )
        if (
            index["current_run_runtime_receipt_sha256"]
            != current_runtime["receipt_sha256"]
        ):
            raise ValueError("posterior current-run runtime hash drifted")
        canonical_source_sha256 = _require_sha256(
            binding["canonical_source_signal_sha256"], "canonical source signal hash"
        )
        timeline_json = _canonical_json(timeline)
        record_body = {
            "patient_id": patient_id,
            "recording_id": recording_id,
            "duration_seconds": duration,
            "canonical_source_signal_sha256": canonical_source_sha256,
            "posterior_artifact_id": artifact_id,
            "posterior_file_sha256": file_sha256,
            "provider_receipt": provider_receipt,
            "posterior_timeline_sha256": hashlib.sha256(
                timeline_json.encode("utf-8")
            ).hexdigest(),
        }
        records.append(
            ValidatedDeepSOZPosteriorRecording(
                patient_id=patient_id,
                recording_id=recording_id,
                duration_seconds=duration,
                canonical_source_signal_sha256=canonical_source_sha256,
                posterior_artifact_id=artifact_id,
                posterior_file_sha256=file_sha256,
                record_binding_sha256=_sha256(record_body),
                provider_receipt_json=provider_receipt_json,
                posterior_timeline_json=timeline_json,
            )
        )
        artifact_hash_rows.append([recording_id, file_sha256, artifact_id])
        total_duration += duration
        original_runtime_total += float(runtime["total_compute_wall_seconds"])

    records.sort(key=lambda row: (row.patient_id, row.recording_id))
    observed_recordings = sorted(seen_recordings)
    observed_patients = sorted({row.patient_id for row in records})
    expected_recordings = _normalized_expected(
        expected_recording_ids, "expected recording"
    )
    expected_patients = _normalized_expected(expected_patient_ids, "expected patient")
    if expected_recordings is not None:
        expected_recordings = sorted(
            _recording_id_for_profile(value, profile=profile)
            for value in expected_recordings
        )
    if expected_patients is not None and any(
        str(int(value)) != value for value in expected_patients
    ):
        raise ValueError("expected patient inventory is not canonically normalized")
    if require_complete_inventory and (
        expected_recordings is None
        or expected_patients is None
        or expected_manifest_sha256 is None
    ):
        raise ValueError(
            "complete inventory validation requires expected manifest, recordings and patients"
        )
    inventory_completeness_verified = bool(
        expected_recordings is not None
        and expected_patients is not None
        and expected_manifest_sha256 is not None
    )
    if official_manifest_profile and (
        not inventory_completeness_verified
        or recording_count != profile.official_recording_count
        or len(observed_patients) != profile.official_patient_count
        or len(expected_recordings or ()) != profile.official_recording_count
        or len(expected_patients or ()) != profile.official_patient_count
    ):
        raise ValueError(
            f"official {profile.split} inventory must contain exactly "
            f"{profile.official_recording_count} recordings and "
            f"{profile.official_patient_count} patients"
        )
    if expected_recordings is not None and expected_recordings != observed_recordings:
        raise ValueError("posterior batch recording inventory differs from expectation")
    if expected_patients is not None and expected_patients != observed_patients:
        raise ValueError("posterior batch patient inventory differs from expectation")
    posterior_directory = root / "posteriors"
    posterior_entries = list(posterior_directory.iterdir())
    if any(
        path.is_symlink() or not path.is_file() or path.suffix != ".json"
        for path in posterior_entries
    ):
        raise ValueError("posterior directory contains an unsafe extra entry")
    actual_files = sorted(path.resolve(strict=True) for path in posterior_entries)
    indexed_files = sorted(
        (root / PurePosixPath(value)).resolve(strict=True) for value in seen_paths
    )
    if actual_files != indexed_files or any(
        path.suffix != ".json" for path in actual_files
    ):
        raise ValueError("posterior directory contains unindexed or missing artifacts")
    if (
        imputed_recordings != batch["records_with_detector_channel_imputation"]
        or imputed_channels != batch["total_detector_imputed_channels"]
    ):
        raise ValueError("posterior batch imputation counters drifted")
    _validate_batch_runtime(
        batch["batch_runtime_receipt"], total_duration=total_duration
    )

    validation: dict[str, Any] = {
        "schema_version": DEEPSOZ_POSTERIOR_BATCH_VALIDATION_SCHEMA_VERSION,
        "validation_id": "DEEPSOZ-BATCH-VALIDATION-PENDING",
        "method_id": DEEPSOZ_POSTERIOR_BATCH_VALIDATION_METHOD_ID,
        "batch_root": str(root),
        "provider_id": DEEPSOZ_PROVIDER_ID,
        "selected_split": profile.split,
        "recording_namespace": profile.recording_namespace,
        "inventory_scope": batch["inventory_scope"],
        "inventory_completeness_verified": inventory_completeness_verified,
        "official_manifest_inventory_counts_verified": (official_manifest_profile),
        "recording_count": len(records),
        "patient_count": len(observed_patients),
        "recording_ids_sha256": _sha256(observed_recordings),
        "patient_ids_sha256": _sha256(observed_patients),
        "manifest_sha256": manifest_sha256,
        "batch_receipt_file_sha256": _file_sha256(batch_bytes),
        "posterior_index_file_sha256": _file_sha256(index_bytes),
        "posterior_artifact_inventory_sha256": _sha256(sorted(artifact_hash_rows)),
        "fold_assignment_receipt_sha256": fold_receipt_sha256,
        "weights_manifest_sha256": batch["weights_manifest_sha256"],
        "adapter_code_sha256": adapter_code_sha256,
        "materializer_code_sha256": materializer_code_sha256,
        "provider_receipt_sha256": _sha256(provider_receipt),
        "total_recording_duration_seconds": total_duration,
        "original_inference_wall_seconds": original_runtime_total,
        "all_original_runtime_receipts_verified": True,
        "all_canonical_physical_bindings_verified": True,
        "all_time_support_receipts_verified": True,
        "all_posterior_artifact_content_ids_verified": True,
        "reference_access": {
            "reference_path_argument_accepted": False,
            "reference_files_opened": 0,
            "edf_annotations_opened": 0,
            "excel_files_opened": 0,
            "clinical_text_opened": 0,
            "source_eval_opened": 0,
        },
        "research_only": True,
        "production_qualified": False,
        "sota_claim_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    validation["validation_id"] = "DSZBATCHVALID-" + _sha256(validation)[:24]
    validation["receipt_sha256"] = _sha256(validation)
    sealed = ValidatedDeepSOZPosteriorBatch(
        batch_root=str(root),
        recordings=tuple(records),
        validation_receipt_json=_canonical_json(validation),
    )
    return revalidate_deepsoz_posterior_batch_without_references(sealed)


__all__ = [
    "DEEPSOZ_652_OVERLAY_MANIFEST_SHA256",
    "DEEPSOZ_ADAPTER_CODE_SHA256",
    "DEEPSOZ_MATERIALIZER_CODE_SHA256",
    "DEEPSOZ_OFFICIAL_FOLD_ASSIGNMENT_RECEIPT_SHA256",
    "DEEPSOZ_POSTERIOR_BATCH_VALIDATION_METHOD_ID",
    "DEEPSOZ_POSTERIOR_BATCH_VALIDATION_SCHEMA_VERSION",
    "ValidatedDeepSOZPosteriorBatch",
    "ValidatedDeepSOZPosteriorRecording",
    "revalidate_deepsoz_posterior_batch_without_references",
    "validate_deepsoz_posterior_batch_without_references",
]
