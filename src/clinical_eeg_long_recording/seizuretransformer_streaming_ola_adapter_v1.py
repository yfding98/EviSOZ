"""Streaming full-record tiling and sample-aligned OLA for SeizureTransformer.

This module implements only the frozen *inference geometry* for the
``seizuretransformer_cleanroom_retrained_v1`` arm.  It deliberately does not
implement preprocessing, training, checkpoint loading, event decoding, or a
clinical operating point.  It accepts either an independently materialized
ST18 or ST16 provider-native float32 carrier and a 256 Hz physical clock.  The
variant-specific provider transform is bound by content hashes rather than
guessed here; ST16 is never obtained by slicing an ST18 tensor.

The primary profile is fixed to 60 second tiles and a 15 second hop.  Dense
tile probabilities are accumulated directly into absolute recording sample
indices with a raised-cosine/plateau window.  Tile outputs are never flattened
or concatenated along time.  The vendored architecture exposes no attention
padding mask, so records shorter than one 60 second model tile fail before any
model forward.  Zero-padding such a record and trimming its output would not
make the observed predictions padding-safe.

Scientific boundary
-------------------
The returned posterior is a detector signal only.  It is not a seizure,
clinical onset, Finding, SOZ label, or report fact.  No public or third-party
weight is loaded by this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np


SEIZURETRANSFORMER_PROVIDER_ID = "seizuretransformer_cleanroom_retrained_v1"
SEIZURETRANSFORMER_STREAMING_PLAN_SCHEMA_VERSION = (
    "seizuretransformer_streaming_tile_plan_v1"
)
SEIZURETRANSFORMER_STREAMING_RESULT_SCHEMA_VERSION = (
    "seizuretransformer_sample_aligned_ola_result_v1"
)
SEIZURETRANSFORMER_PHYSICAL_CLOCK_SCHEMA_VERSION = (
    "seizuretransformer_target_physical_clock_binding_v1"
)
SEIZURETRANSFORMER_STREAMING_METHOD_ID = (
    "st_cleanroom_independent_variant_256hz_w15360_h3840_absolute_sample_ola_v2"
)

SEIZURETRANSFORMER_ST18_VARIANT_ID = "seizuretransformer_st18_cleanroom_v1"
SEIZURETRANSFORMER_ST16_VARIANT_ID = (
    "seizuretransformer_st16_common_support_cleanroom_v1"
)
SEIZURETRANSFORMER_ST18_TYPED_UNITS = (
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
)
SEIZURETRANSFORMER_ST16_TYPED_UNITS = tuple(
    unit
    for unit in SEIZURETRANSFORMER_ST18_TYPED_UNITS
    if unit not in {"FZ-CZ", "CZ-PZ"}
)
# Backward-compatible name for the original ST18 roster.
SEIZURETRANSFORMER_TYPED_UNITS = SEIZURETRANSFORMER_ST18_TYPED_UNITS
SEIZURETRANSFORMER_SAMPLING_RATE_HZ = 256
SEIZURETRANSFORMER_WINDOW_SECONDS = 60
SEIZURETRANSFORMER_HOP_SECONDS = 15
SEIZURETRANSFORMER_WINDOW_SAMPLES = (
    SEIZURETRANSFORMER_SAMPLING_RATE_HZ * SEIZURETRANSFORMER_WINDOW_SECONDS
)
SEIZURETRANSFORMER_HOP_SAMPLES = (
    SEIZURETRANSFORMER_SAMPLING_RATE_HZ * SEIZURETRANSFORMER_HOP_SECONDS
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_PLAN_ID_PENDING = "SEIZURETRANSFORMER-TILE-PLAN-PENDING"
_RESULT_ID_PENDING = "SEIZURETRANSFORMER-OLA-RESULT-PENDING"
_CONTENT_PENDING = "CONTENT-ADDRESS-PENDING"

_FORWARD_SCOPE_RECEIPT = {
    "provider_preprocessed_eeg_samples_used": True,
    "typed_unit_and_physical_clock_metadata_used": True,
    "lineage_hashes_used_as_model_features": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_history_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
}

def _primary_profile(variant_id: str, typed_units: Sequence[str]) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "profile_id": f"{variant_id}_256hz_w60s_h15s_primary_accuracy_v2",
        "typed_unit_count": len(typed_units),
        "typed_units": list(typed_units),
        "target_sampling_rate_hz": SEIZURETRANSFORMER_SAMPLING_RATE_HZ,
        "target_tile_seconds": SEIZURETRANSFORMER_WINDOW_SECONDS,
        "stride_seconds": SEIZURETRANSFORMER_HOP_SECONDS,
        "overlap_fraction": 0.75,
        "window_samples": SEIZURETRANSFORMER_WINDOW_SAMPLES,
        "hop_samples": SEIZURETRANSFORMER_HOP_SAMPLES,
        "tile_start_policy": (
            "require_N_ge_W_then_sorted_unique_of_range_0_through_N_minus_W_"
            "in_steps_H_plus_exact_N_minus_W"
        ),
        "minimum_record_samples": SEIZURETRANSFORMER_WINDOW_SAMPLES,
        "short_record_policy": (
            "terminal_technical_failure_before_model_forward_because_"
            "architecture_has_no_attention_padding_mask"
        ),
        "architecture_attention_padding_mask_supported": False,
        "model_forward_on_padded_short_record_allowed": False,
        "aggregation": (
            "sample_aligned_probability_weighted_overlap_add_never_batch_flatten"
        ),
        "base_weight": (
            "raised_cosine_15_second_left_ramp_unit_30_second_plateau_"
            "raised_cosine_15_second_right_ramp"
        ),
        "left_ramp_formula": (
            "w_n_equals_sin_squared_of_pi_times_n_plus_one_half_divided_by_two_H_"
            "for_integer_0_le_n_lt_H"
        ),
        "right_ramp_formula": (
            "w_n_equals_left_ramp_at_W_minus_1_minus_n_for_integer_"
            "W_minus_H_le_n_lt_W"
        ),
        "plateau_formula": "w_n_equals_1_for_integer_H_le_n_lt_W_minus_H",
        "record_edge_weight": (
            "replace_left_ramp_by_unit_for_first_tile_and_right_ramp_by_unit_for_"
            "exact_last_observed_tile"
        ),
        "nonobserved_padding_weight": 0,
        "normalization": (
            "divide_each_absolute_recording_sample_weighted_probability_sum_by_"
            "its_strictly_positive_weight_sum"
        ),
        "tile_output_flatten_or_concatenate_allowed": False,
        "runtime_slice_from_other_variant_allowed": False,
    }


_ST18_PRIMARY_PROFILE = _primary_profile(
    SEIZURETRANSFORMER_ST18_VARIANT_ID, SEIZURETRANSFORMER_ST18_TYPED_UNITS
)
_ST16_PRIMARY_PROFILE = _primary_profile(
    SEIZURETRANSFORMER_ST16_VARIANT_ID, SEIZURETRANSFORMER_ST16_TYPED_UNITS
)

_VARIANT_REGISTRY = {
    SEIZURETRANSFORMER_ST18_VARIANT_ID: {
        "status": "streaming_geometry_implemented_preprocessing_and_checkpoint_pending",
        "typed_units": list(SEIZURETRANSFORMER_ST18_TYPED_UNITS),
        "primary_profile": deepcopy(_ST18_PRIMARY_PROFILE),
        "independent_model_variant_required": True,
    },
    SEIZURETRANSFORMER_ST16_VARIANT_ID: {
        "status": "streaming_geometry_implemented_preprocessing_and_checkpoint_pending",
        "typed_units": list(SEIZURETRANSFORMER_ST16_TYPED_UNITS),
        "primary_profile": deepcopy(_ST16_PRIMARY_PROFILE),
        "independent_model_variant_required": True,
        "runtime_ST18_tensor_slicing_allowed": False,
        "ST18_checkpoint_reuse_allowed": False,
    },
}


class SeizureTransformerTileReader(Protocol):
    """Strongly bound provider-carrier reader used by the streaming adapter."""

    variant_id: str
    typed_units: Sequence[str]
    sampling_rate_numerator: int
    sampling_rate_denominator: int
    sample_count: int
    source_signal_sha256: str
    preprocessing_receipt_sha256: str
    input_clock_receipt_sha256: str

    def read_samples(self, start_sample: int, sample_count: int) -> object:
        """Return exact-variant float32 samples ``[typed_units, sample_count]``."""


@dataclass(frozen=True)
class SeizureTransformerTile:
    """One model-sized tile and its absolute-time binding."""

    descriptor: dict[str, Any]
    signal: np.ndarray
    input_payload_receipt: dict[str, Any]


@dataclass(frozen=True)
class SeizureTransformerStreamingResult:
    """One full-record dense posterior and its replayable OLA receipt."""

    posterior_probability: np.ndarray
    receipt: dict[str, Any]


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
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return str(value)


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return int(value)


def _fraction(value: Fraction) -> list[int]:
    return [int(value.numerator), int(value.denominator)]


def _sample_edge_seconds_fraction(sample_index: int) -> list[int]:
    return _fraction(Fraction(sample_index, SEIZURETRANSFORMER_SAMPLING_RATE_HZ))


def _array_from_tensor_like(value: object, *, context: str) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        numpy_method = getattr(candidate, "numpy", None)
        if not callable(numpy_method):
            raise TypeError(f"{context} tensor cannot be converted to NumPy")
        candidate = numpy_method()
    try:
        return np.asarray(candidate)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be array-like") from exc


def _little_endian_float32(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype="<f4")


def _payload_receipt(value: np.ndarray, *, semantic: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype == np.dtype("float32"):
        array = np.ascontiguousarray(array, dtype="<f4")
        dtype = "float32_little_endian"
    elif array.dtype == np.dtype("float64"):
        array = np.ascontiguousarray(array, dtype="<f8")
        dtype = "float64_little_endian"
    elif array.dtype == np.dtype("uint16"):
        array = np.ascontiguousarray(array, dtype="<u2")
        dtype = "uint16_little_endian"
    else:
        raise TypeError("unsupported receipt payload dtype")
    body = {
        "semantic": semantic,
        "dtype": dtype,
        "shape": [int(item) for item in array.shape],
        "payload_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    if array.dtype.kind == "f":
        body["minimum"] = float(np.min(array))
        body["maximum"] = float(np.max(array))
    else:
        body["minimum"] = int(np.min(array))
        body["maximum"] = int(np.max(array))
    return body


def seizuretransformer_streaming_adapter_code_sha256() -> str:
    """Return the exact implementation hash used by a runtime receipt."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def seizuretransformer_streaming_variant_registry() -> dict[str, Any]:
    """Expose implemented and deliberately unimplemented independent variants."""

    return deepcopy(_VARIANT_REGISTRY)


def _implemented_variant_profile(variant_id: object) -> dict[str, Any]:
    if not isinstance(variant_id, str) or not variant_id:
        raise ValueError("SeizureTransformer variant_id must be a non-empty string")
    if variant_id not in _VARIANT_REGISTRY:
        raise ValueError("unknown SeizureTransformer independent model variant")
    row = _VARIANT_REGISTRY[variant_id]
    if row["status"] == "not_implemented_fail_closed":
        raise NotImplementedError(
            f"{variant_id} is not implemented: {row['unimplemented_reason']}"
        )
    profile = row["primary_profile"]
    if not isinstance(profile, dict):
        raise RuntimeError("implemented SeizureTransformer variant has no profile")
    return deepcopy(profile)


def seizuretransformer_tile_starts(sample_count: int) -> tuple[int, ...]:
    """Return the exact frozen primary-profile tile starts."""

    count = _positive_integer(sample_count, "sample_count")
    window = SEIZURETRANSFORMER_WINDOW_SAMPLES
    hop = SEIZURETRANSFORMER_HOP_SAMPLES
    if count < window:
        raise ValueError(
            "record shorter than 60 seconds is unadmitted because the vendored "
            "architecture has no attention padding mask"
        )
    if count == window:
        return (0,)
    final_start = count - window
    starts = set(range(0, final_start + 1, hop))
    starts.add(final_start)
    return tuple(sorted(starts))


def _tile_id(descriptor: Mapping[str, Any]) -> str:
    pending = deepcopy(dict(descriptor))
    pending["tile_id"] = _CONTENT_PENDING
    return "ST-TILE-" + _canonical_sha256(pending)[:24]


def _tile_descriptors(sample_count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    starts = seizuretransformer_tile_starts(sample_count)
    window = SEIZURETRANSFORMER_WINDOW_SAMPLES
    for tile_index, start in enumerate(starts):
        observed_stop = min(start + window, sample_count)
        observed_count = observed_stop - start
        padding = window - observed_count
        if padding != 0 or observed_count != window:
            raise RuntimeError(
                "admitted SeizureTransformer plans may not contain padded model tiles"
            )
        descriptor: dict[str, Any] = {
            "tile_id": _CONTENT_PENDING,
            "tile_index": tile_index,
            "absolute_model_input_sample_range": [start, start + window],
            "absolute_observed_sample_range": [start, observed_stop],
            "absolute_observed_seconds_fraction": [
                _sample_edge_seconds_fraction(start),
                _sample_edge_seconds_fraction(observed_stop),
            ],
            "local_observed_sample_range": [0, observed_count],
            "local_nonobserved_padding_sample_range": (
                [] if padding == 0 else [observed_count, window]
            ),
            "observed_sample_count": observed_count,
            "right_padding_samples": padding,
            "is_first_tile": tile_index == 0,
            "is_exact_last_observed_tile": observed_stop == sample_count,
            "weight_profile": (
                "raised_cosine_plateau_with_record_edge_units_and_padding_zero_v1"
            ),
        }
        descriptor["tile_id"] = _tile_id(descriptor)
        result.append(descriptor)
    return result


def _merge_intervals(intervals: Sequence[Sequence[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, stop in sorted((int(row[0]), int(row[1])) for row in intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return merged


def _physical_clock_receipt(
    *,
    sample_count: int,
    input_clock_receipt_sha256: str,
    source_signal_sha256: str,
    preprocessing_receipt_sha256: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": SEIZURETRANSFORMER_PHYSICAL_CLOCK_SCHEMA_VERSION,
        "clock_policy": "integer_sample_edges_on_recording_relative_rational_clock_v1",
        "parent_input_clock_receipt_sha256": input_clock_receipt_sha256,
        "source_signal_sha256": source_signal_sha256,
        "preprocessing_receipt_sha256": preprocessing_receipt_sha256,
        "sampling_rate_numerator": SEIZURETRANSFORMER_SAMPLING_RATE_HZ,
        "sampling_rate_denominator": 1,
        "sample_period_seconds_fraction": [
            1,
            SEIZURETRANSFORMER_SAMPLING_RATE_HZ,
        ],
        "recording_relative_origin_seconds_fraction": [0, 1],
        "recording_sample_range": [0, sample_count],
        "recording_stop_seconds_fraction": _sample_edge_seconds_fraction(sample_count),
        "absolute_sample_edge_mapping": "edge_i_seconds_equals_i_divided_by_256",
        "posterior_sample_i_target_interval": "half_open_edges_i_to_i_plus_1",
        "nonobserved_padding_has_physical_time": False,
        "receipt_sha256": _CONTENT_PENDING,
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _plan_coverage_receipt(
    *, sample_count: int, descriptors: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    intervals = [row["absolute_observed_sample_range"] for row in descriptors]
    merged = _merge_intervals(intervals)
    covered = sum(stop - start for start, stop in merged)
    return {
        "recording_sample_range": [0, sample_count],
        "planned_tile_count": len(descriptors),
        "planned_observed_interval_roster_sha256": _canonical_sha256(intervals),
        "merged_observed_sample_intervals": merged,
        "covered_record_sample_count": covered,
        "uncovered_record_sample_count": sample_count - covered,
        "complete_first_sample_coverage_planned": bool(
            merged and merged[0][0] == 0 and merged[0][1] > 0
        ),
        "complete_final_sample_coverage_planned": bool(
            merged and merged[-1][1] == sample_count
        ),
        "complete_record_coverage_planned": merged == [[0, sample_count]],
        "total_nonobserved_right_padding_samples": sum(
            int(row["right_padding_samples"]) for row in descriptors
        ),
        "padding_weight": 0,
    }


def build_seizuretransformer_tile_plan(
    sample_count: int,
    *,
    variant_id: str,
    input_clock_receipt_sha256: str,
    source_signal_sha256: str,
    preprocessing_receipt_sha256: str,
) -> dict[str, Any]:
    """Build a content-addressed, full-record 60 s / 15 s tile plan.

    The three hashes preserve the upstream physical-signal, provider-transform,
    and exact provider-output-clock lineage.  They are receipt inputs only and
    are never passed to the predictor.
    """

    count = _positive_integer(sample_count, "sample_count")
    profile = _implemented_variant_profile(variant_id)
    clock_hash = _require_sha256(
        input_clock_receipt_sha256, "input_clock_receipt_sha256"
    )
    source_hash = _require_sha256(source_signal_sha256, "source_signal_sha256")
    preprocessing_hash = _require_sha256(
        preprocessing_receipt_sha256, "preprocessing_receipt_sha256"
    )
    clock = _physical_clock_receipt(
        sample_count=count,
        input_clock_receipt_sha256=clock_hash,
        source_signal_sha256=source_hash,
        preprocessing_receipt_sha256=preprocessing_hash,
    )
    descriptors = _tile_descriptors(count)
    body: dict[str, Any] = {
        "schema_version": SEIZURETRANSFORMER_STREAMING_PLAN_SCHEMA_VERSION,
        "plan_id": _PLAN_ID_PENDING,
        "provider_id": SEIZURETRANSFORMER_PROVIDER_ID,
        "variant_id": variant_id,
        "method_id": SEIZURETRANSFORMER_STREAMING_METHOD_ID,
        "adapter_code_sha256": seizuretransformer_streaming_adapter_code_sha256(),
        "primary_profile": profile,
        "source_binding": {
            "source_signal_sha256": source_hash,
            "preprocessing_receipt_sha256": preprocessing_hash,
            "input_clock_receipt_sha256": clock_hash,
        },
        "physical_clock_receipt": clock,
        "sample_count": count,
        "tile_descriptors": descriptors,
        "coverage_receipt": _plan_coverage_receipt(
            sample_count=count, descriptors=descriptors
        ),
        "forward_scope_receipt": deepcopy(_FORWARD_SCOPE_RECEIPT),
        "receipt_sha256": _CONTENT_PENDING,
    }
    identity = deepcopy(body)
    body["plan_id"] = "ST-TILEPLAN-" + _canonical_sha256(identity)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_seizuretransformer_tile_plan(body)


def validate_seizuretransformer_tile_plan(payload: object) -> dict[str, Any]:
    """Replay the complete frozen tiling, clock, and coverage plan."""

    required = {
        "schema_version",
        "plan_id",
        "provider_id",
        "variant_id",
        "method_id",
        "adapter_code_sha256",
        "primary_profile",
        "source_binding",
        "physical_clock_receipt",
        "sample_count",
        "tile_descriptors",
        "coverage_receipt",
        "forward_scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("SeizureTransformer tile plan fields drifted")
    data = deepcopy(payload)
    if data["schema_version"] != SEIZURETRANSFORMER_STREAMING_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported SeizureTransformer tile plan schema")
    if data["provider_id"] != SEIZURETRANSFORMER_PROVIDER_ID:
        raise ValueError("SeizureTransformer provider ID drifted")
    expected_profile = _implemented_variant_profile(data["variant_id"])
    if data["method_id"] != SEIZURETRANSFORMER_STREAMING_METHOD_ID:
        raise ValueError("SeizureTransformer streaming method drifted")
    if (
        data["adapter_code_sha256"]
        != seizuretransformer_streaming_adapter_code_sha256()
    ):
        raise ValueError("SeizureTransformer adapter code binding drifted")
    if data["primary_profile"] != expected_profile:
        raise ValueError("SeizureTransformer primary inference profile drifted")
    if data["forward_scope_receipt"] != _FORWARD_SCOPE_RECEIPT:
        raise ValueError("SeizureTransformer EEG-only scope drifted")

    count = _positive_integer(data["sample_count"], "tile plan sample_count")
    binding_required = {
        "source_signal_sha256",
        "preprocessing_receipt_sha256",
        "input_clock_receipt_sha256",
    }
    binding = data["source_binding"]
    if type(binding) is not dict or set(binding) != binding_required:
        raise ValueError("SeizureTransformer source binding fields drifted")
    source_hash = _require_sha256(binding["source_signal_sha256"], "source hash")
    preprocessing_hash = _require_sha256(
        binding["preprocessing_receipt_sha256"], "preprocessing hash"
    )
    clock_hash = _require_sha256(
        binding["input_clock_receipt_sha256"], "input clock hash"
    )
    expected_clock = _physical_clock_receipt(
        sample_count=count,
        input_clock_receipt_sha256=clock_hash,
        source_signal_sha256=source_hash,
        preprocessing_receipt_sha256=preprocessing_hash,
    )
    if data["physical_clock_receipt"] != expected_clock:
        raise ValueError("SeizureTransformer physical clock receipt drifted")

    expected_descriptors = _tile_descriptors(count)
    if data["tile_descriptors"] != expected_descriptors:
        raise ValueError("SeizureTransformer absolute tile ledger drifted")
    expected_coverage = _plan_coverage_receipt(
        sample_count=count, descriptors=expected_descriptors
    )
    if data["coverage_receipt"] != expected_coverage:
        raise ValueError("SeizureTransformer planned coverage receipt drifted")
    if expected_coverage["complete_record_coverage_planned"] is not True:
        raise ValueError("SeizureTransformer plan does not cover the full record")

    identity = deepcopy(data)
    identity["plan_id"] = _PLAN_ID_PENDING
    identity["receipt_sha256"] = _CONTENT_PENDING
    expected_plan_id = "ST-TILEPLAN-" + _canonical_sha256(identity)[:24]
    if data["plan_id"] != expected_plan_id:
        raise ValueError("SeizureTransformer tile plan ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = _CONTENT_PENDING
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("SeizureTransformer tile plan receipt is not content-bound")
    return data


def seizuretransformer_tile_weights(
    descriptor: Mapping[str, Any],
) -> np.ndarray:
    """Materialize the exact float64 OLA weights for one validated tile row."""

    if type(descriptor) is not dict:
        raise TypeError("tile descriptor must be an object")
    window = SEIZURETRANSFORMER_WINDOW_SAMPLES
    hop = SEIZURETRANSFORMER_HOP_SAMPLES
    observed_count = _positive_integer(
        descriptor.get("observed_sample_count"), "observed_sample_count"
    )
    if observed_count > window:
        raise ValueError("observed tile length exceeds the frozen window")
    ramp_index = np.arange(hop, dtype=np.float64)
    left_ramp = np.sin(np.pi * (ramp_index + 0.5) / (2.0 * float(hop))) ** 2
    weights = np.ones(window, dtype=np.float64)
    weights[:hop] = left_ramp
    weights[window - hop :] = left_ramp[::-1]
    if descriptor.get("is_first_tile") is True:
        weights[:hop] = 1.0
    if descriptor.get("is_exact_last_observed_tile") is True:
        weights[window - hop :] = 1.0
    if observed_count < window:
        weights[observed_count:] = 0.0
    weights.setflags(write=False)
    return weights


def _validate_reader_binding(
    reader: SeizureTransformerTileReader,
    *,
    plan: Mapping[str, Any],
) -> None:
    method = getattr(reader, "read_samples", None)
    if not callable(method):
        raise TypeError("reader must expose read_samples(start_sample, sample_count)")
    if getattr(reader, "variant_id", None) != plan["variant_id"]:
        raise ValueError(
            "reader variant_id disagrees with the bound independent model variant"
        )
    typed_units = getattr(reader, "typed_units", None)
    if isinstance(typed_units, (str, bytes)):
        raise ValueError("reader typed_units must be the exact ordered variant roster")
    try:
        units = tuple(typed_units)
    except TypeError as exc:
        raise ValueError(
            "reader typed_units must be the exact ordered variant roster"
        ) from exc
    expected_units = tuple(plan["primary_profile"]["typed_units"])
    if units != expected_units:
        raise ValueError(
            "reader typed-unit order disagrees with the bound variant roster"
        )
    for field, expected in (
        ("sampling_rate_numerator", SEIZURETRANSFORMER_SAMPLING_RATE_HZ),
        ("sampling_rate_denominator", 1),
        ("sample_count", int(plan["sample_count"])),
    ):
        observed = getattr(reader, field, None)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed != expected
        ):
            raise ValueError(f"reader {field} disagrees with the bound tile plan")
    binding = plan["source_binding"]
    for field in (
        "source_signal_sha256",
        "preprocessing_receipt_sha256",
        "input_clock_receipt_sha256",
    ):
        observed = getattr(reader, field, None)
        _require_sha256(observed, f"reader {field}")
        if observed != binding[field]:
            raise ValueError(f"reader {field} disagrees with the bound tile plan")


def _read_tile_samples(
    reader: SeizureTransformerTileReader,
    *,
    start_sample: int,
    sample_count: int,
    typed_unit_count: int,
) -> np.ndarray:
    raw = reader.read_samples(start_sample, sample_count)
    array = _array_from_tensor_like(raw, context="streaming reader output")
    if array.dtype != np.dtype("float32"):
        raise TypeError("streaming reader must return provider-native float32 samples")
    expected_shape = (typed_unit_count, sample_count)
    if array.shape != expected_shape:
        raise ValueError(
            f"streaming reader returned shape {array.shape}, expected {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("streaming reader returned nonfinite EEG samples")
    return _little_endian_float32(array)


def iter_seizuretransformer_tiles(
    reader: SeizureTransformerTileReader,
    plan: Mapping[str, Any],
) -> Iterator[SeizureTransformerTile]:
    """Read and yield one bounded provider-native tile at a time."""

    validated = validate_seizuretransformer_tile_plan(dict(plan))
    _validate_reader_binding(reader, plan=validated)
    window = SEIZURETRANSFORMER_WINDOW_SAMPLES
    typed_unit_count = int(validated["primary_profile"]["typed_unit_count"])
    for descriptor in validated["tile_descriptors"]:
        start, observed_stop = descriptor["absolute_observed_sample_range"]
        observed_count = observed_stop - start
        observed = _read_tile_samples(
            reader,
            start_sample=start,
            sample_count=observed_count,
            typed_unit_count=typed_unit_count,
        )
        tile = np.zeros((typed_unit_count, window), dtype="<f4")
        tile[:, :observed_count] = observed
        tile = np.ascontiguousarray(tile, dtype="<f4")
        tile.setflags(write=False)
        yield SeizureTransformerTile(
            descriptor=deepcopy(descriptor),
            signal=tile,
            input_payload_receipt=_payload_receipt(
                tile,
                semantic=(
                    f"provider_preprocessed_{typed_unit_count}_bipolar_"
                    f"{validated['variant_id']}_model_tile"
                ),
            ),
        )


def _probability_vector(value: object) -> np.ndarray:
    raw = _array_from_tensor_like(value, context="tile probability")
    if raw.ndim != 1 or raw.shape[0] != SEIZURETRANSFORMER_WINDOW_SAMPLES:
        raise ValueError(
            "each tile probability must have exact shape [15360]; temporal "
            "flatten/concatenation is forbidden"
        )
    if raw.dtype.kind != "f":
        raise TypeError("tile probability must use a floating dtype")
    if not np.isfinite(raw).all():
        raise ValueError("tile probability contains nonfinite values")
    if np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ValueError("tile probability must be in the closed interval [0, 1]")
    return _little_endian_float32(raw)


def _expected_weight_coverage(
    plan: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    count = int(plan["sample_count"])
    weight_sum = np.zeros(count, dtype=np.float64)
    contribution_count = np.zeros(count, dtype=np.uint16)
    for descriptor in plan["tile_descriptors"]:
        start, stop = descriptor["absolute_observed_sample_range"]
        observed_count = stop - start
        weights = seizuretransformer_tile_weights(descriptor)[:observed_count]
        weight_sum[start:stop] += weights
        contribution_count[start:stop] += np.uint16(1)
    return weight_sum, contribution_count


def _coverage_receipt(
    *, weight_sum: np.ndarray, contribution_count: np.ndarray, tile_count: int
) -> dict[str, Any]:
    uncovered = int(np.count_nonzero(contribution_count == 0))
    nonpositive = int(np.count_nonzero(weight_sum <= 0.0))
    return {
        "planned_tile_count": tile_count,
        "contributed_tile_count": tile_count,
        "all_planned_tiles_contributed_exactly_once": True,
        "record_sample_count": int(weight_sum.shape[0]),
        "uncovered_record_sample_count": uncovered,
        "nonpositive_weight_sum_sample_count": nonpositive,
        "complete_record_posterior_coverage": uncovered == 0 and nonpositive == 0,
        "minimum_tile_contribution_count": int(np.min(contribution_count)),
        "maximum_tile_contribution_count": int(np.max(contribution_count)),
        "minimum_positive_weight_sum": float(np.min(weight_sum)),
        "maximum_weight_sum": float(np.max(weight_sum)),
        "weight_sum_payload_receipt": _payload_receipt(
            weight_sum, semantic="absolute_recording_sample_ola_weight_sum"
        ),
        "contribution_count_payload_receipt": _payload_receipt(
            contribution_count,
            semantic="absolute_recording_sample_tile_contribution_count",
        ),
        "normalization_denominator_strictly_positive": nonpositive == 0,
        "nonobserved_padding_contributed_to_numerator_or_denominator": False,
        "temporal_tile_output_flatten_or_concatenate_used": False,
    }


def _manual_execution_receipt(tile_count: int) -> dict[str, Any]:
    return {
        "mode": "external_incremental_tile_contributions_v1",
        "reader_interface": None,
        "reader_carrier_binding_verified": False,
        "read_call_count": None,
        "read_absolute_interval_roster_sha256": None,
        "predictor_call_count": None,
        "predictor_batch_sizes": [],
        "maximum_predictor_batch_size": None,
        "whole_record_input_materialized_by_adapter": False,
        "batch_axis_stack_only": False,
        "temporal_input_or_output_flatten_or_concatenate_used": False,
        "expected_tile_count": tile_count,
    }


def _validate_execution_receipt(
    payload: object,
    *,
    expected_tile_count: int,
    expected_read_interval_roster_sha256: str,
) -> dict[str, Any]:
    required = {
        "mode",
        "reader_interface",
        "reader_carrier_binding_verified",
        "read_call_count",
        "read_absolute_interval_roster_sha256",
        "predictor_call_count",
        "predictor_batch_sizes",
        "maximum_predictor_batch_size",
        "whole_record_input_materialized_by_adapter",
        "batch_axis_stack_only",
        "temporal_input_or_output_flatten_or_concatenate_used",
        "expected_tile_count",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("streaming execution receipt fields drifted")
    data = deepcopy(payload)
    if data["expected_tile_count"] != expected_tile_count:
        raise ValueError("streaming execution tile count drifted")
    if data["whole_record_input_materialized_by_adapter"] is not False:
        raise ValueError("adapter may not materialize the whole input recording")
    if data["temporal_input_or_output_flatten_or_concatenate_used"] is not False:
        raise ValueError("temporal flatten/concatenation is forbidden")
    if data["mode"] == "external_incremental_tile_contributions_v1":
        if data != _manual_execution_receipt(expected_tile_count):
            raise ValueError("manual incremental execution receipt drifted")
        return data
    if data["mode"] != "adapter_streaming_reader_batched_predictor_v1":
        raise ValueError("unsupported streaming execution mode")
    if data["reader_interface"] != "read_samples(start_sample,sample_count)":
        raise ValueError("streaming reader interface drifted")
    if data["reader_carrier_binding_verified"] is not True:
        raise ValueError("streaming reader carrier binding was not verified")
    for field in ("read_call_count", "predictor_call_count"):
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if data["read_call_count"] != expected_tile_count:
        raise ValueError("streaming read call count does not cover all tiles")
    _require_sha256(
        data["read_absolute_interval_roster_sha256"], "read interval roster hash"
    )
    if (
        data["read_absolute_interval_roster_sha256"]
        != expected_read_interval_roster_sha256
    ):
        raise ValueError(
            "streaming read intervals disagree with the absolute tile plan"
        )
    batch_sizes = data["predictor_batch_sizes"]
    if (
        not isinstance(batch_sizes, list)
        or not batch_sizes
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in batch_sizes
        )
        or sum(batch_sizes) != expected_tile_count
        or len(batch_sizes) != data["predictor_call_count"]
        or max(batch_sizes) != data["maximum_predictor_batch_size"]
    ):
        raise ValueError("predictor batch-size ledger is invalid")
    if data["batch_axis_stack_only"] is not True:
        raise ValueError("batched execution must attest batch-axis-only stacking")
    return data


class SeizureTransformerSampleAlignedOLA:
    """Incrementally accumulate dense tile probabilities on absolute samples."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        self._plan = validate_seizuretransformer_tile_plan(dict(plan))
        sample_count = int(self._plan["sample_count"])
        self._weighted_sum = np.zeros(sample_count, dtype=np.float64)
        self._weight_sum = np.zeros(sample_count, dtype=np.float64)
        self._contribution_count = np.zeros(sample_count, dtype=np.uint16)
        self._contributions: dict[int, dict[str, Any]] = {}
        self._finalized = False

    @property
    def plan(self) -> dict[str, Any]:
        return deepcopy(self._plan)

    def add(self, tile: SeizureTransformerTile, probability: object) -> None:
        """Add exactly one tile by its absolute ledger row."""

        if self._finalized:
            raise RuntimeError("cannot add a tile after OLA finalization")
        if not isinstance(tile, SeizureTransformerTile):
            raise TypeError("tile must be a SeizureTransformerTile")
        descriptor = tile.descriptor
        index = descriptor.get("tile_index") if isinstance(descriptor, dict) else None
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("tile index is invalid")
        descriptors = self._plan["tile_descriptors"]
        if index < 0 or index >= len(descriptors) or descriptor != descriptors[index]:
            raise ValueError("tile descriptor is not from the bound absolute tile plan")
        if index in self._contributions:
            raise ValueError("duplicate tile contribution is forbidden")
        expected_next_index = len(self._contributions)
        if index != expected_next_index:
            raise ValueError(
                "streaming tile contributions must follow absolute plan order; "
                f"expected tile {expected_next_index}, received {index}"
            )
        signal = np.asarray(tile.signal)
        typed_unit_count = int(self._plan["primary_profile"]["typed_unit_count"])
        if signal.dtype != np.dtype("float32") or signal.shape != (
            typed_unit_count,
            SEIZURETRANSFORMER_WINDOW_SAMPLES,
        ):
            raise ValueError("tile signal payload shape or dtype drifted")
        if not np.isfinite(signal).all():
            raise ValueError("tile signal payload contains nonfinite values")
        expected_input = _payload_receipt(
            signal,
            semantic=(
                f"provider_preprocessed_{typed_unit_count}_bipolar_"
                f"{self._plan['variant_id']}_model_tile"
            ),
        )
        if tile.input_payload_receipt != expected_input:
            raise ValueError("tile input payload receipt drifted")
        observed_count = int(descriptor["observed_sample_count"])
        if descriptor["right_padding_samples"] and np.any(
            signal[:, observed_count:] != 0
        ):
            raise ValueError("nonobserved right padding must remain exact zero")

        vector = _probability_vector(probability)
        weights = seizuretransformer_tile_weights(descriptor)
        start, stop = descriptor["absolute_observed_sample_range"]
        observed_probability = vector[:observed_count].astype(np.float64)
        observed_weights = weights[:observed_count]
        self._weighted_sum[start:stop] += observed_probability * observed_weights
        self._weight_sum[start:stop] += observed_weights
        self._contribution_count[start:stop] += np.uint16(1)
        self._contributions[index] = {
            "tile_id": descriptor["tile_id"],
            "tile_index": index,
            "absolute_observed_sample_range": [start, stop],
            "observed_sample_count": observed_count,
            "right_padding_samples": int(descriptor["right_padding_samples"]),
            "input_tile_payload_sha256": expected_input["payload_sha256"],
            "dense_tile_probability_payload_receipt": _payload_receipt(
                vector, semantic="dense_tile_probability_including_discarded_padding"
            ),
            "observed_tile_probability_payload_receipt": _payload_receipt(
                vector[:observed_count],
                semantic="dense_tile_probability_observed_prefix",
            ),
            "effective_tile_weight_payload_receipt": _payload_receipt(
                weights, semantic="tile_ola_weight_including_padding_zero"
            ),
            "padding_probability_discarded_before_ola": bool(
                descriptor["right_padding_samples"]
            ),
            "absolute_sample_alignment_used": True,
        }

    def finalize(
        self,
        *,
        streaming_execution_receipt: Mapping[str, Any] | None = None,
    ) -> SeizureTransformerStreamingResult:
        """Normalize every absolute sample after all planned tiles arrive."""

        if self._finalized:
            raise RuntimeError("OLA has already been finalized")
        expected_indices = set(range(len(self._plan["tile_descriptors"])))
        observed_indices = set(self._contributions)
        if observed_indices != expected_indices:
            missing = sorted(expected_indices.difference(observed_indices))
            raise ValueError(
                f"cannot finalize incomplete tile ledger; missing={missing}"
            )
        expected_weight_sum, expected_count = _expected_weight_coverage(self._plan)
        if not np.array_equal(self._contribution_count, expected_count):
            raise ValueError("actual tile contribution count disagrees with the plan")
        if not np.array_equal(self._weight_sum, expected_weight_sum):
            raise ValueError("actual absolute OLA weights disagree with the plan")
        if np.any(self._weight_sum <= 0.0):
            raise ValueError("every record sample must have a positive OLA denominator")

        posterior = _little_endian_float32(self._weighted_sum / self._weight_sum)
        if (
            not np.isfinite(posterior).all()
            or np.any(posterior < 0.0)
            or np.any(posterior > 1.0)
        ):
            raise ValueError("normalized full-record posterior is invalid")
        posterior.setflags(write=False)
        execution = (
            _manual_execution_receipt(len(expected_indices))
            if streaming_execution_receipt is None
            else _validate_execution_receipt(
                dict(streaming_execution_receipt),
                expected_tile_count=len(expected_indices),
                expected_read_interval_roster_sha256=_canonical_sha256(
                    [
                        row["absolute_observed_sample_range"]
                        for row in self._plan["tile_descriptors"]
                    ]
                ),
            )
        )
        coverage = _coverage_receipt(
            weight_sum=self._weight_sum,
            contribution_count=self._contribution_count,
            tile_count=len(expected_indices),
        )
        body: dict[str, Any] = {
            "schema_version": SEIZURETRANSFORMER_STREAMING_RESULT_SCHEMA_VERSION,
            "result_id": _RESULT_ID_PENDING,
            "provider_id": SEIZURETRANSFORMER_PROVIDER_ID,
            "variant_id": self._plan["variant_id"],
            "method_id": SEIZURETRANSFORMER_STREAMING_METHOD_ID,
            "adapter_code_sha256": seizuretransformer_streaming_adapter_code_sha256(),
            "tile_plan": deepcopy(self._plan),
            "physical_clock_receipt": deepcopy(self._plan["physical_clock_receipt"]),
            "tile_contribution_receipts": [
                deepcopy(self._contributions[index])
                for index in sorted(self._contributions)
            ],
            "ola_coverage_receipt": coverage,
            "posterior_payload_receipt": _payload_receipt(
                posterior, semantic="full_record_dense_seizure_probability"
            ),
            "streaming_execution_receipt": execution,
            "scientific_permissions": {
                "cleanroom_checkpoint_required_but_verified_here": False,
                "qualified_operating_point_exists": False,
                "posterior_is_confirmed_seizure_or_clinical_onset": False,
                "posterior_is_findings_or_soz_evidence": False,
                "clinical_or_production_use_authorized": False,
            },
            "receipt_sha256": _CONTENT_PENDING,
        }
        identity = deepcopy(body)
        body["result_id"] = "ST-OLA-" + _canonical_sha256(identity)[:24]
        body["receipt_sha256"] = _canonical_sha256(body)
        self._finalized = True
        result = SeizureTransformerStreamingResult(
            posterior_probability=posterior,
            receipt=body,
        )
        validate_seizuretransformer_streaming_result(result)
        return result


def _predictor_batch(value: object, *, batch_size: int) -> np.ndarray:
    array = _array_from_tensor_like(value, context="batched dense tile probability")
    expected = (batch_size, SEIZURETRANSFORMER_WINDOW_SAMPLES)
    if batch_size == 1 and array.shape == (SEIZURETRANSFORMER_WINDOW_SAMPLES,):
        array = array.reshape(expected)
    if array.shape != expected:
        raise ValueError(
            f"predictor must return exact shape {expected}; temporal flatten or "
            "concatenation is forbidden"
        )
    if array.dtype.kind != "f":
        raise TypeError("predictor probability output must use a floating dtype")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("predictor output must contain finite probabilities in [0, 1]")
    return _little_endian_float32(array)


def run_seizuretransformer_streaming_ola(
    reader: SeizureTransformerTileReader,
    predictor: Callable[[np.ndarray], object],
    plan: Mapping[str, Any],
    *,
    batch_size: int = 1,
) -> SeizureTransformerStreamingResult:
    """Run bounded tile batches and stitch dense probabilities by absolute sample.

    ``np.stack`` is used only to form the model batch axis.  Neither EEG nor
    posterior arrays are flattened or concatenated along their time axis.
    """

    if not callable(predictor):
        raise TypeError("predictor must be callable")
    batch_limit = _positive_integer(batch_size, "batch_size")
    validated = validate_seizuretransformer_tile_plan(dict(plan))
    accumulator = SeizureTransformerSampleAlignedOLA(validated)
    pending: list[SeizureTransformerTile] = []
    batch_sizes: list[int] = []
    read_intervals: list[list[int]] = []

    def consume(batch: list[SeizureTransformerTile]) -> None:
        if not batch:
            return
        model_input = np.stack([tile.signal for tile in batch], axis=0)
        if model_input.shape != (
            len(batch),
            int(validated["primary_profile"]["typed_unit_count"]),
            SEIZURETRANSFORMER_WINDOW_SAMPLES,
        ):
            raise RuntimeError("internal model batch shape drifted")
        probabilities = _predictor_batch(predictor(model_input), batch_size=len(batch))
        for row, tile in zip(probabilities, batch):
            accumulator.add(tile, row)
        batch_sizes.append(len(batch))

    for tile in iter_seizuretransformer_tiles(reader, validated):
        read_intervals.append(
            deepcopy(tile.descriptor["absolute_observed_sample_range"])
        )
        pending.append(tile)
        if len(pending) == batch_limit:
            consume(pending)
            pending = []
    consume(pending)
    execution = {
        "mode": "adapter_streaming_reader_batched_predictor_v1",
        "reader_interface": "read_samples(start_sample,sample_count)",
        "reader_carrier_binding_verified": True,
        "read_call_count": len(read_intervals),
        "read_absolute_interval_roster_sha256": _canonical_sha256(read_intervals),
        "predictor_call_count": len(batch_sizes),
        "predictor_batch_sizes": batch_sizes,
        "maximum_predictor_batch_size": max(batch_sizes),
        "whole_record_input_materialized_by_adapter": False,
        "batch_axis_stack_only": True,
        "temporal_input_or_output_flatten_or_concatenate_used": False,
        "expected_tile_count": len(validated["tile_descriptors"]),
    }
    return accumulator.finalize(streaming_execution_receipt=execution)


def validate_seizuretransformer_streaming_result(
    payload: SeizureTransformerStreamingResult,
) -> dict[str, Any]:
    """Validate output bytes, absolute coverage, clock binding, and permissions."""

    if not isinstance(payload, SeizureTransformerStreamingResult):
        raise TypeError("payload must be a SeizureTransformerStreamingResult")
    posterior = np.asarray(payload.posterior_probability)
    data = deepcopy(payload.receipt)
    required = {
        "schema_version",
        "result_id",
        "provider_id",
        "variant_id",
        "method_id",
        "adapter_code_sha256",
        "tile_plan",
        "physical_clock_receipt",
        "tile_contribution_receipts",
        "ola_coverage_receipt",
        "posterior_payload_receipt",
        "streaming_execution_receipt",
        "scientific_permissions",
        "receipt_sha256",
    }
    if type(data) is not dict or set(data) != required:
        raise ValueError("SeizureTransformer streaming result fields drifted")
    if data["schema_version"] != SEIZURETRANSFORMER_STREAMING_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported SeizureTransformer streaming result schema")
    if data["provider_id"] != SEIZURETRANSFORMER_PROVIDER_ID:
        raise ValueError("SeizureTransformer result provider drifted")
    if data["variant_id"] != data["tile_plan"].get("variant_id"):
        raise ValueError("SeizureTransformer result variant drifted")
    if data["method_id"] != SEIZURETRANSFORMER_STREAMING_METHOD_ID:
        raise ValueError("SeizureTransformer result method drifted")
    if (
        data["adapter_code_sha256"]
        != seizuretransformer_streaming_adapter_code_sha256()
    ):
        raise ValueError("SeizureTransformer result code binding drifted")
    plan = validate_seizuretransformer_tile_plan(data["tile_plan"])
    if data["physical_clock_receipt"] != plan["physical_clock_receipt"]:
        raise ValueError("result physical clock does not preserve the plan clock")

    if posterior.dtype != np.dtype("float32") or posterior.shape != (
        plan["sample_count"],
    ):
        raise ValueError("full-record posterior shape or dtype drifted")
    if (
        not np.isfinite(posterior).all()
        or np.any(posterior < 0.0)
        or np.any(posterior > 1.0)
    ):
        raise ValueError("full-record posterior is not a finite probability vector")
    expected_posterior_receipt = _payload_receipt(
        posterior, semantic="full_record_dense_seizure_probability"
    )
    if data["posterior_payload_receipt"] != expected_posterior_receipt:
        raise ValueError("full-record posterior payload receipt drifted")

    contributions = data["tile_contribution_receipts"]
    descriptors = plan["tile_descriptors"]
    if not isinstance(contributions, list) or len(contributions) != len(descriptors):
        raise ValueError("tile contribution receipt count drifted")
    contribution_required = {
        "tile_id",
        "tile_index",
        "absolute_observed_sample_range",
        "observed_sample_count",
        "right_padding_samples",
        "input_tile_payload_sha256",
        "dense_tile_probability_payload_receipt",
        "observed_tile_probability_payload_receipt",
        "effective_tile_weight_payload_receipt",
        "padding_probability_discarded_before_ola",
        "absolute_sample_alignment_used",
    }
    for index, (row, descriptor) in enumerate(zip(contributions, descriptors)):
        if type(row) is not dict or set(row) != contribution_required:
            raise ValueError("tile contribution fields drifted")
        if (
            row["tile_index"] != index
            or row["tile_id"] != descriptor["tile_id"]
            or row["absolute_observed_sample_range"]
            != descriptor["absolute_observed_sample_range"]
            or row["observed_sample_count"] != descriptor["observed_sample_count"]
            or row["right_padding_samples"] != descriptor["right_padding_samples"]
            or row["absolute_sample_alignment_used"] is not True
            or row["padding_probability_discarded_before_ola"]
            is not bool(descriptor["right_padding_samples"])
        ):
            raise ValueError("tile contribution does not bind its absolute ledger row")
        _require_sha256(row["input_tile_payload_sha256"], "tile input payload hash")
        expected_weight = _payload_receipt(
            seizuretransformer_tile_weights(descriptor),
            semantic="tile_ola_weight_including_padding_zero",
        )
        if row["effective_tile_weight_payload_receipt"] != expected_weight:
            raise ValueError("tile OLA weight receipt drifted")
        dense = row["dense_tile_probability_payload_receipt"]
        observed = row["observed_tile_probability_payload_receipt"]
        for receipt, shape, semantic in (
            (
                dense,
                [SEIZURETRANSFORMER_WINDOW_SAMPLES],
                "dense_tile_probability_including_discarded_padding",
            ),
            (
                observed,
                [descriptor["observed_sample_count"]],
                "dense_tile_probability_observed_prefix",
            ),
        ):
            if (
                type(receipt) is not dict
                or receipt.get("semantic") != semantic
                or receipt.get("dtype") != "float32_little_endian"
                or receipt.get("shape") != shape
                or not _is_sha256(receipt.get("payload_sha256"))
                or isinstance(receipt.get("minimum"), bool)
                or not isinstance(receipt.get("minimum"), (int, float))
                or isinstance(receipt.get("maximum"), bool)
                or not isinstance(receipt.get("maximum"), (int, float))
                or not 0.0
                <= float(receipt["minimum"])
                <= float(receipt["maximum"])
                <= 1.0
            ):
                raise ValueError("tile probability payload receipt is invalid")

    expected_weight_sum, expected_count = _expected_weight_coverage(plan)
    expected_coverage = _coverage_receipt(
        weight_sum=expected_weight_sum,
        contribution_count=expected_count,
        tile_count=len(descriptors),
    )
    if data["ola_coverage_receipt"] != expected_coverage:
        raise ValueError("full-record OLA coverage receipt drifted")
    if expected_coverage["complete_record_posterior_coverage"] is not True:
        raise ValueError("full-record posterior coverage is incomplete")
    _validate_execution_receipt(
        data["streaming_execution_receipt"],
        expected_tile_count=len(descriptors),
        expected_read_interval_roster_sha256=_canonical_sha256(
            [row["absolute_observed_sample_range"] for row in descriptors]
        ),
    )
    expected_permissions = {
        "cleanroom_checkpoint_required_but_verified_here": False,
        "qualified_operating_point_exists": False,
        "posterior_is_confirmed_seizure_or_clinical_onset": False,
        "posterior_is_findings_or_soz_evidence": False,
        "clinical_or_production_use_authorized": False,
    }
    if data["scientific_permissions"] != expected_permissions:
        raise ValueError("SeizureTransformer scientific permissions drifted")

    identity = deepcopy(data)
    identity["result_id"] = _RESULT_ID_PENDING
    identity["receipt_sha256"] = _CONTENT_PENDING
    expected_result_id = "ST-OLA-" + _canonical_sha256(identity)[:24]
    if data["result_id"] != expected_result_id:
        raise ValueError("SeizureTransformer result ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = _CONTENT_PENDING
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("SeizureTransformer result receipt is not content-bound")
    return data


__all__ = [
    "SEIZURETRANSFORMER_HOP_SAMPLES",
    "SEIZURETRANSFORMER_HOP_SECONDS",
    "SEIZURETRANSFORMER_PROVIDER_ID",
    "SEIZURETRANSFORMER_SAMPLING_RATE_HZ",
    "SEIZURETRANSFORMER_ST16_TYPED_UNITS",
    "SEIZURETRANSFORMER_ST16_VARIANT_ID",
    "SEIZURETRANSFORMER_ST18_TYPED_UNITS",
    "SEIZURETRANSFORMER_ST18_VARIANT_ID",
    "SEIZURETRANSFORMER_TYPED_UNITS",
    "SEIZURETRANSFORMER_WINDOW_SAMPLES",
    "SEIZURETRANSFORMER_WINDOW_SECONDS",
    "SeizureTransformerSampleAlignedOLA",
    "SeizureTransformerStreamingResult",
    "SeizureTransformerTile",
    "SeizureTransformerTileReader",
    "build_seizuretransformer_tile_plan",
    "iter_seizuretransformer_tiles",
    "run_seizuretransformer_streaming_ola",
    "seizuretransformer_streaming_adapter_code_sha256",
    "seizuretransformer_streaming_variant_registry",
    "seizuretransformer_tile_starts",
    "seizuretransformer_tile_weights",
    "validate_seizuretransformer_streaming_result",
    "validate_seizuretransformer_tile_plan",
]
