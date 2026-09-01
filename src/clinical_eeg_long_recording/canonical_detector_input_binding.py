"""Bind a checkpoint-native detector carrier to the canonical EEG root.

The detector and Findings branches intentionally use different transforms,
but they must originate from the same physical samples.  This module verifies
that a Standard-19 physical-electrode carrier (including explicit zero-filled
missing channels) is sample-wise equivalent to a ``CanonicalEEGRecord`` before
provider-specific filtering, resampling, normalization, or montage changes.

The receipt is provenance, not clinical evidence.  Imputed detector channels
remain ineligible for Findings and SOZ claims, and successful binding does not
qualify a detector, seizure, onset, or SOZ result.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.soz.geometry import STANDARD_19

from .canonical_edf_materialization import CanonicalEEGRecord
from .canonical_signal_views import validate_canonical_signal_receipt


CANONICAL_DETECTOR_INPUT_BINDING_SCHEMA_VERSION = (
    "canonical_standard19_detector_input_binding_v1"
)
CANONICAL_DETECTOR_INPUT_BINDING_METHOD_ID = (
    "samplewise_physical_carrier_equivalence_before_provider_transform_v1"
)
_UNIT_TO_VOLTS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty trimmed identifier")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _validate_source_header_binding(record: CanonicalEEGRecord) -> dict[str, Any]:
    header = deepcopy(record.source_header_receipt)
    canonical = validate_canonical_signal_receipt(record.canonical_receipt)
    if not isinstance(header, dict):
        raise TypeError("canonical source-header receipt must be an object")
    required = {
        "schema_version",
        "reader_policy",
        "source_signal_sha256",
        "source_tensor_sha256",
        "observed_channel_ids",
        "unobserved_channel_ids",
        "channel_signal_headers",
        "scope_receipt",
        "receipt_sha256",
    }
    if set(header) != required:
        raise ValueError("canonical source-header receipt fields drifted")
    digest = deepcopy(header)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if header["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("canonical source-header receipt is not content-bound")
    if header["source_signal_sha256"] != canonical["source_signal_sha256"]:
        raise ValueError("canonical source header and signal receipt disagree")
    if list(record.observed_channel_ids) != header["observed_channel_ids"]:
        raise ValueError("canonical record channel order disagrees with its header")
    if tuple(record.observed_signal_volts.shape[:1]) != (
        len(record.observed_channel_ids),
    ):
        raise ValueError("canonical observed tensor row count drifted")
    if record.observed_signal_volts.ndim != 2:
        raise ValueError("canonical observed signal tensor must be two-dimensional")
    if not torch.isfinite(record.observed_signal_volts).all():
        raise ValueError("canonical observed signal tensor contains nonfinite values")
    return {"header": header, "canonical": canonical}


def _detector_tensor_sha256(
    value: np.ndarray,
    *,
    channel_ids: Sequence[str],
    sampling_rate_hz: float,
    physical_unit: str,
) -> str:
    normalized = np.ascontiguousarray(value, dtype="<f8")
    descriptor = {
        "hash_domain": "detector-standard19-physical-carrier-float64-le-v1",
        "shape": list(normalized.shape),
        "channel_ids": list(channel_ids),
        "sampling_rate_hz": float(sampling_rate_hz),
        "physical_unit": physical_unit,
        "payload_sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
    }
    return _canonical_sha256(descriptor)


def build_canonical_detector_input_binding(
    *,
    canonical_record: CanonicalEEGRecord,
    provider_id: str,
    detector_input: object,
    detector_channel_ids: Sequence[str],
    detector_sampling_rate_hz: float,
    detector_physical_unit: str,
    observed_channel_ids: Sequence[str],
    imputed_channel_ids: Sequence[str],
    provider_input_receipt_id: str,
    provider_input_receipt_sha256: str,
    absolute_tolerance_volts: float = 1e-10,
    relative_tolerance: float = 2e-6,
) -> dict[str, Any]:
    """Verify and bind a pre-transform Standard-19 detector input carrier."""

    provider_id = _identifier(provider_id, "provider_id")
    provider_input_receipt_id = _identifier(
        provider_input_receipt_id, "provider_input_receipt_id"
    )
    if not _is_sha256(provider_input_receipt_sha256):
        raise ValueError("provider input receipt SHA-256 is invalid")
    if tuple(detector_channel_ids) != tuple(STANDARD_19):
        raise ValueError("detector physical carrier must use exact STANDARD_19 order")
    if detector_physical_unit not in _UNIT_TO_VOLTS:
        raise ValueError("detector physical unit is unsupported")
    sampling_rate = _finite(
        detector_sampling_rate_hz, "detector_sampling_rate_hz"
    )
    if sampling_rate <= 0:
        raise ValueError("detector sampling rate must be positive")
    absolute_tolerance = _finite(
        absolute_tolerance_volts, "absolute_tolerance_volts"
    )
    relative = _finite(relative_tolerance, "relative_tolerance")
    if absolute_tolerance < 0 or relative < 0:
        raise ValueError("detector equivalence tolerances must be non-negative")

    if isinstance(detector_input, torch.Tensor):
        array = detector_input.detach().cpu().numpy()
    elif isinstance(detector_input, np.ndarray):
        array = detector_input
    else:
        raise TypeError("detector_input must be a NumPy array or torch Tensor")
    detector = np.asarray(array, dtype=np.float64)
    if detector.ndim != 2 or detector.shape[0] != len(STANDARD_19):
        raise ValueError("detector input must have shape [19, samples]")
    if detector.shape[1] < 1 or not np.isfinite(detector).all():
        raise ValueError("detector input contains no finite physical samples")

    record_binding = _validate_source_header_binding(canonical_record)
    header = record_binding["header"]
    canonical = record_binding["canonical"]
    observed = tuple(str(value) for value in observed_channel_ids)
    imputed = tuple(str(value) for value in imputed_channel_ids)
    expected_observed = tuple(canonical_record.observed_channel_ids)
    expected_imputed = tuple(
        channel for channel in STANDARD_19 if channel not in set(expected_observed)
    )
    if observed != expected_observed or imputed != expected_imputed:
        raise ValueError("detector observed/imputed partition disagrees with canonical EEG")

    source_headers = list(header["channel_signal_headers"])
    rates = {
        (
            int(row["sampling_rate_numerator"]),
            int(row["sampling_rate_denominator"]),
        )
        for row in source_headers
    }
    sample_counts = {int(row["sample_count"]) for row in source_headers}
    if len(rates) != 1 or len(sample_counts) != 1:
        raise ValueError("canonical detector carrier requires one physical sampling grid")
    rate_numerator, rate_denominator = next(iter(rates))
    canonical_rate = rate_numerator / rate_denominator
    sample_count = next(iter(sample_counts))
    if abs(canonical_rate - sampling_rate) > 1e-9:
        raise ValueError("detector and canonical sampling rates disagree")
    if detector.shape[1] != sample_count:
        raise ValueError("detector and canonical sample counts disagree")

    detector_volts = detector * _UNIT_TO_VOLTS[detector_physical_unit]
    canonical_volts = (
        canonical_record.observed_signal_volts.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    detector_index = {name: index for index, name in enumerate(STANDARD_19)}
    observed_detector = np.stack(
        [detector_volts[detector_index[name]] for name in observed]
    )
    difference = np.abs(observed_detector - canonical_volts)
    tolerance = absolute_tolerance + relative * np.abs(canonical_volts)
    if np.any(difference > tolerance):
        raise ValueError("detector physical samples are not canonical-equivalent")
    imputed_nonzero_count = int(
        np.count_nonzero(
            np.stack([detector[detector_index[name]] for name in imputed])
        )
        if imputed
        else 0
    )
    if imputed_nonzero_count:
        raise ValueError("detector imputed physical channels must be exact zero carriers")

    duration = sample_count / canonical_rate
    body: dict[str, Any] = {
        "schema_version": CANONICAL_DETECTOR_INPUT_BINDING_SCHEMA_VERSION,
        "binding_id": "CANONICAL-DETECTOR-INPUT-PENDING",
        "method_id": CANONICAL_DETECTOR_INPUT_BINDING_METHOD_ID,
        "provider_id": provider_id,
        "provider_input_receipt_id": provider_input_receipt_id,
        "provider_input_receipt_sha256": provider_input_receipt_sha256,
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_source_signal_sha256": canonical["source_signal_sha256"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "canonical_source_header_receipt_sha256": header["receipt_sha256"],
        "canonical_source_tensor_sha256": header["source_tensor_sha256"],
        "detector_input_tensor_sha256": _detector_tensor_sha256(
            detector,
            channel_ids=detector_channel_ids,
            sampling_rate_hz=sampling_rate,
            physical_unit=detector_physical_unit,
        ),
        "detector_channel_ids": list(detector_channel_ids),
        "observed_channel_ids": list(observed),
        "imputed_channel_ids": list(imputed),
        "detector_physical_unit": detector_physical_unit,
        "sampling_rate_numerator": rate_numerator,
        "sampling_rate_denominator": rate_denominator,
        "sample_count": sample_count,
        "recording_duration_seconds": duration,
        "equivalence_receipt": {
            "samplewise_observed_equivalence_verified": True,
            "absolute_tolerance_volts": absolute_tolerance,
            "relative_tolerance": relative,
            "maximum_absolute_error_volts": float(difference.max(initial=0.0)),
            "root_mean_square_error_volts": float(
                np.sqrt(np.mean(np.square(difference)))
            ),
            "imputed_channels_exact_zero_verified": True,
            "imputed_nonzero_sample_count": imputed_nonzero_count,
            "comparison_before_provider_filter_resample_normalization": True,
        },
        "evidence_permissions": {
            "detector_input_is_navigation_only": True,
            "imputed_channels_clinical_evidence_eligible": False,
            "binding_qualifies_detector_or_clinical_term": False,
        },
        "scope_receipt": {
            "eeg_samples_used": True,
            "edf_signal_header_used": True,
            "edf_annotations_used": False,
            "spreadsheet_used": False,
            "doctor_labels_used": False,
            "clinical_text_used": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["binding_id"] = "CANDET-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_canonical_detector_input_binding(body)


def validate_canonical_detector_input_binding(payload: object) -> dict[str, Any]:
    """Validate the content and fail-closed permission semantics of a binding."""

    if type(payload) is not dict:
        raise TypeError("canonical detector input binding must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "binding_id",
        "method_id",
        "provider_id",
        "provider_input_receipt_id",
        "provider_input_receipt_sha256",
        "canonical_signal_id",
        "canonical_source_signal_sha256",
        "canonical_receipt_sha256",
        "canonical_source_header_receipt_sha256",
        "canonical_source_tensor_sha256",
        "detector_input_tensor_sha256",
        "detector_channel_ids",
        "observed_channel_ids",
        "imputed_channel_ids",
        "detector_physical_unit",
        "sampling_rate_numerator",
        "sampling_rate_denominator",
        "sample_count",
        "recording_duration_seconds",
        "equivalence_receipt",
        "evidence_permissions",
        "scope_receipt",
        "receipt_sha256",
    }
    if set(data) != required:
        raise ValueError("canonical detector input binding fields drifted")
    if data["schema_version"] != CANONICAL_DETECTOR_INPUT_BINDING_SCHEMA_VERSION:
        raise ValueError("canonical detector input binding schema drifted")
    if data["method_id"] != CANONICAL_DETECTOR_INPUT_BINDING_METHOD_ID:
        raise ValueError("canonical detector input binding method drifted")
    for name in ("binding_id", "provider_id", "provider_input_receipt_id"):
        _identifier(data[name], name)
    for name in (
        "provider_input_receipt_sha256",
        "canonical_source_signal_sha256",
        "canonical_receipt_sha256",
        "canonical_source_header_receipt_sha256",
        "canonical_source_tensor_sha256",
        "detector_input_tensor_sha256",
        "receipt_sha256",
    ):
        if not _is_sha256(data[name]):
            raise ValueError(f"canonical detector input {name} is invalid")
    if data["detector_channel_ids"] != list(STANDARD_19):
        raise ValueError("canonical detector channel order drifted")
    observed = data["observed_channel_ids"]
    imputed = data["imputed_channel_ids"]
    if (
        not isinstance(observed, list)
        or not isinstance(imputed, list)
        or [item for item in STANDARD_19 if item in set(observed)] != observed
        or [item for item in STANDARD_19 if item in set(imputed)] != imputed
        or set(observed).intersection(imputed)
        or set(observed).union(imputed) != set(STANDARD_19)
    ):
        raise ValueError("canonical detector observed/imputed partition is invalid")
    if data["detector_physical_unit"] not in _UNIT_TO_VOLTS:
        raise ValueError("canonical detector physical unit drifted")
    if (
        isinstance(data["sampling_rate_numerator"], bool)
        or not isinstance(data["sampling_rate_numerator"], int)
        or data["sampling_rate_numerator"] <= 0
        or isinstance(data["sampling_rate_denominator"], bool)
        or not isinstance(data["sampling_rate_denominator"], int)
        or data["sampling_rate_denominator"] <= 0
        or isinstance(data["sample_count"], bool)
        or not isinstance(data["sample_count"], int)
        or data["sample_count"] <= 0
    ):
        raise ValueError("canonical detector physical sampling grid is invalid")
    expected_duration = data["sample_count"] / (
        data["sampling_rate_numerator"] / data["sampling_rate_denominator"]
    )
    if abs(_finite(data["recording_duration_seconds"], "duration") - expected_duration) > 1e-9:
        raise ValueError("canonical detector duration disagrees with sampling grid")
    equivalence = data["equivalence_receipt"]
    expected_equivalence_bools = {
        "samplewise_observed_equivalence_verified": True,
        "imputed_channels_exact_zero_verified": True,
        "comparison_before_provider_filter_resample_normalization": True,
    }
    if not isinstance(equivalence, Mapping) or any(
        equivalence.get(name) is not expected
        for name, expected in expected_equivalence_bools.items()
    ):
        raise ValueError("canonical detector equivalence receipt is incomplete")
    if equivalence.get("imputed_nonzero_sample_count") != 0:
        raise ValueError("canonical detector binding contains nonzero imputation")
    for name in (
        "absolute_tolerance_volts",
        "relative_tolerance",
        "maximum_absolute_error_volts",
        "root_mean_square_error_volts",
    ):
        if _finite(equivalence.get(name), name) < 0:
            raise ValueError("canonical detector equivalence error is invalid")
    expected_permissions = {
        "detector_input_is_navigation_only": True,
        "imputed_channels_clinical_evidence_eligible": False,
        "binding_qualifies_detector_or_clinical_term": False,
    }
    if data["evidence_permissions"] != expected_permissions:
        raise ValueError("canonical detector evidence permissions drifted")
    expected_scope = {
        "eeg_samples_used": True,
        "edf_signal_header_used": True,
        "edf_annotations_used": False,
        "spreadsheet_used": False,
        "doctor_labels_used": False,
        "clinical_text_used": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("canonical detector input binding violates EEG-only scope")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("canonical detector input receipt is not content-bound")
    id_source = deepcopy(data)
    id_source["binding_id"] = "CANONICAL-DETECTOR-INPUT-PENDING"
    id_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["binding_id"] != "CANDET-" + _canonical_sha256(id_source)[:24]:
        raise ValueError("canonical detector input binding ID is not content-bound")
    return data


__all__ = [
    "CANONICAL_DETECTOR_INPUT_BINDING_METHOD_ID",
    "CANONICAL_DETECTOR_INPUT_BINDING_SCHEMA_VERSION",
    "build_canonical_detector_input_binding",
    "validate_canonical_detector_input_binding",
]
