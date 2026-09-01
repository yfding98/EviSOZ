"""Post-freeze public-TUSZ interval targets for BA-IEG segmental training.

This module has two intentionally ordered capabilities.  First, an already
tokenized ``BAIEGEventTokens`` object is sealed with its detector candidate,
adaptive envelope and target-independent candidate-roster receipts.  This
returns an opaque capability containing no reference labels.  Only that exact
validated capability may then be joined to a content-bound public TUSZ seizure
interval sidecar.

The join never changes candidate selection, analysis windows or tokenization.
Complete reference coverage can yield absent, one-bout or recurrent ``2+``
partial labels.  Incomplete coverage always yields ``not_evaluable`` and is
never interpreted as absence.  Quantized boundary timestamps become closed
resolution intervals; boundaries outside the acquired event window become
left/right censoring rather than clipped point targets.

This module deliberately does not quantize an annotation interval to model
tokens or lattice edges.  The raw resolution interval and its content-bound
target receipt survive unchanged until the target-free segmental forward has
frozen a physical-time lattice; only the supervision module may then derive a
separate replayable support projection.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_permission_split_segmental_supervision_v1 import (
    BAIEGSegmentalEventTargetV1,
    BAIEGSegmentalTargetFirewallV1,
)
from .ba_ieg_training_contract import BAIEGEventTokens


BA_IEG_TUSZ_CANDIDATE_ENVELOPE_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_tusz_target_independent_candidate_envelope_v1"
BA_IEG_TUSZ_PUBLIC_INTERVAL_REFERENCE_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_tusz_public_seizure_interval_reference_v1"
BA_IEG_TUSZ_INTERVAL_TARGET_MATERIALIZATION_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_tusz_candidate_interval_target_materialization_v1"
BA_IEG_TUSZ_INTERVAL_TARGET_MATERIALIZATION_METHOD_ID: Final[
    str
] = "post_tokenization_complete_reference_interval_censor_join_v1"

_ALLOWED_SPLITS: Final[frozenset[str]] = frozenset({"source_train", "source_dev"})
_ALLOWED_REFERENCE_FORMATS: Final[frozenset[str]] = frozenset(
    {"tusz_tse", "tusz_csv_bi", "tusz_lbl_sidecar_export"}
)
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TIME_TOLERANCE_SECONDS: Final[float] = 1e-6

_CANDIDATE_SCOPE: Final[dict[str, object]] = {
    "freeze_stage": ("after_target_free_window_and_tokenization_before_reference_join"),
    "detector_candidate_target_independent": True,
    "analysis_envelope_target_independent": True,
    "window_or_tokenization_target_conditioned": False,
    "reference_sidecar_read_before_freeze": False,
    "edf_embedded_annotation_used": False,
    "excel_used": False,
    "doctor_label_or_clinical_text_used": False,
    "private_source_used": False,
}

_REFERENCE_SCOPE: Final[dict[str, object]] = {
    "implementation_status": "public_reference_contract_shadow",
    "public_tusz_reference_only": True,
    "reference_join_stage": "post_candidate_window_tokenization_freeze_only",
    "candidate_selection_or_windowing_from_reference": False,
    "edf_embedded_annotation_used": False,
    "excel_used": False,
    "doctor_label_or_clinical_text_used": False,
    "private_source_used": False,
    "reference_available_to_model_forward": False,
    "source_artifact_bytes_opened_by_this_builder": False,
    "host_reference_authority_claimed": False,
}

_MATERIALIZATION_SCOPE: Final[dict[str, object]] = {
    "implementation_status": "contract_shadow_no_real_data_materialized",
    "candidate_capability_and_source_event_replayed_at_join": True,
    "candidate_capability_required_for_receipt_replay": True,
    "reference_used_for_supervision_only": True,
    "reference_used_for_candidate_selection": False,
    "reference_used_for_window_or_tokenization": False,
    "incomplete_reference_coverage_is_absent": False,
    "source_artifact_bytes_opened_by_this_module": False,
    "host_reference_authority_claimed": False,
    "training_or_calibration_executed": False,
    "private_or_clinical_route_authorized": False,
    "target_available_to_model_forward": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _positive_interval(value: object, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-number array")
    start, stop = (_finite(item, context) for item in value)
    if stop <= start:
        raise ValueError(f"{context} must have positive duration")
    return [start, stop]


@dataclass(frozen=True, slots=True)
class ValidatedBAIEGTUSZCandidateEnvelopeV1:
    """Opaque, reference-free capability accepted by the target join."""

    _canonical_payload_json: str
    _source_event: BAIEGEventTokens = field(repr=False, compare=False)

    def payload(self) -> dict[str, Any]:
        value = json.loads(self._canonical_payload_json)
        if type(value) is not dict:
            raise RuntimeError("candidate-envelope capability payload corrupted")
        return value


def freeze_ba_ieg_tusz_candidate_envelope_after_tokenization_v1(
    event: BAIEGEventTokens,
    *,
    detector_candidate_id: str,
    detector_candidate_receipt_sha256: str,
    adaptive_envelope_receipt_sha256: str,
    adaptive_acquisition_receipt_sha256: str,
    target_independent_candidate_roster_receipt_sha256: str,
    detector_candidate_support_interval_recording_seconds: Sequence[float],
) -> ValidatedBAIEGTUSZCandidateEnvelopeV1:
    """Seal a target-free candidate only after windowing and tokenization."""

    if not isinstance(event, BAIEGEventTokens):
        raise TypeError("candidate-envelope freeze requires BAIEGEventTokens")
    event.verify_integrity()
    if event.model_split not in _ALLOWED_SPLITS:
        raise ValueError(
            "TUSZ target materialization accepts source_train/source_dev only"
        )
    candidate_id = _identifier(detector_candidate_id, "detector_candidate_id")
    hashes = {
        "detector_candidate_receipt_sha256": _sha256(
            detector_candidate_receipt_sha256,
            "detector_candidate_receipt_sha256",
        ),
        "adaptive_envelope_receipt_sha256": _sha256(
            adaptive_envelope_receipt_sha256,
            "adaptive_envelope_receipt_sha256",
        ),
        "adaptive_acquisition_receipt_sha256": _sha256(
            adaptive_acquisition_receipt_sha256,
            "adaptive_acquisition_receipt_sha256",
        ),
        "target_independent_candidate_roster_receipt_sha256": _sha256(
            target_independent_candidate_roster_receipt_sha256,
            "target_independent_candidate_roster_receipt_sha256",
        ),
    }
    support = [
        _finite(value, "detector candidate support")
        for value in detector_candidate_support_interval_recording_seconds
    ]
    if len(support) != 2 or support[1] <= support[0]:
        raise ValueError("detector candidate support must be a positive interval")
    analysis = list(event.analysis_interval_seconds)
    if (
        support[0] < analysis[0] - _TIME_TOLERANCE_SECONDS
        or support[1] > analysis[1] + _TIME_TOLERANCE_SECONDS
    ):
        raise ValueError("detector candidate support lies outside frozen event input")
    body: dict[str, Any] = {
        "schema_version": BA_IEG_TUSZ_CANDIDATE_ENVELOPE_SCHEMA_VERSION,
        "receipt_id": "BAIEG-TUSZ-CANDIDATE-ENVELOPE-PENDING",
        "event_id": event.event_id,
        "recording_id": event.recording_id,
        "patient_uid": event.patient_uid,
        "model_split": event.model_split,
        "input_event_receipt_sha256": event.input_receipt_sha256,
        "adaptive_window_receipt_sha256": event.adaptive_window_receipt_sha256,
        "detector_candidate_id": candidate_id,
        **hashes,
        "analysis_interval_recording_seconds": analysis,
        "navigation_anchor_recording_seconds": event.navigation_anchor_seconds,
        "detector_candidate_support_interval_recording_seconds": support,
        "scope_receipt": deepcopy(_CANDIDATE_SCOPE),
    }
    body["receipt_id"] = "BAIEGTUSZCAND-" + _canonical_sha256(body)[:24]
    return validate_ba_ieg_tusz_candidate_envelope_v1(body, event=event)


def validate_ba_ieg_tusz_candidate_envelope_v1(
    payload: object,
    *,
    event: BAIEGEventTokens,
) -> ValidatedBAIEGTUSZCandidateEnvelopeV1:
    required = {
        "schema_version",
        "receipt_id",
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "input_event_receipt_sha256",
        "adaptive_window_receipt_sha256",
        "detector_candidate_id",
        "detector_candidate_receipt_sha256",
        "adaptive_envelope_receipt_sha256",
        "adaptive_acquisition_receipt_sha256",
        "target_independent_candidate_roster_receipt_sha256",
        "analysis_interval_recording_seconds",
        "navigation_anchor_recording_seconds",
        "detector_candidate_support_interval_recording_seconds",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("TUSZ candidate-envelope receipt has invalid fields")
    if not isinstance(event, BAIEGEventTokens):
        raise TypeError("candidate-envelope validation requires BAIEGEventTokens")
    event.verify_integrity()
    data = deepcopy(payload)
    canonical_before_normalization = _canonical_json(data)
    if data["schema_version"] != BA_IEG_TUSZ_CANDIDATE_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("TUSZ candidate-envelope schema drifted")
    for name in (
        "receipt_id",
        "event_id",
        "recording_id",
        "patient_uid",
        "detector_candidate_id",
    ):
        _identifier(data[name], name)
    if data["model_split"] not in _ALLOWED_SPLITS:
        raise ValueError("candidate-envelope split is not source_train/source_dev")
    for name in (
        "input_event_receipt_sha256",
        "adaptive_window_receipt_sha256",
        "detector_candidate_receipt_sha256",
        "adaptive_envelope_receipt_sha256",
        "adaptive_acquisition_receipt_sha256",
        "target_independent_candidate_roster_receipt_sha256",
    ):
        _sha256(data[name], name)
    analysis = _positive_interval(
        data["analysis_interval_recording_seconds"], "analysis interval"
    )
    support = _positive_interval(
        data["detector_candidate_support_interval_recording_seconds"],
        "candidate support",
    )
    anchor = _finite(data["navigation_anchor_recording_seconds"], "navigation anchor")
    if (
        not analysis[0] <= anchor <= analysis[1]
        or support[0] < analysis[0] - _TIME_TOLERANCE_SECONDS
        or support[1] > analysis[1] + _TIME_TOLERANCE_SECONDS
    ):
        raise ValueError("candidate-envelope recording clocks do not close")
    data["analysis_interval_recording_seconds"] = analysis
    data["detector_candidate_support_interval_recording_seconds"] = support
    data["navigation_anchor_recording_seconds"] = anchor
    if _canonical_json(data) != canonical_before_normalization:
        raise ValueError("candidate-envelope receipt is not canonical")
    if data["scope_receipt"] != _CANDIDATE_SCOPE:
        raise ValueError("candidate-envelope target-independence firewall drifted")
    expected_event_binding = {
        "event_id": event.event_id,
        "recording_id": event.recording_id,
        "patient_uid": event.patient_uid,
        "model_split": event.model_split,
        "input_event_receipt_sha256": event.input_receipt_sha256,
        "adaptive_window_receipt_sha256": event.adaptive_window_receipt_sha256,
        "analysis_interval_recording_seconds": list(event.analysis_interval_seconds),
        "navigation_anchor_recording_seconds": event.navigation_anchor_seconds,
    }
    if any(data[name] != value for name, value in expected_event_binding.items()):
        raise ValueError("candidate-envelope does not replay its frozen event input")
    digest = deepcopy(data)
    digest["receipt_id"] = "BAIEG-TUSZ-CANDIDATE-ENVELOPE-PENDING"
    if data["receipt_id"] != "BAIEGTUSZCAND-" + _canonical_sha256(digest)[:24]:
        raise ValueError("candidate-envelope receipt is not content-bound")
    return ValidatedBAIEGTUSZCandidateEnvelopeV1(_canonical_json(data), event)


@dataclass(frozen=True, slots=True)
class ValidatedBAIEGTUSZPublicIntervalReferenceV1:
    """Opaque, content-bound public-TUSZ interval-reference capability."""

    _canonical_payload_json: str

    def payload(self) -> dict[str, Any]:
        value = json.loads(self._canonical_payload_json)
        if type(value) is not dict:
            raise RuntimeError("public-TUSZ reference capability payload corrupted")
        return value


def _public_tusz_dataset_id(value: object) -> str:
    dataset_id = _identifier(value, "source_dataset_id")
    if dataset_id != "TUSZ" and not dataset_id.startswith("TUSZ-PUBLIC-"):
        raise ValueError(
            "interval references must come from an explicit public TUSZ dataset"
        )
    return dataset_id


def _recording_interval(
    value: object,
    *,
    recording_duration_seconds: float,
    context: str,
) -> list[float]:
    interval = _positive_interval(value, context)
    if interval[0] < 0.0 or interval[1] > recording_duration_seconds:
        raise ValueError(f"{context} lies outside the recording clock")
    return interval


def _normalize_covered_intervals(
    value: object,
    *,
    recording_duration_seconds: float,
) -> list[list[float]]:
    if type(value) is not list:
        raise TypeError("covered_recording_intervals_seconds must be an array")
    normalized: list[list[float]] = []
    previous_stop: float | None = None
    for index, raw in enumerate(value):
        interval = _recording_interval(
            raw,
            recording_duration_seconds=recording_duration_seconds,
            context=f"covered interval {index}",
        )
        if previous_stop is not None and interval[0] <= previous_stop:
            raise ValueError(
                "covered recording intervals must be sorted, non-overlapping "
                "and non-adjacent"
            )
        normalized.append(interval)
        previous_stop = interval[1]
    return normalized


def _interval_is_covered(
    interval: Sequence[float], covered: Sequence[Sequence[float]]
) -> bool:
    return any(
        interval[0] >= item[0] - _TIME_TOLERANCE_SECONDS
        and interval[1] <= item[1] + _TIME_TOLERANCE_SECONDS
        for item in covered
    )


def _normalize_seizure_intervals(
    value: object,
    *,
    recording_duration_seconds: float,
    covered_intervals: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise TypeError("seizure_intervals must be an array")
    required = {
        "public_event_id",
        "onset_recording_seconds",
        "offset_recording_seconds",
    }
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_stop: float | None = None
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != required:
            raise ValueError(f"seizure interval {index} has invalid fields")
        event_id = _identifier(raw["public_event_id"], "public_event_id")
        if event_id in seen_ids:
            raise ValueError("public seizure event IDs must be unique")
        seen_ids.add(event_id)
        interval = _recording_interval(
            [raw["onset_recording_seconds"], raw["offset_recording_seconds"]],
            recording_duration_seconds=recording_duration_seconds,
            context=f"seizure interval {index}",
        )
        if previous_stop is not None and interval[0] < previous_stop:
            raise ValueError("seizure intervals must be sorted and non-overlapping")
        if not _interval_is_covered(interval, covered_intervals):
            raise ValueError(
                "a seizure interval lies outside declared reference coverage"
            )
        normalized.append(
            {
                "public_event_id": event_id,
                "onset_recording_seconds": interval[0],
                "offset_recording_seconds": interval[1],
            }
        )
        previous_stop = interval[1]
    return normalized


def _reference_semantic_digest_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    digest = deepcopy(dict(payload))
    digest["receipt_id"] = "BAIEG-TUSZ-REFERENCE-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    return digest


def _seal_reference_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    semantic = _reference_semantic_digest_source(result)
    result["receipt_id"] = "BAIEGTUSZREF-" + _canonical_sha256(semantic)[:24]
    hash_source = deepcopy(result)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = _canonical_sha256(hash_source)
    return result


def _normalize_public_reference_semantics(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    data = deepcopy(dict(payload))
    if data["schema_version"] != BA_IEG_TUSZ_PUBLIC_INTERVAL_REFERENCE_SCHEMA_VERSION:
        raise ValueError("public-TUSZ interval-reference schema drifted")
    if data["reference_materialization_status"] != "complete":
        raise ValueError(
            "partial or failed public-reference materialization is inadmissible"
        )
    data["source_dataset_id"] = _public_tusz_dataset_id(data["source_dataset_id"])
    for name in (
        "source_dataset_version",
        "patient_uid",
        "recording_id",
        "source_reference_artifact_id",
    ):
        data[name] = _identifier(data[name], name)
    if data["model_split"] not in _ALLOWED_SPLITS:
        raise ValueError("public-TUSZ references are source_train/source_dev only")
    data["source_reference_artifact_sha256"] = _sha256(
        data["source_reference_artifact_sha256"],
        "source_reference_artifact_sha256",
    )
    if data["source_format"] not in _ALLOWED_REFERENCE_FORMATS:
        raise ValueError("public-TUSZ reference format is unsupported")
    duration = _finite(data["recording_duration_seconds"], "recording duration")
    if duration <= 0.0:
        raise ValueError("recording duration must be positive")
    resolution = _finite(
        data["annotation_timestamp_resolution_seconds"],
        "annotation timestamp resolution",
    )
    if resolution <= 0.0:
        raise ValueError("annotation timestamp resolution must be positive")
    if data["resolution_semantics"] != "centered_closed_interval_half_resolution":
        raise ValueError("annotation timestamp-resolution semantics drifted")
    if data["reference_coverage_status"] not in {
        "complete_recording",
        "incomplete",
    }:
        raise ValueError("reference coverage status is unsupported")
    covered = _normalize_covered_intervals(
        data["covered_recording_intervals_seconds"],
        recording_duration_seconds=duration,
    )
    if data["reference_coverage_status"] == "complete_recording" and covered != [
        [0.0, duration]
    ]:
        raise ValueError("complete reference coverage must exactly span the recording")
    intervals = _normalize_seizure_intervals(
        data["seizure_intervals"],
        recording_duration_seconds=duration,
        covered_intervals=covered,
    )
    if data["scope_receipt"] != _REFERENCE_SCOPE:
        raise ValueError("public-TUSZ reference firewall drifted")
    data["recording_duration_seconds"] = duration
    data["annotation_timestamp_resolution_seconds"] = resolution
    data["covered_recording_intervals_seconds"] = covered
    data["seizure_intervals"] = intervals
    return data


def build_ba_ieg_tusz_public_interval_reference_v1(
    *,
    source_dataset_id: str,
    source_dataset_version: str,
    patient_uid: str,
    recording_id: str,
    model_split: str,
    recording_duration_seconds: float,
    source_reference_artifact_id: str,
    source_reference_artifact_sha256: str,
    source_format: str,
    annotation_timestamp_resolution_seconds: float,
    reference_coverage_status: str,
    covered_recording_intervals_seconds: Sequence[Sequence[float]],
    seizure_intervals: Sequence[Mapping[str, Any]],
) -> ValidatedBAIEGTUSZPublicIntervalReferenceV1:
    """Build a parsed public-reference sidecar without opening EEG inputs."""

    body: dict[str, Any] = {
        "schema_version": BA_IEG_TUSZ_PUBLIC_INTERVAL_REFERENCE_SCHEMA_VERSION,
        "receipt_id": "BAIEG-TUSZ-REFERENCE-PENDING",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "reference_materialization_status": "complete",
        "source_dataset_id": source_dataset_id,
        "source_dataset_version": source_dataset_version,
        "patient_uid": patient_uid,
        "recording_id": recording_id,
        "model_split": model_split,
        "recording_duration_seconds": recording_duration_seconds,
        "source_reference_artifact_id": source_reference_artifact_id,
        "source_reference_artifact_sha256": source_reference_artifact_sha256,
        "source_format": source_format,
        "annotation_timestamp_resolution_seconds": (
            annotation_timestamp_resolution_seconds
        ),
        "resolution_semantics": "centered_closed_interval_half_resolution",
        "reference_coverage_status": reference_coverage_status,
        "covered_recording_intervals_seconds": [
            list(interval) for interval in covered_recording_intervals_seconds
        ],
        "seizure_intervals": [dict(interval) for interval in seizure_intervals],
        "scope_receipt": deepcopy(_REFERENCE_SCOPE),
    }
    body = _normalize_public_reference_semantics(body)
    return validate_ba_ieg_tusz_public_interval_reference_v1(
        _seal_reference_receipt(body)
    )


def validate_ba_ieg_tusz_public_interval_reference_v1(
    payload: object,
) -> ValidatedBAIEGTUSZPublicIntervalReferenceV1:
    required = {
        "schema_version",
        "receipt_id",
        "receipt_sha256",
        "reference_materialization_status",
        "source_dataset_id",
        "source_dataset_version",
        "patient_uid",
        "recording_id",
        "model_split",
        "recording_duration_seconds",
        "source_reference_artifact_id",
        "source_reference_artifact_sha256",
        "source_format",
        "annotation_timestamp_resolution_seconds",
        "resolution_semantics",
        "reference_coverage_status",
        "covered_recording_intervals_seconds",
        "seizure_intervals",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("public-TUSZ interval reference has invalid fields")
    raw = deepcopy(payload)
    _identifier(raw["receipt_id"], "receipt_id")
    _sha256(raw["receipt_sha256"], "receipt_sha256")
    data = _normalize_public_reference_semantics(raw)
    raw_semantics = deepcopy(raw)
    normalized_semantics = deepcopy(data)
    for value in (raw_semantics, normalized_semantics):
        value.pop("receipt_id")
        value.pop("receipt_sha256")
    if _canonical_json(raw_semantics) != _canonical_json(normalized_semantics):
        raise ValueError("public-TUSZ interval reference is not canonical")
    expected_id = (
        "BAIEGTUSZREF-"
        + _canonical_sha256(_reference_semantic_digest_source(data))[:24]
    )
    if data["receipt_id"] != expected_id:
        raise ValueError("public-TUSZ reference receipt ID is not content-bound")
    hash_source = deepcopy(data)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(hash_source):
        raise ValueError("public-TUSZ reference receipt hash is not content-bound")
    return ValidatedBAIEGTUSZPublicIntervalReferenceV1(_canonical_json(data))


def _positive_duration_overlap(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _matching_public_bouts(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> list[dict[str, Any]]:
    analysis = candidate["analysis_interval_recording_seconds"]
    return [
        deepcopy(row)
        for row in reference["seizure_intervals"]
        if _positive_duration_overlap(
            analysis,
            [
                row["onset_recording_seconds"],
                row["offset_recording_seconds"],
            ],
        )
        > 0.0
    ]


def _annotation_resolution_interval(
    boundary_recording_seconds: float,
    *,
    resolution_seconds: float,
    recording_duration_seconds: float,
) -> list[float]:
    half_resolution = 0.5 * resolution_seconds
    return [
        max(0.0, boundary_recording_seconds - half_resolution),
        min(
            recording_duration_seconds,
            boundary_recording_seconds + half_resolution,
        ),
    ]


def _derive_target_and_trace(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> tuple[BAIEGSegmentalEventTargetV1, dict[str, Any]]:
    analysis = candidate["analysis_interval_recording_seconds"]
    duration = reference["recording_duration_seconds"]
    if analysis[0] < 0.0 or analysis[1] > duration:
        raise ValueError(
            "frozen candidate analysis interval lies outside reference recording"
        )

    coverage = reference["reference_coverage_status"]
    matched: list[dict[str, Any]] = []
    onset_boundary: float | None = None
    offset_boundary: float | None = None
    onset_resolution_interval: list[float] | None = None
    offset_resolution_interval: list[float] | None = None
    onset_interval: tuple[float, float] | None = None
    offset_interval: tuple[float, float] | None = None

    if coverage != "complete_recording":
        event_status = "not_evaluable"
        onset_status = "not_evaluable"
        offset_status = "not_evaluable"
        bout_count_status = "not_evaluable"
        matching_status = "not_performed_incomplete_reference_coverage"
        matched_count: int | None = None
    else:
        matched = _matching_public_bouts(candidate, reference)
        matched_count = len(matched)
        matching_status = "performed_complete_recording_reference"
        if matched_count == 0:
            event_status = "absent"
            onset_status = "not_observed"
            offset_status = "not_observed"
            bout_count_status = "zero_bouts"
        elif matched_count >= 2:
            event_status = "present"
            onset_status = "not_evaluable"
            offset_status = "not_evaluable"
            bout_count_status = "two_or_more_bouts"
        else:
            event_status = "present"
            bout_count_status = "single_bout"
            onset_boundary = matched[0]["onset_recording_seconds"]
            offset_boundary = matched[0]["offset_recording_seconds"]
            resolution = reference["annotation_timestamp_resolution_seconds"]
            onset_resolution_interval = _annotation_resolution_interval(
                onset_boundary,
                resolution_seconds=resolution,
                recording_duration_seconds=duration,
            )
            offset_resolution_interval = _annotation_resolution_interval(
                offset_boundary,
                resolution_seconds=resolution,
                recording_duration_seconds=duration,
            )
            if onset_resolution_interval[0] <= (analysis[0] + _TIME_TOLERANCE_SECONDS):
                onset_status = "left_censored"
            else:
                onset_status = "observed_interval"
                onset_interval = tuple(onset_resolution_interval)
            if offset_resolution_interval[1] >= (analysis[1] - _TIME_TOLERANCE_SECONDS):
                offset_status = "right_censored"
            else:
                offset_status = "observed_interval"
                offset_interval = tuple(offset_resolution_interval)

    target = BAIEGSegmentalEventTargetV1(
        event_id=candidate["event_id"],
        recording_id=candidate["recording_id"],
        patient_uid=candidate["patient_uid"],
        model_split=candidate["model_split"],
        source_event_receipt_sha256=candidate["input_event_receipt_sha256"],
        adaptive_acquisition_receipt_sha256=candidate[
            "adaptive_acquisition_receipt_sha256"
        ],
        target_independent_candidate_roster_receipt_sha256=candidate[
            "target_independent_candidate_roster_receipt_sha256"
        ],
        source_reference_receipt_sha256=reference["receipt_sha256"],
        authority="public_seizure_interval",
        event_status=event_status,
        onset_status=onset_status,
        offset_status=offset_status,
        bout_count_status=bout_count_status,
        onset_interval_seconds=onset_interval,
        offset_interval_seconds=offset_interval,
        firewall=BAIEGSegmentalTargetFirewallV1(),
    )
    trace = {
        "reference_coverage_status": coverage,
        "bout_matching_status": matching_status,
        "overlap_semantics": (
            "positive_duration_intersection_with_frozen_analysis_interval"
        ),
        "overlapping_public_event_ids": [row["public_event_id"] for row in matched],
        "overlapping_public_event_intervals_recording_seconds": [
            [
                row["onset_recording_seconds"],
                row["offset_recording_seconds"],
            ]
            for row in matched
        ],
        "overlapping_bout_count": matched_count,
        "annotation_timestamp_resolution_seconds": reference[
            "annotation_timestamp_resolution_seconds"
        ],
        "resolution_semantics": reference["resolution_semantics"],
        "single_bout_onset_boundary_recording_seconds": onset_boundary,
        "single_bout_offset_boundary_recording_seconds": offset_boundary,
        "onset_resolution_interval_recording_seconds": onset_resolution_interval,
        "offset_resolution_interval_recording_seconds": offset_resolution_interval,
        "target_projection": {
            "event_status": target.event_status,
            "onset_status": target.onset_status,
            "offset_status": target.offset_status,
            "bout_count_status": target.bout_count_status,
            "onset_interval_seconds": (
                list(target.onset_interval_seconds)
                if target.onset_interval_seconds is not None
                else None
            ),
            "offset_interval_seconds": (
                list(target.offset_interval_seconds)
                if target.offset_interval_seconds is not None
                else None
            ),
        },
    }
    return target, trace


def _segmental_target_payload(target: BAIEGSegmentalEventTargetV1) -> dict[str, Any]:
    target.verify_integrity()
    return {
        "schema_version": target.schema_version,
        "event_id": target.event_id,
        "recording_id": target.recording_id,
        "patient_uid": target.patient_uid,
        "model_split": target.model_split,
        "source_event_receipt_sha256": target.source_event_receipt_sha256,
        "adaptive_acquisition_receipt_sha256": (
            target.adaptive_acquisition_receipt_sha256
        ),
        "target_independent_candidate_roster_receipt_sha256": (
            target.target_independent_candidate_roster_receipt_sha256
        ),
        "source_reference_receipt_sha256": target.source_reference_receipt_sha256,
        "authority": target.authority,
        "event_status": target.event_status,
        "onset_status": target.onset_status,
        "offset_status": target.offset_status,
        "bout_count_status": target.bout_count_status,
        "onset_interval_seconds": (
            list(target.onset_interval_seconds)
            if target.onset_interval_seconds is not None
            else None
        ),
        "offset_interval_seconds": (
            list(target.offset_interval_seconds)
            if target.offset_interval_seconds is not None
            else None
        ),
        "firewall": target.firewall.to_dict(),
        "receipt_sha256": target.receipt_sha256,
    }


def _validate_join_identity(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    for field in ("recording_id", "patient_uid", "model_split"):
        if candidate[field] != reference[field]:
            raise ValueError(f"candidate/reference {field} mismatch")


def _materialization_semantic_digest_source(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    digest = deepcopy(dict(payload))
    digest["receipt_id"] = "BAIEG-TUSZ-MATERIALIZATION-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    return digest


def _seal_materialization_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    semantic = _materialization_semantic_digest_source(result)
    result["receipt_id"] = "BAIEGTUSZMAT-" + _canonical_sha256(semantic)[:24]
    hash_source = deepcopy(result)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = _canonical_sha256(hash_source)
    return result


@dataclass(frozen=True, slots=True)
class BAIEGTUSZIntervalTargetMaterializationV1:
    """One target plus the complete receipt needed to replay its join."""

    target: BAIEGSegmentalEventTargetV1
    receipt: dict[str, Any]
    _candidate_envelope: ValidatedBAIEGTUSZCandidateEnvelopeV1 = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.target, BAIEGSegmentalEventTargetV1):
            raise TypeError("TUSZ target materialization requires a segmental target")
        self.target.verify_integrity()
        replayed = validate_ba_ieg_tusz_interval_target_materialization_receipt_v1(
            self.receipt,
            candidate_envelope=self._candidate_envelope,
        )
        if replayed["target"]["receipt_sha256"] != self.target.receipt_sha256:
            raise ValueError("materialization receipt belongs to another target")
        object.__setattr__(self, "receipt", replayed)

    @property
    def materialization_receipt(self) -> dict[str, Any]:
        return deepcopy(self.receipt)

    def verify_integrity(self) -> None:
        self.target.verify_integrity()
        replayed = validate_ba_ieg_tusz_interval_target_materialization_receipt_v1(
            self.receipt,
            candidate_envelope=self._candidate_envelope,
        )
        if replayed["target"]["receipt_sha256"] != self.target.receipt_sha256:
            raise ValueError("TUSZ interval target materialization changed")


def materialize_ba_ieg_tusz_candidate_interval_target_v1(
    candidate_envelope: ValidatedBAIEGTUSZCandidateEnvelopeV1,
    public_interval_reference: ValidatedBAIEGTUSZPublicIntervalReferenceV1,
) -> BAIEGTUSZIntervalTargetMaterializationV1:
    """Join a frozen target-free candidate to public supervision only."""

    if type(candidate_envelope) is not ValidatedBAIEGTUSZCandidateEnvelopeV1:
        raise TypeError(
            "materialization requires the exact validated candidate capability"
        )
    if (
        type(public_interval_reference)
        is not ValidatedBAIEGTUSZPublicIntervalReferenceV1
    ):
        raise TypeError(
            "materialization requires the exact validated reference capability"
        )
    candidate = validate_ba_ieg_tusz_candidate_envelope_v1(
        candidate_envelope.payload(),
        event=candidate_envelope._source_event,
    ).payload()
    reference = validate_ba_ieg_tusz_public_interval_reference_v1(
        public_interval_reference.payload()
    ).payload()
    _validate_join_identity(candidate, reference)
    target, trace = _derive_target_and_trace(candidate, reference)
    body: dict[str, Any] = {
        "schema_version": BA_IEG_TUSZ_INTERVAL_TARGET_MATERIALIZATION_SCHEMA_VERSION,
        "receipt_id": "BAIEG-TUSZ-MATERIALIZATION-PENDING",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": BA_IEG_TUSZ_INTERVAL_TARGET_MATERIALIZATION_METHOD_ID,
        "candidate_envelope": candidate,
        "candidate_envelope_payload_sha256": _canonical_sha256(candidate),
        "public_interval_reference": reference,
        "source_reference_receipt_sha256": reference["receipt_sha256"],
        "join_identity": {
            "event_id": candidate["event_id"],
            "recording_id": candidate["recording_id"],
            "patient_uid": candidate["patient_uid"],
            "model_split": candidate["model_split"],
        },
        "derivation_trace": trace,
        "target": _segmental_target_payload(target),
        "scope_receipt": deepcopy(_MATERIALIZATION_SCOPE),
    }
    receipt = _seal_materialization_receipt(body)
    return BAIEGTUSZIntervalTargetMaterializationV1(
        target=target,
        receipt=receipt,
        _candidate_envelope=candidate_envelope,
    )


def validate_ba_ieg_tusz_interval_target_materialization_receipt_v1(
    payload: object,
    *,
    candidate_envelope: ValidatedBAIEGTUSZCandidateEnvelopeV1,
) -> dict[str, Any]:
    """Replay candidate, reference, matching, censoring and target projection."""

    required = {
        "schema_version",
        "receipt_id",
        "receipt_sha256",
        "method_id",
        "candidate_envelope",
        "candidate_envelope_payload_sha256",
        "public_interval_reference",
        "source_reference_receipt_sha256",
        "join_identity",
        "derivation_trace",
        "target",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("TUSZ interval-target materialization has invalid fields")
    if type(candidate_envelope) is not ValidatedBAIEGTUSZCandidateEnvelopeV1:
        raise TypeError(
            "materialization replay requires the exact candidate capability"
        )
    data = deepcopy(payload)
    if (
        data["schema_version"]
        != BA_IEG_TUSZ_INTERVAL_TARGET_MATERIALIZATION_SCHEMA_VERSION
        or data["method_id"] != BA_IEG_TUSZ_INTERVAL_TARGET_MATERIALIZATION_METHOD_ID
    ):
        raise ValueError("TUSZ interval-target materialization schema/method drifted")
    _identifier(data["receipt_id"], "receipt_id")
    _sha256(data["receipt_sha256"], "receipt_sha256")
    candidate = validate_ba_ieg_tusz_candidate_envelope_v1(
        candidate_envelope.payload(),
        event=candidate_envelope._source_event,
    ).payload()
    if data["candidate_envelope"] != candidate:
        raise ValueError("materialization embeds another candidate envelope")
    reference = validate_ba_ieg_tusz_public_interval_reference_v1(
        data["public_interval_reference"]
    ).payload()
    if data["candidate_envelope_payload_sha256"] != _canonical_sha256(candidate):
        raise ValueError("materialization candidate-envelope hash mismatch")
    if data["source_reference_receipt_sha256"] != reference["receipt_sha256"]:
        raise ValueError("materialization public-reference hash mismatch")
    _validate_join_identity(candidate, reference)
    expected_identity = {
        "event_id": candidate["event_id"],
        "recording_id": candidate["recording_id"],
        "patient_uid": candidate["patient_uid"],
        "model_split": candidate["model_split"],
    }
    if data["join_identity"] != expected_identity:
        raise ValueError("materialization join identity drifted")
    target, trace = _derive_target_and_trace(candidate, reference)
    if data["derivation_trace"] != trace:
        raise ValueError("materialization bout matching or censor derivation drifted")
    if data["target"] != _segmental_target_payload(target):
        raise ValueError("materialization target projection does not replay")
    if data["scope_receipt"] != _MATERIALIZATION_SCOPE:
        raise ValueError("TUSZ interval-target materialization firewall drifted")
    expected_id = (
        "BAIEGTUSZMAT-"
        + _canonical_sha256(_materialization_semantic_digest_source(data))[:24]
    )
    if data["receipt_id"] != expected_id:
        raise ValueError("materialization receipt ID is not content-bound")
    hash_source = deepcopy(data)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(hash_source):
        raise ValueError("materialization receipt hash is not content-bound")
    return data


def validate_ba_ieg_tusz_candidate_interval_target_materialization_v1(
    payload: object,
    *,
    candidate_envelope: ValidatedBAIEGTUSZCandidateEnvelopeV1,
) -> dict[str, Any]:
    """Compatibility spelling for the full replay validator."""

    return validate_ba_ieg_tusz_interval_target_materialization_receipt_v1(
        payload,
        candidate_envelope=candidate_envelope,
    )


__all__ = [
    "BA_IEG_TUSZ_CANDIDATE_ENVELOPE_SCHEMA_VERSION",
    "BA_IEG_TUSZ_INTERVAL_TARGET_MATERIALIZATION_METHOD_ID",
    "BA_IEG_TUSZ_INTERVAL_TARGET_MATERIALIZATION_SCHEMA_VERSION",
    "BA_IEG_TUSZ_PUBLIC_INTERVAL_REFERENCE_SCHEMA_VERSION",
    "BAIEGTUSZIntervalTargetMaterializationV1",
    "ValidatedBAIEGTUSZCandidateEnvelopeV1",
    "ValidatedBAIEGTUSZPublicIntervalReferenceV1",
    "build_ba_ieg_tusz_public_interval_reference_v1",
    "freeze_ba_ieg_tusz_candidate_envelope_after_tokenization_v1",
    "materialize_ba_ieg_tusz_candidate_interval_target_v1",
    "validate_ba_ieg_tusz_candidate_envelope_v1",
    "validate_ba_ieg_tusz_candidate_interval_target_materialization_v1",
    "validate_ba_ieg_tusz_interval_target_materialization_receipt_v1",
    "validate_ba_ieg_tusz_public_interval_reference_v1",
]
