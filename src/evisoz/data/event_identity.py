"""Content-closed identity for one seizure-conditioned EviSOZ event.

The receipt is deliberately separate from waveform, montage and label
artifacts.  Those artifacts must all bind the same receipt before they may be
assembled into a training example.  Raw patient identifiers are never stored;
the linkage ledger is joined through a source-patient SHA-256.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from .artifact_ref import (
    canonical_json_sha256,
    validate_artifact_ref,
)


EVENT_IDENTITY_SCHEMA_VERSION = "evisoz_event_identity_v1"
ANCHOR_QUALITIES = ("exact", "approximate", "provided_interval")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_TOP_KEYS = {
    "schema_version",
    "identity_id",
    "dataset_id",
    "sample_id",
    "event_id",
    "linkage_group_id",
    "source_patient_sha256",
    "parent_signal_ref",
    "anchor_source_ref",
    "anchor",
    "raw_patient_identifiers_stored",
    "receipt_sha256",
}


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a stable ASCII identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["identity_id"] = _PENDING_ID
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _anchor(anchor_quality: str) -> dict[str, object]:
    if anchor_quality not in ANCHOR_QUALITIES:
        raise ValueError("unsupported anchor_quality")
    return {
        "condition": "known_seizure_segment",
        "quality": anchor_quality,
        "sampling_rate_numerator": 200,
        "sampling_rate_denominator": 1,
        "sample_count": 12000,
        "t0_sample_index": 2400,
        "analysis_interval_seconds": [-12.0, 48.0],
    }


def build_event_identity(
    *,
    dataset_id: str,
    sample_id: str,
    event_id: str,
    linkage_group_id: str,
    source_patient_sha256: str,
    parent_signal_ref: Mapping[str, object],
    anchor_source_ref: Mapping[str, object],
    anchor_quality: str,
) -> dict[str, Any]:
    """Build the immutable join key for one exact analysis window."""

    parent = validate_artifact_ref(parent_signal_ref)
    anchor_source = validate_artifact_ref(anchor_source_ref)
    if (
        parent["artifact_kind"] != "canonical_signal"
        or parent["content_hash"]["domain"] != "raw_bytes_v1"
    ):
        raise ValueError("event identity parent must be a raw canonical_signal")
    if anchor_source["artifact_kind"] not in {
        "source_event_annotation",
        "analysis_selection_receipt",
        "manual_anchor_release",
    }:
        raise ValueError("event identity anchor source kind is unsupported")
    body: dict[str, Any] = {
        "schema_version": EVENT_IDENTITY_SCHEMA_VERSION,
        "identity_id": _PENDING_ID,
        "dataset_id": _identifier(dataset_id, "dataset_id"),
        "sample_id": _identifier(sample_id, "sample_id"),
        "event_id": _identifier(event_id, "event_id"),
        "linkage_group_id": _identifier(linkage_group_id, "linkage_group_id"),
        "source_patient_sha256": _sha256(
            source_patient_sha256,
            "source_patient_sha256",
        ),
        "parent_signal_ref": parent,
        "anchor_source_ref": anchor_source,
        "anchor": _anchor(anchor_quality),
        "raw_patient_identifiers_stored": False,
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["identity_id"] = "EVISOZ-EVENT-" + canonical_json_sha256(
        _id_source(body)
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_event_identity(body)


def validate_event_identity(value: object) -> dict[str, Any]:
    """Validate an event identity without trusting its recorded seals."""

    if type(value) is not dict or set(value) != _TOP_KEYS:
        raise ValueError("event identity fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != EVENT_IDENTITY_SCHEMA_VERSION:
        raise ValueError("event identity schema_version drifted")
    for field in ("dataset_id", "sample_id", "event_id", "linkage_group_id"):
        data[field] = _identifier(data[field], field)
    data["source_patient_sha256"] = _sha256(
        data["source_patient_sha256"],
        "source_patient_sha256",
    )
    parent = validate_artifact_ref(data["parent_signal_ref"])
    if (
        parent["artifact_kind"] != "canonical_signal"
        or parent["content_hash"]["domain"] != "raw_bytes_v1"
    ):
        raise ValueError("event identity parent must be a raw canonical_signal")
    anchor_source = validate_artifact_ref(data["anchor_source_ref"])
    if anchor_source["artifact_kind"] not in {
        "source_event_annotation",
        "analysis_selection_receipt",
        "manual_anchor_release",
    }:
        raise ValueError("event identity anchor source kind is unsupported")
    anchor = data["anchor"]
    if type(anchor) is not dict or anchor != _anchor(anchor.get("quality")):
        raise ValueError("event identity anchor/clock contract drifted")
    if data["raw_patient_identifiers_stored"] is not False:
        raise ValueError("event identity must not store raw patient identifiers")
    expected_id = "EVISOZ-EVENT-" + canonical_json_sha256(_id_source(data))[:24]
    if data["identity_id"] != expected_id:
        raise ValueError("event identity_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("event identity receipt hash drifted")
    data["parent_signal_ref"] = parent
    data["anchor_source_ref"] = anchor_source
    return data


__all__ = [
    "EVENT_IDENTITY_SCHEMA_VERSION",
    "ANCHOR_QUALITIES",
    "build_event_identity",
    "validate_event_identity",
]
