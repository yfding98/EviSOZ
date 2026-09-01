"""Content binding between canonical EEG and adaptive boundary search.

The detector manifest historically identifies an EDF *container* by its file
hash, whereas canonical EEG deliberately identifies only selected physical
EEG samples and signal-header fields.  Those identities have different
semantics and must never be compared as if they were interchangeable.  This
small receipt binds adaptive preprocessing/search to the immutable canonical
signal root while leaving the detector's container binding intact.

No function in this module opens an EDF or accepts annotations, spreadsheets,
clinical text, or labels.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Final

from src.soz.geometry import STANDARD_19

from .canonical_signal_views import validate_canonical_signal_receipt


CANONICAL_ADAPTIVE_BINDING_SCHEMA_VERSION: Final[str] = (
    "canonical_adaptive_signal_binding_v1"
)

_SCOPE_RECEIPT: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return value


def validate_canonical_adaptive_signal_binding(payload: object) -> dict[str, Any]:
    """Validate a complete canonical identity/availability partition."""

    if type(payload) is not dict:
        raise TypeError("canonical adaptive signal binding must be an object")
    required = {
        "schema_version",
        "canonical_signal_id",
        "canonical_recording_id",
        "canonical_source_signal_sha256",
        "canonical_receipt_sha256",
        "canonical_source_header_receipt_sha256",
        "recording_duration_seconds",
        "semantic_channel_order",
        "observed_channel_ids",
        "unobserved_channel_ids",
        "missing_channel_policy",
        "scope_receipt",
        "binding_sha256",
    }
    if set(payload) != required:
        raise ValueError(
            "canonical adaptive signal binding has missing or unknown fields"
        )
    data = deepcopy(payload)
    if data["schema_version"] != CANONICAL_ADAPTIVE_BINDING_SCHEMA_VERSION:
        raise ValueError("unsupported canonical adaptive binding schema")
    _identifier(data["canonical_signal_id"], "canonical_signal_id")
    _identifier(data["canonical_recording_id"], "canonical_recording_id")
    for name in (
        "canonical_source_signal_sha256",
        "canonical_receipt_sha256",
        "canonical_source_header_receipt_sha256",
        "binding_sha256",
    ):
        _sha256(data[name], name)
    duration = data["recording_duration_seconds"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise ValueError("canonical adaptive recording duration is invalid")
    if data["semantic_channel_order"] != list(STANDARD_19):
        raise ValueError("canonical adaptive semantic channel order drifted")
    observed = data["observed_channel_ids"]
    unobserved = data["unobserved_channel_ids"]
    if (
        not isinstance(observed, list)
        or not isinstance(unobserved, list)
        or observed != [item for item in STANDARD_19 if item in set(observed)]
        or unobserved != [item for item in STANDARD_19 if item in set(unobserved)]
        or set(observed).intersection(unobserved)
        or set(observed).union(unobserved) != set(STANDARD_19)
    ):
        raise ValueError("canonical adaptive observed/unobserved partition is invalid")
    if data["missing_channel_policy"] != (
        "zero_carrier_strict_mask_no_interpolation_no_evidence_v1"
    ):
        raise ValueError("canonical adaptive missing-channel policy drifted")
    if data["scope_receipt"] != _SCOPE_RECEIPT:
        raise ValueError("canonical adaptive binding violates the EEG-only scope")
    digest_source = deepcopy(data)
    digest_source["binding_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["binding_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("canonical adaptive binding hash does not bind content")
    return data


def build_canonical_adaptive_signal_binding(canonical_bundle: object) -> dict[str, Any]:
    """Build a binding from a validated canonical EDF view bundle."""

    # Local import avoids making the canonical materializer depend on this
    # downstream adaptive contract.
    from .canonical_edf_materialization import (
        CanonicalEDFViewBundle,
        validate_canonical_edf_materialization,
    )

    if not isinstance(canonical_bundle, CanonicalEDFViewBundle):
        raise TypeError("canonical_bundle must be CanonicalEDFViewBundle")
    validate_canonical_edf_materialization(canonical_bundle)
    canonical = validate_canonical_signal_receipt(
        canonical_bundle.canonical_record.canonical_receipt
    )
    return build_canonical_adaptive_signal_binding_from_receipt(
        canonical,
        canonical_source_header_receipt_sha256=(
            canonical_bundle.canonical_record.source_header_receipt[
                "receipt_sha256"
            ]
        ),
    )


def build_canonical_adaptive_signal_binding_from_receipt(
    canonical_receipt: object,
    *,
    canonical_source_header_receipt_sha256: str,
) -> dict[str, Any]:
    """Build an adaptive identity binding from a canonical signal receipt.

    The production EDF materializer remains the authoritative source of the
    source-header receipt hash.  This lower-level constructor is useful to
    producers and contract tests that already hold a validated canonical
    receipt and must bind a downstream adaptive search before any task view is
    materialized.
    """

    canonical = validate_canonical_signal_receipt(canonical_receipt)
    source_header_sha256 = _sha256(
        canonical_source_header_receipt_sha256,
        "canonical_source_header_receipt_sha256",
    )
    observed = [
        str(row["channel_id"])
        for row in canonical["channels"]
        if bool(row["observed"])
    ]
    unobserved = [
        str(row["channel_id"])
        for row in canonical["channels"]
        if not bool(row["observed"])
    ]
    body: dict[str, Any] = {
        "schema_version": CANONICAL_ADAPTIVE_BINDING_SCHEMA_VERSION,
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_recording_id": canonical["recording_id"],
        "canonical_source_signal_sha256": canonical["source_signal_sha256"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "canonical_source_header_receipt_sha256": source_header_sha256,
        "recording_duration_seconds": canonical["recording_duration_seconds"],
        "semantic_channel_order": list(STANDARD_19),
        "observed_channel_ids": observed,
        "unobserved_channel_ids": unobserved,
        "missing_channel_policy": (
            "zero_carrier_strict_mask_no_interpolation_no_evidence_v1"
        ),
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "binding_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["binding_sha256"] = _canonical_sha256(body)
    return validate_canonical_adaptive_signal_binding(body)


def validate_canonical_adaptive_binding_against_receipt(
    binding_payload: object,
    canonical_receipt: object,
    *,
    canonical_source_header_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Prove that an adaptive binding and canonical EEG are the same source.

    Duration and clock agreement alone are intentionally insufficient: two
    different recordings can have identical shapes and clocks.  This gate
    therefore compares the content-addressed canonical identity, recording
    identity, physical-signal hash, receipt hash, duration, and the exact
    observed/unobserved Standard-19 partition.  When the host also has the EDF
    source-header receipt its hash is checked as an additional container-to-
    canonical link.
    """

    binding = validate_canonical_adaptive_signal_binding(binding_payload)
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    observed = tuple(
        str(row["channel_id"])
        for row in canonical["channels"]
        if bool(row["observed"])
    )
    unobserved = tuple(
        str(row["channel_id"])
        for row in canonical["channels"]
        if not bool(row["observed"])
    )
    mismatches: list[str] = []
    comparisons = (
        ("canonical_signal_id", canonical["canonical_signal_id"]),
        ("canonical_recording_id", canonical["recording_id"]),
        ("canonical_source_signal_sha256", canonical["source_signal_sha256"]),
        ("canonical_receipt_sha256", canonical["receipt_sha256"]),
    )
    for field, expected in comparisons:
        if binding[field] != expected:
            mismatches.append(field)
    if not math.isclose(
        float(binding["recording_duration_seconds"]),
        float(canonical["recording_duration_seconds"]),
        abs_tol=1e-6,
    ):
        mismatches.append("recording_duration_seconds")
    if tuple(binding["observed_channel_ids"]) != observed:
        mismatches.append("observed_channel_ids")
    if tuple(binding["unobserved_channel_ids"]) != unobserved:
        mismatches.append("unobserved_channel_ids")
    if canonical_source_header_receipt_sha256 is not None:
        source_header_sha256 = _sha256(
            canonical_source_header_receipt_sha256,
            "canonical_source_header_receipt_sha256",
        )
        if binding["canonical_source_header_receipt_sha256"] != source_header_sha256:
            mismatches.append("canonical_source_header_receipt_sha256")
    if mismatches:
        raise ValueError(
            "canonical adaptive identity mismatch: " + ", ".join(mismatches)
        )
    return binding


__all__ = [
    "CANONICAL_ADAPTIVE_BINDING_SCHEMA_VERSION",
    "build_canonical_adaptive_signal_binding",
    "build_canonical_adaptive_signal_binding_from_receipt",
    "validate_canonical_adaptive_binding_against_receipt",
    "validate_canonical_adaptive_signal_binding",
]
