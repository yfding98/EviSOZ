"""Typed, externally replayed signal-lineage authority for EEG detectors.

Channel-support routing and provider transforms must not trust a caller-owned
``observed``/``usable`` roster or a hash-shaped string.  This module binds four
different facts and keeps their semantics separate:

* the canonical physical EEG tensor/header identity;
* the observed Standard-19 carrier-axis roster;
* the EEG *electrical* reference system and exact common sampling clock; and
* a versioned EEG-only channel-QC decision derived from canonical quality
  primitives.

The word ``reference`` in the electrical-reference authority is a legitimate
signal provenance fact.  It is intentionally distinct from a seizure target,
reference interval, physician label, or evaluation label, all of which remain
forbidden.

There are two authority tiers.  A provider-transform authority is rebuilt from
an actual :class:`CanonicalEEGRecord` and its physical payload.  A corpus
support audit may instead use a fully validated canonical-physical audit
outcome; that tier can prove a *policy route* only because the signal-only
duplicate audit did not materialize QC or the electrical reference graph.  It
must never be promoted to executable provider input.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch

from src.soz.geometry import STANDARD_19

from .canonical_edf_materialization import (
    CanonicalEEGRecord,
    canonical_source_tensor_sha256,
    validate_canonical_edf_source_header_receipt,
)
from .canonical_signal_views import (
    EVIDENCE_FAMILIES,
    validate_canonical_signal_receipt,
)
from .montage_reference_observability import (
    direct_electrode_index_by_signal,
    validate_montage_reference_observability_receipt,
)
from .tusz_canonical_physical_signal_audit_v1 import (
    validate_tusz_canonical_physical_analysis_projection_v1,
    validate_tusz_canonical_physical_duplicate_audit_v1,
)


SCHEMA_VERSION: Final[str] = "clinical_eeg_detector_signal_lineage_authority_v1"
METHOD_ID: Final[str] = (
    "canonical_physical_roster_electrical_reference_EEG_QC_semantic_replay_v1"
)
QC_SCHEMA_VERSION: Final[str] = "clinical_eeg_detector_channel_QC_authority_v1"
QC_OPERATOR_ID: Final[str] = (
    "canonical_unusable_interval_union_full_record_exclusion_v1"
)
ELECTRICAL_REFERENCE_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_electrical_reference_system_authority_v1"
)
OBSERVED_ROSTER_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_observed_physical_roster_authority_v1"
)
CLOCK_SCHEMA_VERSION: Final[str] = "clinical_eeg_common_sample_clock_authority_v1"
POLICY_AUDIT_TRUST_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_canonical_policy_audit_trust_anchor_v1"
)

_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
_SHA_CHARS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_AUTHORITY_SEAL = object()
_AUDIT_ANCHOR_SEAL = object()


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA_CHARS for character in value)
    )


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return str(value)


def _content_address(
    body: Mapping[str, Any], *, hash_field: str = "receipt_sha256"
) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result[hash_field] = _PENDING
    result[hash_field] = _sha256(result)
    return result


def _validate_content_address(
    value: object,
    *,
    required: set[str],
    context: str,
    hash_field: str = "receipt_sha256",
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{context} fields drifted")
    row = deepcopy(value)
    supplied = _require_sha256(row[hash_field], f"{context} {hash_field}")
    row[hash_field] = _PENDING
    if supplied != _sha256(row):
        raise ValueError(f"{context} is not content-addressed")
    return deepcopy(value)


@dataclass(frozen=True)
class ValidatedDetectorSignalLineageAuthority:
    """Opaque authority accepted by router/provider consumers.

    A raw mapping is intentionally not accepted by downstream APIs.  The
    private seal is issued only after an external canonical-record payload or
    a validated corpus-audit artifact has been replayed.
    """

    _receipt_json: str = field(repr=False)
    _validation_seal: object = field(repr=False, compare=False)

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)

    @property
    def receipt_sha256(self) -> str:
        return str(self.receipt["receipt_sha256"])


@dataclass(frozen=True)
class CanonicalPolicyAuditTrustAnchor:
    """Opaque membership proof for validated audit/projection outcomes."""

    audit_file_sha256: str
    projection_file_sha256: str
    audit_id: str
    audit_receipt_sha256: str
    projection_id: str
    projection_receipt_sha256: str
    _outcome_by_identity: Mapping[str, Mapping[str, Any]] = field(
        repr=False, compare=False
    )
    _projection_tensor_by_identity: Mapping[str, str] = field(
        repr=False, compare=False
    )
    _validation_seal: object = field(repr=False, compare=False)


def _seal_authority(receipt: Mapping[str, Any]) -> ValidatedDetectorSignalLineageAuthority:
    _validate_authority_receipt_semantics(dict(receipt))
    return ValidatedDetectorSignalLineageAuthority(
        _receipt_json=_canonical_json(receipt),
        _validation_seal=_AUTHORITY_SEAL,
    )


def require_validated_detector_signal_lineage_authority(
    authority: object,
) -> dict[str, Any]:
    """Return a replayed receipt only for the opaque validated authority."""

    if (
        not isinstance(authority, ValidatedDetectorSignalLineageAuthority)
        or authority._validation_seal is not _AUTHORITY_SEAL
    ):
        raise TypeError(
            "detector input requires a typed externally replayed signal-lineage "
            "authority; a raw mapping/hash/roster is not authority"
        )
    receipt = authority.receipt
    _validate_authority_receipt_semantics(receipt)
    return receipt


def _canonical_roster(values: Sequence[object], *, context: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{context} must be an ordered channel roster")
    if (
        any(not isinstance(item, str) or not item for item in values)
        or len(values) != len(set(values))
        or list(values)
        != [channel for channel in STANDARD_19 if channel in set(values)]
    ):
        raise ValueError(f"{context} is not a canonical Standard-19 subset")
    return [str(item) for item in values]


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[list[float]]:
    ordered = sorted((float(start), float(stop)) for start, stop in intervals)
    merged: list[list[float]] = []
    for start, stop in ordered:
        if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
            raise ValueError("canonical QC interval is invalid")
        if not merged or start > merged[-1][1] + 1e-12:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return merged


def _rounded_fraction(value: float) -> float:
    return float(format(float(value), ".12g"))


def _build_eeg_only_qc_authority(
    canonical: Mapping[str, Any], *, observed_channel_ids: Sequence[str]
) -> dict[str, Any]:
    """Replay one conservative record-wide channel-usability policy.

    A channel is removed from detector support only when the union of
    canonical ``severity=unusable`` intervals covers the complete recording.
    Partial artifact intervals remain in the QC ledger but do not silently
    change the detector input width.  The rule is intentionally simple and
    versioned; future threshold policies require a new authority schema.
    """

    duration = float(canonical["recording_duration_seconds"])
    rows: list[dict[str, Any]] = []
    usable: list[str] = []
    for channel_id in observed_channel_ids:
        intervals: list[tuple[float, float]] = []
        primitive_ids: list[str] = []
        for primitive in canonical["quality_primitives"]:
            if (
                channel_id in primitive["channel_ids"]
                and primitive["severity"] == "unusable"
                and set(primitive["disabled_evidence_families"])
                == set(EVIDENCE_FAMILIES)
            ):
                intervals.append(
                    (
                        float(primitive["start_recording_seconds"]),
                        float(primitive["stop_recording_seconds"]),
                    )
                )
                primitive_ids.append(str(primitive["quality_id"]))
        merged = _merge_intervals(intervals)
        unusable_seconds = sum(stop - start for start, stop in merged)
        fraction = min(1.0, max(0.0, unusable_seconds / duration))
        excluded = fraction >= 1.0 - 1e-12
        if not excluded:
            usable.append(channel_id)
        rows.append(
            {
                "channel_id": channel_id,
                "source_unusable_quality_ids": sorted(primitive_ids),
                "merged_unusable_intervals_seconds": merged,
                "unusable_duration_seconds": _rounded_fraction(unusable_seconds),
                "unusable_record_fraction": _rounded_fraction(fraction),
                "excluded_from_record_wide_detector_support": excluded,
            }
        )
    return _content_address(
        {
            "schema_version": QC_SCHEMA_VERSION,
            "operator_id": QC_OPERATOR_ID,
            "source_canonical_signal_receipt_sha256": canonical["receipt_sha256"],
            "source_quality_primitive_count": len(canonical["quality_primitives"]),
            "decision_threshold": {
                "metric": "union_unusable_interval_duration_divided_by_record_duration",
                "exclude_when_greater_than_or_equal": 1.0,
                "interval_union_tolerance_seconds": 1e-12,
            },
            "channel_decisions": rows,
            "usable_standard_channel_ids": usable,
            "scope_receipt": {
                "EEG_derived_canonical_quality_primitives_used": True,
                "seizure_target_or_reference_label_used": False,
                "EDF_annotation_used": False,
                "spreadsheet_or_doctor_text_used": False,
                "patient_or_subject_identity_used": False,
                "detector_posterior_used": False,
            },
            "receipt_sha256": _PENDING,
        }
    )


def _validate_qc_authority(
    value: object,
    *,
    canonical: Mapping[str, Any] | None,
    observed_channel_ids: Sequence[str],
    policy_only: bool,
) -> dict[str, Any]:
    if policy_only:
        required = {
            "schema_version",
            "operator_id",
            "status",
            "source_canonical_outcome_receipt_sha256",
            "usable_standard_channel_ids",
            "EEG_QC_exclusion_claim_authorized",
            "provider_transform_authorized",
            "scope_receipt",
            "receipt_sha256",
        }
        data = _validate_content_address(
            value, required=required, context="policy-only channel-QC authority"
        )
        if (
            data["schema_version"] != QC_SCHEMA_VERSION
            or data["operator_id"]
            != "canonical_audit_observed_equals_policy_usable_no_QC_claim_v1"
            or data["status"] != "QC_not_materialized_policy_route_only"
            or data["usable_standard_channel_ids"] != list(observed_channel_ids)
            or data["EEG_QC_exclusion_claim_authorized"] is not False
            or data["provider_transform_authorized"] is not False
            or data["scope_receipt"]
            != {
                "EEG_samples_or_quality_primitives_read": False,
                "seizure_target_or_reference_label_used": False,
                "EDF_annotation_used": False,
                "spreadsheet_or_doctor_text_used": False,
                "patient_or_subject_identity_used": False,
                "detector_posterior_used": False,
            }
        ):
            raise ValueError("policy-only channel-QC semantics drifted")
        _require_sha256(
            data["source_canonical_outcome_receipt_sha256"],
            "policy-only source outcome receipt",
        )
        return data

    if canonical is not None:
        expected = _build_eeg_only_qc_authority(
            canonical, observed_channel_ids=observed_channel_ids
        )
        if value != expected:
            raise ValueError("EEG-only channel-QC authority does not replay")
        return expected

    required = {
        "schema_version",
        "operator_id",
        "source_canonical_signal_receipt_sha256",
        "source_quality_primitive_count",
        "decision_threshold",
        "channel_decisions",
        "usable_standard_channel_ids",
        "scope_receipt",
        "receipt_sha256",
    }
    data = _validate_content_address(
        value, required=required, context="EEG-only channel-QC authority"
    )
    if (
        data["schema_version"] != QC_SCHEMA_VERSION
        or data["operator_id"] != QC_OPERATOR_ID
        or type(data["source_quality_primitive_count"]) is not int
        or data["source_quality_primitive_count"] < 0
        or data["decision_threshold"]
        != {
            "metric": "union_unusable_interval_duration_divided_by_record_duration",
            "exclude_when_greater_than_or_equal": 1.0,
            "interval_union_tolerance_seconds": 1e-12,
        }
        or data["scope_receipt"]
        != {
            "EEG_derived_canonical_quality_primitives_used": True,
            "seizure_target_or_reference_label_used": False,
            "EDF_annotation_used": False,
            "spreadsheet_or_doctor_text_used": False,
            "patient_or_subject_identity_used": False,
            "detector_posterior_used": False,
        }
    ):
        raise ValueError("EEG-only channel-QC authority semantics drifted")
    _require_sha256(
        data["source_canonical_signal_receipt_sha256"],
        "channel-QC source canonical receipt",
    )
    decisions = data["channel_decisions"]
    if (
        not isinstance(decisions, list)
        or [row.get("channel_id") for row in decisions]
        != list(observed_channel_ids)
    ):
        raise ValueError("EEG-only channel-QC decision roster drifted")
    replayed_usable: list[str] = []
    for row in decisions:
        required_row = {
            "channel_id",
            "source_unusable_quality_ids",
            "merged_unusable_intervals_seconds",
            "unusable_duration_seconds",
            "unusable_record_fraction",
            "excluded_from_record_wide_detector_support",
        }
        if type(row) is not dict or set(row) != required_row:
            raise ValueError("EEG-only channel-QC decision fields drifted")
        fraction = row["unusable_record_fraction"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 <= float(fraction) <= 1.0
            or row["excluded_from_record_wide_detector_support"]
            is not (float(fraction) >= 1.0 - 1e-12)
        ):
            raise ValueError("EEG-only channel-QC threshold decision drifted")
        if not row["excluded_from_record_wide_detector_support"]:
            replayed_usable.append(str(row["channel_id"]))
    if data["usable_standard_channel_ids"] != replayed_usable:
        raise ValueError("EEG-only channel-QC usable roster does not replay")
    return data


def _build_provider_authority_receipt(record: CanonicalEEGRecord) -> dict[str, Any]:
    if not isinstance(record, CanonicalEEGRecord):
        raise TypeError("provider lineage authority requires CanonicalEEGRecord")
    source_header = validate_canonical_edf_source_header_receipt(
        record.source_header_receipt
    )
    canonical = validate_canonical_signal_receipt(record.canonical_receipt)
    electrical_source = validate_montage_reference_observability_receipt(
        record.montage_reference_observability_receipt
    )
    tensor = record.observed_signal_volts.detach().cpu().to(torch.float32).contiguous()
    observed = _canonical_roster(
        list(record.observed_channel_ids), context="canonical record observed roster"
    )
    if tensor.ndim != 2 or tensor.shape[0] != len(observed) or tensor.shape[1] < 1:
        raise ValueError("canonical record tensor shape disagrees with observed roster")
    if not torch.isfinite(tensor).all():
        raise ValueError("canonical record tensor is nonfinite")
    tensor_hash = canonical_source_tensor_sha256(tensor, channel_ids=observed)
    if (
        source_header["source_tensor_sha256"] != tensor_hash
        or source_header["observed_channel_ids"] != observed
        or canonical["source_signal_sha256"] != source_header["source_signal_sha256"]
        or electrical_source["source_signal_sha256"]
        != source_header["source_signal_sha256"]
    ):
        raise ValueError("canonical record source/header/reference lineage drifted")

    canonical_observed = [
        str(row["channel_id"])
        for row in canonical["channels"]
        if row["observed"] is True
    ]
    if canonical_observed != observed:
        raise ValueError("canonical signal observed roster disagrees with source tensor")
    source_rows = source_header["channel_signal_headers"]
    rates = {
        (int(row["sampling_rate_numerator"]), int(row["sampling_rate_denominator"]))
        for row in source_rows
    }
    counts = {int(row["sample_count"]) for row in source_rows}
    if len(rates) != 1 or len(counts) != 1:
        raise ValueError("detector authority requires one common physical sample clock")
    numerator, denominator = next(iter(rates))
    sample_count = next(iter(counts))
    if tensor.shape[1] != sample_count:
        raise ValueError("canonical payload length disagrees with signal header clock")
    for channel_id, header_row in zip(observed, source_rows):
        canonical_row = next(
            row for row in canonical["channels"] if row["channel_id"] == channel_id
        )
        if (
            header_row["channel_id"] != channel_id
            or canonical_row["sample_rate_numerator"] != numerator
            or canonical_row["sample_rate_denominator"] != denominator
            or canonical_row["sample_count"] != sample_count
            or canonical_row["imputed"] is not False
        ):
            raise ValueError("canonical source channel mapping or clock drifted")

    signal_index_by_channel = direct_electrode_index_by_signal(electrical_source)
    if set(signal_index_by_channel) != set(observed):
        raise ValueError("electrical-reference direct-electrode roster drifted")
    observations = electrical_source["signal_label_observations"]
    axis_mapping: list[dict[str, Any]] = []
    for axis_index, (channel_id, header_row) in enumerate(zip(observed, source_rows)):
        signal_index = signal_index_by_channel[channel_id]
        if observations[signal_index]["raw_label"] != header_row["raw_label"]:
            raise ValueError("source header and reference-label mapping disagree")
        axis_mapping.append(
            {
                "carrier_axis_index": axis_index,
                "edf_signal_index": signal_index,
                "channel_id": channel_id,
                "raw_label_sha256": _sha256(
                    {
                        "domain": "edf-signal-label-v1",
                        "raw_label": header_row["raw_label"],
                    }
                ),
            }
        )

    clock = _content_address(
        {
            "schema_version": CLOCK_SCHEMA_VERSION,
            "source_header_receipt_sha256": source_header["receipt_sha256"],
            "sampling_rate_fraction_hz": [numerator, denominator],
            "sample_count": sample_count,
            "all_observed_carriers_share_exact_clock": True,
            "receipt_sha256": _PENDING,
        }
    )
    roster = _content_address(
        {
            "schema_version": OBSERVED_ROSTER_SCHEMA_VERSION,
            "source_header_receipt_sha256": source_header["receipt_sha256"],
            "canonical_signal_receipt_sha256": canonical["receipt_sha256"],
            "canonical_source_tensor_sha256": tensor_hash,
            "observed_standard_channel_ids": observed,
            "carrier_axis_mapping": axis_mapping,
            "synthetic_or_imputed_channel_count": 0,
            "receipt_sha256": _PENDING,
        }
    )
    compatibility = electrical_source["common_reference_compatibility"]
    common_compatible = bool(compatibility["compatible"])
    acquisition_matrix = electrical_source["acquisition_reference_model"][
        "reference_matrix_observability"
    ]
    electrical = _content_address(
        {
            "schema_version": ELECTRICAL_REFERENCE_SCHEMA_VERSION,
            "source_montage_reference_receipt_sha256": electrical_source[
                "receipt_sha256"
            ],
            "source_signal_sha256": source_header["source_signal_sha256"],
            "system_status": (
                "common_compatible_referential_qualified"
                if common_compatible
                else "mixed_or_unknown_reference_not_provider_qualified"
            ),
            "electrical_reference_system_id": (
                None
                if not common_compatible
                else "EEGREFSYS-"
                + _sha256(
                    {
                        "source_signal_sha256": source_header["source_signal_sha256"],
                        "reference_token": compatibility["reference_token"],
                        "observed_channel_ids": observed,
                    }
                )[:24]
            ),
            "reference_token": compatibility["reference_token"],
            "montage_class": electrical_source["montage_class"],
            "acquisition_reference_matrix_sha256": (
                None if acquisition_matrix is None else acquisition_matrix["matrix_sha256"]
            ),
            "common_reference_carrier_subtraction_authorized": common_compatible,
            "semantics": (
                "EEG_electrical_reference_provenance_not_a_seizure_reference_label_v1"
            ),
            "receipt_sha256": _PENDING,
        }
    )
    qc = _build_eeg_only_qc_authority(canonical, observed_channel_ids=observed)
    provider_authorized = bool(
        common_compatible
        and roster["synthetic_or_imputed_channel_count"] == 0
        and clock["all_observed_carriers_share_exact_clock"]
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "authority_id": _PENDING,
        "authority_tier": "provider_transform_payload_replayed",
        "trust_binding": {
            "external_authority_kind": (
                "CanonicalEEGRecord_object_plus_actual_payload_semantic_replay_v1"
            ),
            "canonical_record_class_required": True,
            "actual_payload_hash_replayed": True,
            "self_hash_only_accepted": False,
            "source_header_receipt_sha256": source_header["receipt_sha256"],
            "canonical_signal_receipt_sha256": canonical["receipt_sha256"],
            "montage_reference_receipt_sha256": electrical_source["receipt_sha256"],
        },
        "canonical_physical_signal": {
            "source_kind": "canonical_EDF_observed_referential_volts_v1",
            "canonical_signal_id": canonical["canonical_signal_id"],
            "canonical_signal_receipt_sha256": canonical["receipt_sha256"],
            "source_header_receipt_sha256": source_header["receipt_sha256"],
            "source_signal_sha256": source_header["source_signal_sha256"],
            "source_tensor_sha256": tensor_hash,
            "physical_unit_at_detector_boundary": "V",
        },
        "observed_roster_authority": roster,
        "common_sampling_clock_authority": clock,
        "electrical_reference_system_authority": electrical,
        "EEG_only_channel_QC_authority": qc,
        "policy_route_authorized": True,
        "provider_transform_authorized": provider_authorized,
        "scope_receipt": {
            "EEG_samples_used": True,
            "EEG_electrical_reference_provenance_used": True,
            "seizure_target_or_reference_label_used": False,
            "EDF_annotation_used": False,
            "spreadsheet_or_doctor_text_used": False,
            "clinical_history_used": False,
            "patient_or_subject_identity_used_for_route": False,
            "detector_posterior_used_for_route": False,
            "lineage_values_used_as_model_features": False,
        },
        "receipt_sha256": _PENDING,
    }
    identity_source = deepcopy(body)
    identity_source["authority_id"] = _PENDING
    identity_source["receipt_sha256"] = _PENDING
    body["authority_id"] = "DETSIGAUTH-" + _sha256(identity_source)[:24]
    body = _content_address(body)
    _validate_authority_receipt_semantics(body, canonical=canonical)
    return body


def authorize_detector_signal_lineage_from_canonical_record(
    record: CanonicalEEGRecord,
) -> ValidatedDetectorSignalLineageAuthority:
    """Issue a provider-capable authority after replaying the actual payload."""

    return _seal_authority(_build_provider_authority_receipt(record))


def load_canonical_policy_audit_trust_anchor(
    *, audit_bytes: bytes, projection_bytes: bytes
) -> CanonicalPolicyAuditTrustAnchor:
    """Validate complete canonical artifacts and freeze outcome membership.

    Both artifact byte hashes and semantic receipts are retained.  No
    annotation, target interval, doctor text or spreadsheet API is present.
    """

    if not isinstance(audit_bytes, bytes) or not isinstance(projection_bytes, bytes):
        raise TypeError("canonical audit trust anchor requires immutable JSON bytes")
    try:
        audit_raw = json.loads(audit_bytes)
        projection_raw = json.loads(projection_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical policy audit artifacts are invalid JSON") from exc
    audit = validate_tusz_canonical_physical_duplicate_audit_v1(audit_raw)
    projection = validate_tusz_canonical_physical_analysis_projection_v1(
        projection_raw
    )
    binding = projection["source_binding"]
    if (
        binding["source_canonical_physical_audit_id"] != audit["audit_id"]
        or binding["source_canonical_physical_audit_receipt_sha256"]
        != audit["receipt_sha256"]
    ):
        raise ValueError("canonical projection does not bind the validated audit")
    outcomes = {
        str(row["analysis_identity_id"]): deepcopy(row)
        for row in audit["outcomes"]
        if row["terminal_status"] == "success"
    }
    projected = {
        str(row["analysis_identity_id"]): str(
            row["canonical_physical_source_tensor_sha256"]
        )
        for row in projection["records"]
    }
    if not set(projected).issubset(outcomes):
        raise ValueError(
            "validated canonical projection contains an identity absent from audit"
        )
    for identity, tensor_hash in projected.items():
        if outcomes[identity]["physical_signal"]["canonical_source_tensor_sha256"] != tensor_hash:
            raise ValueError("canonical audit/projection tensor binding drifted")
    return CanonicalPolicyAuditTrustAnchor(
        audit_file_sha256=hashlib.sha256(audit_bytes).hexdigest(),
        projection_file_sha256=hashlib.sha256(projection_bytes).hexdigest(),
        audit_id=str(audit["audit_id"]),
        audit_receipt_sha256=str(audit["receipt_sha256"]),
        projection_id=str(projection["projection_id"]),
        projection_receipt_sha256=str(projection["receipt_sha256"]),
        _outcome_by_identity=MappingProxyType(
            {identity: outcomes[identity] for identity in projected}
        ),
        _projection_tensor_by_identity=MappingProxyType(projected),
        _validation_seal=_AUDIT_ANCHOR_SEAL,
    )


def _build_policy_only_authority_receipt(
    anchor: CanonicalPolicyAuditTrustAnchor, *, analysis_identity_id: str
) -> dict[str, Any]:
    if (
        not isinstance(anchor, CanonicalPolicyAuditTrustAnchor)
        or anchor._validation_seal is not _AUDIT_ANCHOR_SEAL
    ):
        raise TypeError("policy authority requires a validated canonical audit anchor")
    if analysis_identity_id not in anchor._outcome_by_identity:
        raise KeyError("analysis identity is absent from validated canonical audit")
    outcome = deepcopy(anchor._outcome_by_identity[analysis_identity_id])
    physical = outcome["physical_signal"]
    observed = _canonical_roster(
        physical["observed_channel_ids"], context="canonical audit observed roster"
    )
    outcome_receipt = _require_sha256(
        outcome["receipt_sha256"], "canonical outcome receipt"
    )
    roster = _content_address(
        {
            "schema_version": OBSERVED_ROSTER_SCHEMA_VERSION,
            "source_canonical_outcome_receipt_sha256": outcome_receipt,
            "canonical_source_tensor_sha256": physical[
                "canonical_source_tensor_sha256"
            ],
            "observed_standard_channel_ids": observed,
            "carrier_axis_mapping": [
                {"carrier_axis_index": index, "channel_id": channel}
                for index, channel in enumerate(observed)
            ],
            "synthetic_or_imputed_channel_count": 0,
            "receipt_sha256": _PENDING,
        }
    )
    clock = _content_address(
        {
            "schema_version": CLOCK_SCHEMA_VERSION,
            "source_canonical_outcome_receipt_sha256": outcome_receipt,
            "sampling_rate_fraction_hz": physical["sampling_rate_fraction"],
            "sample_count": physical["sample_count"],
            "all_observed_carriers_share_exact_clock": True,
            "receipt_sha256": _PENDING,
        }
    )
    electrical = _content_address(
        {
            "schema_version": ELECTRICAL_REFERENCE_SCHEMA_VERSION,
            "source_canonical_outcome_receipt_sha256": outcome_receipt,
            "source_header_receipt_sha256": physical[
                "canonical_source_header_receipt_sha256"
            ],
            "system_status": "not_materialized_in_signal_identity_audit",
            "electrical_reference_system_id": None,
            "reference_token": None,
            "common_reference_carrier_subtraction_authorized": False,
            "semantics": (
                "policy_route_only_EEG_electrical_reference_provenance_unresolved_v1"
            ),
            "receipt_sha256": _PENDING,
        }
    )
    qc = _content_address(
        {
            "schema_version": QC_SCHEMA_VERSION,
            "operator_id": (
                "canonical_audit_observed_equals_policy_usable_no_QC_claim_v1"
            ),
            "status": "QC_not_materialized_policy_route_only",
            "source_canonical_outcome_receipt_sha256": outcome_receipt,
            "usable_standard_channel_ids": observed,
            "EEG_QC_exclusion_claim_authorized": False,
            "provider_transform_authorized": False,
            "scope_receipt": {
                "EEG_samples_or_quality_primitives_read": False,
                "seizure_target_or_reference_label_used": False,
                "EDF_annotation_used": False,
                "spreadsheet_or_doctor_text_used": False,
                "patient_or_subject_identity_used": False,
                "detector_posterior_used": False,
            },
            "receipt_sha256": _PENDING,
        }
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "authority_id": _PENDING,
        "authority_tier": "canonical_audit_policy_route_only",
        "trust_binding": {
            "external_authority_kind": (
                "validated_canonical_physical_audit_and_projection_artifact_membership_v1"
            ),
            "audit_file_sha256": anchor.audit_file_sha256,
            "audit_id": anchor.audit_id,
            "audit_receipt_sha256": anchor.audit_receipt_sha256,
            "projection_file_sha256": anchor.projection_file_sha256,
            "projection_id": anchor.projection_id,
            "projection_receipt_sha256": anchor.projection_receipt_sha256,
            "canonical_outcome_receipt_sha256": outcome_receipt,
            "outcome_membership_semantically_replayed": True,
            "self_hash_only_accepted": False,
        },
        "canonical_physical_signal": {
            "source_kind": "validated_canonical_physical_audit_outcome_v1",
            "analysis_identity_id": analysis_identity_id,
            "canonical_outcome_receipt_sha256": outcome_receipt,
            "source_header_receipt_sha256": physical[
                "canonical_source_header_receipt_sha256"
            ],
            "source_signal_sha256": physical["canonical_source_signal_sha256"],
            "source_tensor_sha256": physical["canonical_source_tensor_sha256"],
            "canonical_physical_equivalence_sha256": physical[
                "canonical_physical_equivalence_sha256"
            ],
            "physical_unit_at_detector_boundary": "V",
        },
        "observed_roster_authority": roster,
        "common_sampling_clock_authority": clock,
        "electrical_reference_system_authority": electrical,
        "EEG_only_channel_QC_authority": qc,
        "policy_route_authorized": True,
        "provider_transform_authorized": False,
        "scope_receipt": {
            "EEG_samples_used": False,
            "EEG_electrical_reference_provenance_used": False,
            "EEG_electrical_reference_provenance_status_used_as_gate": True,
            "seizure_target_or_reference_label_used": False,
            "EDF_annotation_used": False,
            "spreadsheet_or_doctor_text_used": False,
            "clinical_history_used": False,
            "patient_or_subject_identity_used_for_route": False,
            "detector_posterior_used_for_route": False,
            "lineage_values_used_as_model_features": False,
        },
        "receipt_sha256": _PENDING,
    }
    identity_source = deepcopy(body)
    identity_source["authority_id"] = _PENDING
    identity_source["receipt_sha256"] = _PENDING
    body["authority_id"] = "DETSIGAUTH-" + _sha256(identity_source)[:24]
    body = _content_address(body)
    _validate_authority_receipt_semantics(body)
    return body


def authorize_detector_policy_lineage_from_canonical_audit(
    anchor: CanonicalPolicyAuditTrustAnchor,
    *,
    analysis_identity_id: str,
) -> ValidatedDetectorSignalLineageAuthority:
    """Issue a policy-route-only authority for one verified corpus outcome."""

    return _seal_authority(
        _build_policy_only_authority_receipt(
            anchor, analysis_identity_id=analysis_identity_id
        )
    )


def validate_detector_signal_lineage_authority_receipt(
    payload: object,
    *,
    canonical_record: CanonicalEEGRecord | None = None,
    policy_audit_anchor: CanonicalPolicyAuditTrustAnchor | None = None,
) -> ValidatedDetectorSignalLineageAuthority:
    """Validate serialized authority only against external source evidence.

    Calling this function without exactly one external authority is rejected.
    A content hash therefore cannot self-prove that a roster, reference graph
    or QC decision came from the corresponding EEG.
    """

    if (canonical_record is None) == (policy_audit_anchor is None):
        raise PermissionError(
            "serialized signal-lineage receipt requires exactly one external "
            "canonical-record or validated-audit authority"
        )
    if type(payload) is not dict:
        raise TypeError("serialized signal-lineage authority must be an object")
    if canonical_record is not None:
        expected = _build_provider_authority_receipt(canonical_record)
    else:
        identity = payload.get("canonical_physical_signal", {}).get(
            "analysis_identity_id"
        )
        if not isinstance(identity, str):
            raise ValueError("policy authority has no analysis identity")
        expected = _build_policy_only_authority_receipt(
            policy_audit_anchor, analysis_identity_id=identity  # type: ignore[arg-type]
        )
    if payload != expected:
        raise ValueError("signal-lineage authority disagrees with external replay")
    return _seal_authority(expected)


def _validate_authority_receipt_semantics(
    value: object, *, canonical: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "authority_id",
        "authority_tier",
        "trust_binding",
        "canonical_physical_signal",
        "observed_roster_authority",
        "common_sampling_clock_authority",
        "electrical_reference_system_authority",
        "EEG_only_channel_QC_authority",
        "policy_route_authorized",
        "provider_transform_authorized",
        "scope_receipt",
        "receipt_sha256",
    }
    data = _validate_content_address(
        value, required=required, context="detector signal-lineage authority"
    )
    if data["schema_version"] != SCHEMA_VERSION or data["method_id"] != METHOD_ID:
        raise ValueError("detector signal-lineage authority schema drifted")
    if data["authority_tier"] not in {
        "provider_transform_payload_replayed",
        "canonical_audit_policy_route_only",
    }:
        raise ValueError("detector signal-lineage authority tier drifted")
    expected_identity = deepcopy(data)
    expected_identity["authority_id"] = _PENDING
    expected_identity["receipt_sha256"] = _PENDING
    expected_id = "DETSIGAUTH-" + _sha256(expected_identity)[:24]
    if data["authority_id"] != expected_id:
        raise ValueError("detector signal-lineage authority ID drifted")
    if data["policy_route_authorized"] is not True:
        raise PermissionError("detector authority does not authorize policy routing")
    trust = data["trust_binding"]
    if type(trust) is not dict or trust.get("self_hash_only_accepted") is not False:
        raise ValueError("detector authority external trust binding drifted")

    policy_only = data["authority_tier"] == "canonical_audit_policy_route_only"
    roster_required = (
        {
            "schema_version",
            "source_canonical_outcome_receipt_sha256",
            "canonical_source_tensor_sha256",
            "observed_standard_channel_ids",
            "carrier_axis_mapping",
            "synthetic_or_imputed_channel_count",
            "receipt_sha256",
        }
        if policy_only
        else {
            "schema_version",
            "source_header_receipt_sha256",
            "canonical_signal_receipt_sha256",
            "canonical_source_tensor_sha256",
            "observed_standard_channel_ids",
            "carrier_axis_mapping",
            "synthetic_or_imputed_channel_count",
            "receipt_sha256",
        }
    )
    roster = _validate_content_address(
        data["observed_roster_authority"],
        required=roster_required,
        context="observed-roster authority",
    )
    if (
        roster["schema_version"] != OBSERVED_ROSTER_SCHEMA_VERSION
        or roster["synthetic_or_imputed_channel_count"] != 0
    ):
        raise ValueError("observed-roster authority semantics drifted")
    observed = _canonical_roster(
        roster["observed_standard_channel_ids"], context="authority observed roster"
    )
    mapping = roster["carrier_axis_mapping"]
    if (
        not isinstance(mapping, list)
        or [row.get("carrier_axis_index") for row in mapping]
        != list(range(len(observed)))
        or [row.get("channel_id") for row in mapping] != observed
    ):
        raise ValueError("observed-roster carrier-axis mapping drifted")
    if roster["canonical_source_tensor_sha256"] != data[
        "canonical_physical_signal"
    ]["source_tensor_sha256"]:
        raise ValueError("canonical tensor binding differs across authority lanes")

    clock_required = {
        "schema_version",
        (
            "source_canonical_outcome_receipt_sha256"
            if policy_only
            else "source_header_receipt_sha256"
        ),
        "sampling_rate_fraction_hz",
        "sample_count",
        "all_observed_carriers_share_exact_clock",
        "receipt_sha256",
    }
    clock = _validate_content_address(
        data["common_sampling_clock_authority"],
        required=clock_required,
        context="common sample-clock authority",
    )
    rate = clock["sampling_rate_fraction_hz"]
    if (
        clock["schema_version"] != CLOCK_SCHEMA_VERSION
        or not isinstance(rate, list)
        or len(rate) != 2
        or type(rate[0]) is not int
        or type(rate[1]) is not int
        or rate[0] <= 0
        or rate[1] <= 0
        or type(clock["sample_count"]) is not int
        or clock["sample_count"] <= 0
        or clock["all_observed_carriers_share_exact_clock"] is not True
    ):
        raise ValueError("common sample-clock authority semantics drifted")

    electrical_required = (
        {
            "schema_version",
            "source_canonical_outcome_receipt_sha256",
            "source_header_receipt_sha256",
            "system_status",
            "electrical_reference_system_id",
            "reference_token",
            "common_reference_carrier_subtraction_authorized",
            "semantics",
            "receipt_sha256",
        }
        if policy_only
        else {
            "schema_version",
            "source_montage_reference_receipt_sha256",
            "source_signal_sha256",
            "system_status",
            "electrical_reference_system_id",
            "reference_token",
            "montage_class",
            "acquisition_reference_matrix_sha256",
            "common_reference_carrier_subtraction_authorized",
            "semantics",
            "receipt_sha256",
        }
    )
    electrical = _validate_content_address(
        data["electrical_reference_system_authority"],
        required=electrical_required,
        context="EEG electrical-reference authority",
    )
    if (
        electrical["schema_version"] != ELECTRICAL_REFERENCE_SCHEMA_VERSION
        or "EEG_electrical_reference_provenance" not in electrical["semantics"]
    ):
        raise ValueError("EEG electrical-reference authority semantics drifted")

    qc = _validate_qc_authority(
        data["EEG_only_channel_QC_authority"],
        canonical=canonical,
        observed_channel_ids=observed,
        policy_only=policy_only,
    )
    usable = _canonical_roster(
        qc["usable_standard_channel_ids"], context="authority usable roster"
    )
    if not set(usable).issubset(observed):
        raise ValueError("EEG-only usable roster adds an unobserved carrier")

    if policy_only:
        if (
            data["provider_transform_authorized"] is not False
            or electrical["common_reference_carrier_subtraction_authorized"]
            is not False
            or trust.get("outcome_membership_semantically_replayed") is not True
        ):
            raise ValueError("policy-only authority was promoted to provider input")
        expected_scope = {
            "EEG_samples_used": False,
            "EEG_electrical_reference_provenance_used": False,
            "EEG_electrical_reference_provenance_status_used_as_gate": True,
            "seizure_target_or_reference_label_used": False,
            "EDF_annotation_used": False,
            "spreadsheet_or_doctor_text_used": False,
            "clinical_history_used": False,
            "patient_or_subject_identity_used_for_route": False,
            "detector_posterior_used_for_route": False,
            "lineage_values_used_as_model_features": False,
        }
    else:
        expected_provider = bool(
            electrical["system_status"]
            == "common_compatible_referential_qualified"
            and electrical["common_reference_carrier_subtraction_authorized"] is True
        )
        if data["provider_transform_authorized"] is not expected_provider:
            raise ValueError("provider transform permission disagrees with EEG reference")
        if trust.get("actual_payload_hash_replayed") is not True:
            raise ValueError("provider authority did not replay canonical payload")
        expected_scope = {
            "EEG_samples_used": True,
            "EEG_electrical_reference_provenance_used": True,
            "seizure_target_or_reference_label_used": False,
            "EDF_annotation_used": False,
            "spreadsheet_or_doctor_text_used": False,
            "clinical_history_used": False,
            "patient_or_subject_identity_used_for_route": False,
            "detector_posterior_used_for_route": False,
            "lineage_values_used_as_model_features": False,
        }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("detector signal-lineage evidence firewall drifted")
    return data


def verify_provider_referential_payload(
    authority: ValidatedDetectorSignalLineageAuthority,
    payload: object,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, int]]:
    """Verify actual volts against typed source identity before ST transform."""

    receipt = require_validated_detector_signal_lineage_authority(authority)
    if receipt["provider_transform_authorized"] is not True:
        raise PermissionError(
            "signal-lineage authority is policy-route-only or has an unqualified "
            "EEG electrical reference system"
        )
    observed = tuple(
        receipt["observed_roster_authority"]["observed_standard_channel_ids"]
    )
    raw = np.asarray(payload)
    if raw.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("referential EEG carrier must be float32 or float64 volts")
    if raw.ndim != 2 or raw.shape[0] != len(observed) or raw.shape[1] <= 0:
        raise ValueError("referential EEG payload shape disagrees with typed authority")
    if not np.isfinite(raw).all():
        raise ValueError("referential EEG payload contains nonfinite values")
    expected_count = receipt["common_sampling_clock_authority"]["sample_count"]
    if raw.shape[1] != expected_count:
        raise ValueError("referential EEG payload length disagrees with typed clock")
    canonical_float32 = np.ascontiguousarray(raw, dtype="<f4")
    observed_tensor = torch.from_numpy(canonical_float32).contiguous()
    observed_hash = canonical_source_tensor_sha256(
        observed_tensor, channel_ids=observed
    )
    expected_hash = receipt["canonical_physical_signal"]["source_tensor_sha256"]
    if observed_hash != expected_hash:
        raise ValueError("referential EEG payload differs from canonical source tensor")
    rate = tuple(
        int(item)
        for item in receipt["common_sampling_clock_authority"][
            "sampling_rate_fraction_hz"
        ]
    )
    # Provider arithmetic starts from the exact canonical float32-volts
    # payload that was content-addressed, promoted deterministically to
    # float64.  Sub-float32 perturbations in a caller-owned float64 array can
    # therefore never change the transform while retaining the same source
    # identity.
    return (
        np.ascontiguousarray(canonical_float32, dtype="<f8"),
        observed,
        rate,  # type: ignore[return-value]
    )


__all__ = [
    "CanonicalPolicyAuditTrustAnchor",
    "ELECTRICAL_REFERENCE_SCHEMA_VERSION",
    "METHOD_ID",
    "OBSERVED_ROSTER_SCHEMA_VERSION",
    "POLICY_AUDIT_TRUST_SCHEMA_VERSION",
    "QC_OPERATOR_ID",
    "QC_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ValidatedDetectorSignalLineageAuthority",
    "authorize_detector_policy_lineage_from_canonical_audit",
    "authorize_detector_signal_lineage_from_canonical_record",
    "load_canonical_policy_audit_trust_anchor",
    "require_validated_detector_signal_lineage_authority",
    "validate_detector_signal_lineage_authority_receipt",
    "verify_provider_referential_payload",
]
