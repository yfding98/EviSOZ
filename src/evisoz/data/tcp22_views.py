"""Signed TCP22 routing, materialization and dual-montage receipts.

The functions in this module preserve bipolar orientation and keep transport
signals separate from evidence.  Spherical/interpolated channels may be used
by a later robustness model, but they are never marked observed and cannot
support a phase-reversal or direct localization claim.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from src.clinical_eeg_long_recording.montage_reference_observability import (
    MONTAGE_REFERENCE_OBSERVABILITY_SCHEMA_VERSION,
    validate_montage_reference_observability_receipt,
)
from src.soz.geometry import STANDARD_19, normalize_electrode_name

from .artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
)
from .channel_registry import (
    CHANNEL_REGISTRY_SCHEMA_VERSION,
    build_default_channel_registry,
    validate_channel_registry,
)
from .event_identity import (
    EVENT_IDENTITY_SCHEMA_VERSION,
    validate_event_identity,
)
from .opaque_reference_authority import (
    OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION,
    validate_opaque_reference_event_authorization,
)


MONTAGE_DERIVATION_RECEIPT_SCHEMA_VERSION = (
    "evisoz_montage_derivation_receipt_v1"
)
TCP22_ROUTING_SCHEMA_VERSION = "evisoz_tcp22_signed_routing_v1"

EDGE_SUPPORT_STATES = (
    "native_bipolar",
    "exact_derived_from_common_reference",
    "exact_derived_from_protocol_authorized_opaque_common_reference",
    "interpolated_transport",
    "unobserved",
)
EVIDENCE_ELIGIBLE_EDGE_STATES = {
    "native_bipolar",
    "exact_derived_from_common_reference",
    "exact_derived_from_protocol_authorized_opaque_common_reference",
}
ELECTRODE_TRANSPORT_STATES = ("observed", "interpolated")

_ID_PREFIX = "EVISOZ-MONTAGE-"
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_LABEL_CLEAN = re.compile(r"[^A-Z0-9-]+")
_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "parent_signal_ref",
    "channel_registry_ref",
    "event_identity_ref",
    "reference_observability",
    "clock",
    "input_profile",
    "views",
    "edge_support",
    "permissions",
    "receipt_sha256",
}

_COMMON_REFERENCE_ENDPOINTS = frozenset((*STANDARD_19, "A1", "A2"))


def _clean_label(value: object) -> str:
    text = str(value).strip().upper().replace("_", "-")
    if text.startswith("EEG ") or text.startswith("EEG-"):
        text = text[4:]
    text = _LABEL_CLEAN.sub("", text).strip("-")
    if not text:
        raise ValueError("channel label must be non-empty")
    return text


def _parse_bipolar_label(value: object) -> tuple[str, str]:
    text = _clean_label(value)
    parts = text.split("-")
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"not a two-endpoint bipolar label: {value!r}")
    return normalize_electrode_name(parts[0]), normalize_electrode_name(parts[1])


def _target_edges(registry: Mapping[str, object]) -> list[tuple[str, str, str]]:
    validated = validate_channel_registry(registry)
    return [
        (
            str(row["positive_electrode"]["normalized"]),
            str(row["negative_electrode"]["normalized"]),
            str(row["unit_id"]),
        )
        for row in validated["tcp22_derivations"]
    ]


def route_tcp22_source_labels(
    source_labels: Sequence[object],
    *,
    channel_registry: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Route an arbitrary bipolar inventory into frozen TCP22 orientation.

    A reversed input derivation receives ``sign=-1``.  Duplicate exact or
    reversed observations for one target edge fail closed instead of being
    averaged.
    """

    if isinstance(source_labels, (str, bytes)) or not isinstance(
        source_labels, Sequence
    ):
        raise TypeError("source_labels must be an ordered array")
    registry = validate_channel_registry(
        channel_registry if channel_registry is not None else build_default_channel_registry()
    )
    parsed: list[tuple[str, str]] = []
    originals: list[str] = []
    for index, label in enumerate(source_labels):
        try:
            parsed.append(_parse_bipolar_label(label))
        except ValueError as exc:
            raise ValueError(f"source_labels[{index}] is not a valid bipolar row") from exc
        originals.append(str(label).strip())

    rows: list[dict[str, object]] = []
    for target_index, (positive, negative, unit_id) in enumerate(
        _target_edges(registry)
    ):
        matches: list[tuple[int, int]] = []
        for source_index, pair in enumerate(parsed):
            if pair == (positive, negative):
                matches.append((source_index, 1))
            elif pair == (negative, positive):
                matches.append((source_index, -1))
        if len(matches) > 1:
            raise ValueError(
                f"multiple source derivations map to TCP22 target {target_index}"
            )
        if not matches:
            rows.append(
                {
                    "target_index": target_index,
                    "edge_unit_id": unit_id,
                    "source_index": None,
                    "source_label": None,
                    "sign": 0,
                    "routing_status": "unobserved",
                    "orientation_ok": False,
                    "evidence_eligible": False,
                }
            )
            continue
        source_index, sign = matches[0]
        rows.append(
            {
                "target_index": target_index,
                "edge_unit_id": unit_id,
                "source_index": source_index,
                "source_label": originals[source_index],
                "sign": sign,
                "routing_status": (
                    "native_orientation" if sign == 1 else "reversed_sign_corrected"
                ),
                "orientation_ok": True,
                "evidence_eligible": True,
            }
        )
    payload = {
        "schema_version": TCP22_ROUTING_SCHEMA_VERSION,
        "source_labels": originals,
        "source_label_count": len(source_labels),
        "target_edge_count": 22,
        "formula": "target_positive_minus_target_negative",
        "channel_registry_ref": build_json_artifact_ref(
            registry,
            artifact_kind="channel_registry",
            payload_schema_version=CHANNEL_REGISTRY_SCHEMA_VERSION,
        ),
        "rows": rows,
    }
    payload["routing_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_tcp22_routing(
    value: object,
    *,
    channel_registry: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Replay every routing row from its bound source-label inventory."""

    if type(value) is not dict or set(value) != {
        "schema_version",
        "source_labels",
        "source_label_count",
        "target_edge_count",
        "formula",
        "channel_registry_ref",
        "rows",
        "routing_sha256",
    }:
        raise ValueError("TCP22 routing fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != TCP22_ROUTING_SCHEMA_VERSION:
        raise ValueError("TCP22 routing schema_version drifted")
    if not isinstance(data["source_labels"], list):
        raise TypeError("TCP22 routing source_labels must be an array")
    registry = validate_channel_registry(
        channel_registry if channel_registry is not None else build_default_channel_registry()
    )
    expected_registry_ref = build_json_artifact_ref(
        registry,
        artifact_kind="channel_registry",
        payload_schema_version=CHANNEL_REGISTRY_SCHEMA_VERSION,
    )
    if validate_artifact_ref(data["channel_registry_ref"]) != expected_registry_ref:
        raise ValueError("TCP22 routing channel registry binding drifted")
    expected = route_tcp22_source_labels(
        data["source_labels"],
        channel_registry=registry,
    )
    if data != expected:
        raise ValueError("TCP22 routing rows do not replay from source labels")
    return data


def materialize_routed_tcp22(
    source_waveforms: np.ndarray,
    routing: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a signed routing receipt to bipolar waveforms.

    Missing rows are zero transport values with a false mask.  Callers must
    carry the returned mask through every downstream operation.
    """

    values = np.asarray(source_waveforms)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise TypeError("source_waveforms must have numeric shape [channels,time]")
    if not np.isfinite(values).all():
        raise ValueError("source_waveforms must be finite")
    validated_routing = validate_tcp22_routing(routing)
    rows = validated_routing["rows"]
    source_count = validated_routing["source_label_count"]
    if source_count != values.shape[0]:
        raise ValueError("routing source count does not match waveform rows")

    output = np.zeros((22, values.shape[1]), dtype=values.dtype)
    mask = np.zeros(22, dtype=bool)
    for expected_index, row in enumerate(rows):
        if type(row) is not dict or row.get("target_index") != expected_index:
            raise ValueError("routing target order drifted")
        source_index = row.get("source_index")
        sign = row.get("sign")
        if source_index is None:
            if sign != 0 or row.get("evidence_eligible") is not False:
                raise ValueError("missing routing row is not fail-closed")
            continue
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise TypeError("routing source index must be an integer")
        if not 0 <= source_index < values.shape[0] or sign not in (-1, 1):
            raise ValueError("routing index/sign is invalid")
        if row.get("orientation_ok") is not True or row.get("evidence_eligible") is not True:
            raise ValueError("observed routing row lost orientation/evidence closure")
        output[expected_index] = values[source_index] * sign
        mask[expected_index] = True
    return output, mask


def _validated_common_reference_observability(
    receipt: Mapping[str, object],
    *,
    electrode_names: Sequence[object] | None = None,
    expected_source_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate one explicit common-reference label-observability receipt.

    The shared clinical receipt deliberately treats explicitly referenced
    auxiliary channels as non-standard observations.  TCP22 nevertheless uses
    A1 and A2 as real edge endpoints, so this adapter additionally requires
    every selected Standard19/A1/A2 input to expose the same reference token.
    """

    observed = validate_montage_reference_observability_receipt(receipt)
    compatibility = observed["common_reference_compatibility"]
    if (
        observed["montage_class"] != "common_compatible_referential"
        or compatibility["compatible"] is not True
    ):
        raise ValueError(
            "TCP22 common-reference derivation requires a compatible "
            "referential montage receipt"
        )
    reference_token = compatibility["reference_token"]
    if not isinstance(reference_token, str) or not reference_token:
        raise ValueError("common-reference receipt lacks an observable token")
    if (
        expected_source_sha256 is not None
        and observed["source_signal_sha256"] != expected_source_sha256
    ):
        raise ValueError("common-reference receipt belongs to a different parent signal")

    observations = observed["signal_label_observations"]
    labels = [str(row["raw_label"]) for row in observations]
    if electrode_names is not None:
        supplied_labels = [str(label).strip() for label in electrode_names]
        if supplied_labels != labels:
            raise ValueError(
                "common-reference receipt does not bind the ordered electrode labels"
            )

    endpoint_tokens: dict[str, str] = {}
    for row in observations:
        role = row["signal_role"]
        if role == "direct_standard_electrode":
            endpoint = str(row["positive_electrode"])
        elif role == "non_standard_referential_ignored":
            raw_label = str(row["raw_label"])
            try:
                endpoint = normalize_electrode_name(raw_label)
            except ValueError:
                continue
            if endpoint not in {"A1", "A2"}:
                continue
        else:
            continue
        token = row["reference_token"]
        if endpoint not in _COMMON_REFERENCE_ENDPOINTS or token != reference_token:
            raise ValueError(
                "TCP22 endpoint has mixed or unknown acquisition reference"
            )
        if endpoint in endpoint_tokens:
            raise ValueError(f"duplicate common-reference endpoint: {endpoint}")
        endpoint_tokens[endpoint] = str(token)

    if electrode_names is not None:
        normalized_inputs: list[str] = []
        for raw_name in electrode_names:
            normalized = normalize_electrode_name(raw_name)
            if normalized not in _COMMON_REFERENCE_ENDPOINTS:
                raise ValueError(
                    f"unsupported TCP22 common-reference electrode: {normalized}"
                )
            normalized_inputs.append(normalized)
        if len(normalized_inputs) != len(set(normalized_inputs)):
            raise ValueError("duplicate normalized common-reference electrode identity")
        if set(normalized_inputs) != set(endpoint_tokens):
            raise ValueError(
                "common-reference receipt does not observe every input electrode"
            )
    return observed, endpoint_tokens


def derive_tcp22_from_common_reference(
    electrode_waveforms: np.ndarray,
    electrode_names: Sequence[object],
    *,
    electrode_states: Sequence[str],
    montage_reference_observability_receipt: Mapping[str, object],
    channel_registry: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Derive signed TCP22 rows from a common-reference electrode field.

    Interpolated endpoints produce transport-only values and a false evidence
    mask.  Missing endpoints produce zeros and ``unobserved`` status.
    """

    values = np.asarray(electrode_waveforms)
    if values.ndim != 2 or values.shape[0] != len(electrode_names):
        raise ValueError("electrode_waveforms and electrode_names must align")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("electrode waveforms must be finite numeric values")
    _validated_common_reference_observability(
        montage_reference_observability_receipt,
        electrode_names=electrode_names,
    )
    states = list(electrode_states)
    if len(states) != len(electrode_names) or any(
        state not in ELECTRODE_TRANSPORT_STATES for state in states
    ):
        raise ValueError("electrode_states must be observed/interpolated and align")

    by_name: dict[str, tuple[int, str]] = {}
    for index, raw_name in enumerate(electrode_names):
        normalized = normalize_electrode_name(raw_name)
        if normalized in by_name:
            raise ValueError(f"duplicate normalized electrode identity: {normalized}")
        by_name[normalized] = (index, states[index])

    registry = validate_channel_registry(
        channel_registry if channel_registry is not None else build_default_channel_registry()
    )
    output = np.zeros((22, values.shape[1]), dtype=values.dtype)
    edge_states: list[str] = []
    evidence_mask = np.zeros(22, dtype=bool)
    for edge_index, row in enumerate(registry["tcp22_derivations"]):
        positive = str(row["positive_electrode"]["normalized"])
        negative = str(row["negative_electrode"]["normalized"])
        if positive not in by_name or negative not in by_name:
            edge_states.append("unobserved")
            continue
        pos_index, pos_state = by_name[positive]
        neg_index, neg_state = by_name[negative]
        output[edge_index] = values[pos_index] - values[neg_index]
        if pos_state == neg_state == "observed":
            edge_states.append("exact_derived_from_common_reference")
            evidence_mask[edge_index] = True
        else:
            edge_states.append("interpolated_transport")
    return output, edge_states, evidence_mask


def _receipt_id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_id"] = _PENDING_ID
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _receipt_hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _reference_observability_binding(
    receipt: Mapping[str, object] | None,
    *,
    parent_content_sha256: str,
    edge_states: Sequence[str],
    registry: Mapping[str, object],
) -> dict[str, Any] | None:
    common_reference_states = {
        "exact_derived_from_common_reference",
        "exact_derived_from_protocol_authorized_opaque_common_reference",
        "interpolated_transport",
    }
    common_reference_used = any(
        state in common_reference_states for state in edge_states
    )
    if not common_reference_used:
        if receipt is not None:
            raise ValueError(
                "common-reference observability was supplied without a derived edge"
            )
        return None
    if receipt is None:
        raise ValueError(
            "common-reference TCP22 edges require an explicit observability receipt"
        )
    if (
        receipt.get("schema_version")
        == OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION
    ):
        observed = validate_opaque_reference_event_authorization(
            receipt,
            expected_parent_signal_sha256=parent_content_sha256,
        )
        endpoint_tokens = {
            str(endpoint): "OPAQUE_PROTOCOL_AUTHORIZED"
            for endpoint in observed["observed_parent_electrodes"]
        }
        if any(
            state == "exact_derived_from_common_reference"
            for state in edge_states
        ):
            raise ValueError(
                "header-exact edge support cannot use an opaque reference authority"
            )
        artifact_kind = "opaque_common_reference_event_authorization"
        payload_schema_version = (
            OPAQUE_REFERENCE_EVENT_AUTHORIZATION_SCHEMA_VERSION
        )
    else:
        observed, endpoint_tokens = _validated_common_reference_observability(
            receipt,
            expected_source_sha256=parent_content_sha256,
        )
        if any(
            state
            == "exact_derived_from_protocol_authorized_opaque_common_reference"
            for state in edge_states
        ):
            raise ValueError(
                "opaque edge support requires an opaque reference authority"
            )
        artifact_kind = "montage_reference_observability"
        payload_schema_version = MONTAGE_REFERENCE_OBSERVABILITY_SCHEMA_VERSION
    for index, state in enumerate(edge_states):
        if state not in common_reference_states:
            continue
        edge = registry["tcp22_derivations"][index]
        positive = str(edge["positive_electrode"]["normalized"])
        negative = str(edge["negative_electrode"]["normalized"])
        if positive not in endpoint_tokens or negative not in endpoint_tokens:
            raise ValueError(
                f"TCP22 derived edge {index} lacks observable common-reference endpoints"
            )
    return {
        "artifact_ref": build_json_artifact_ref(
            observed,
            artifact_kind=artifact_kind,
            payload_schema_version=payload_schema_version,
        ),
        "receipt": observed,
    }


def _bool_mask(value: Sequence[object], length: int, context: str) -> list[bool]:
    rows = list(value)
    if len(rows) != length or any(type(item) is not bool for item in rows):
        raise ValueError(f"{context} must be a {length}-item boolean mask")
    return rows


def _view(
    *,
    artifact_ref: Mapping[str, object] | None,
    coordinate: str,
    unit_count: int,
    sample_interval: tuple[int, int],
    time_interval_seconds: tuple[float, float],
    parent_artifact_id: str,
    preprocessing_scope: str,
    context_artifact_dependency: str | None,
    unit_observed_mask: Sequence[bool],
) -> dict[str, object]:
    return {
        "artifact_ref": (
            validate_artifact_ref(artifact_ref) if artifact_ref is not None else None
        ),
        "coordinate": coordinate,
        "shape": [unit_count, sample_interval[1] - sample_interval[0]],
        "sample_interval": list(sample_interval),
        "time_interval_seconds": list(time_interval_seconds),
        "source_parent_artifact_id": parent_artifact_id,
        "preprocessing_scope": preprocessing_scope,
        "context_artifact_dependency": context_artifact_dependency,
        "unit_observed_mask": list(unit_observed_mask),
    }


def build_montage_derivation_receipt(
    *,
    parent_signal_ref: Mapping[str, object],
    event_identity: Mapping[str, object],
    v29_tensor_ref: Mapping[str, object] | None,
    tcp22_context_tensor_ref: Mapping[str, object] | None,
    tcp22_onset_tensor_ref: Mapping[str, object] | None,
    standard19_observed_mask: Sequence[object],
    tcp22_edge_states: Sequence[str],
    tcp22_orientation_ok: Sequence[object],
    montage_reference_observability_receipt: Mapping[str, object] | None = None,
    channel_registry: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Bind CAR19 and TCP22 sibling artifacts to one exact 200 Hz clock."""

    parent = validate_artifact_ref(parent_signal_ref)
    if (
        parent["artifact_kind"] != "canonical_signal"
        or parent["content_hash"]["domain"] != "raw_bytes_v1"
    ):
        raise ValueError("montage parent must be a raw canonical_signal")
    identity = validate_event_identity(event_identity)
    if identity["parent_signal_ref"] != parent:
        raise ValueError("montage and event identity parent signals differ")
    event_identity_ref = build_json_artifact_ref(
        identity,
        artifact_kind="event_identity",
        payload_schema_version=EVENT_IDENTITY_SCHEMA_VERSION,
    )
    registry = validate_channel_registry(
        channel_registry if channel_registry is not None else build_default_channel_registry()
    )
    registry_ref = build_json_artifact_ref(
        registry,
        artifact_kind="channel_registry",
        payload_schema_version=CHANNEL_REGISTRY_SCHEMA_VERSION,
    )
    node_mask = _bool_mask(standard19_observed_mask, 19, "standard19_observed_mask")
    edge_states = list(tcp22_edge_states)
    if len(edge_states) != 22 or any(state not in EDGE_SUPPORT_STATES for state in edge_states):
        raise ValueError("tcp22_edge_states must contain 22 supported states")
    if tcp22_orientation_ok is None:
        raise ValueError("TCP22 orientation receipt must be supplied explicitly")
    orientation = _bool_mask(tcp22_orientation_ok, 22, "tcp22_orientation_ok")
    if any(
        state == "unobserved" and orientation[index]
        for index, state in enumerate(edge_states)
    ):
        raise ValueError("unobserved TCP22 edges cannot assert known orientation")
    edge_evidence_mask = [
        state in EVIDENCE_ELIGIBLE_EDGE_STATES and orientation[index]
        for index, state in enumerate(edge_states)
    ]
    edge_transport_mask = [state != "unobserved" for state in edge_states]
    if v29_tensor_ref is not None and not all(node_mask):
        raise ValueError(
            "a formal v29 tensor requires all 19 directly observed electrodes"
        )
    tcp_refs_present = (
        tcp22_context_tensor_ref is not None,
        tcp22_onset_tensor_ref is not None,
    )
    if tcp_refs_present[0] != tcp_refs_present[1]:
        raise ValueError("TCP22 context and source-isolated onset artifacts are inseparable")
    if any(edge_transport_mask) != all(tcp_refs_present):
        raise ValueError(
            "TCP22 support states and materialized sibling artifacts disagree"
        )
    reference_observability = _reference_observability_binding(
        montage_reference_observability_receipt,
        parent_content_sha256=str(parent["content_hash"]["sha256"]),
        edge_states=edge_states,
        registry=registry,
    )
    v29_available = v29_tensor_ref is not None
    tcp22_available = all(tcp_refs_present)
    if v29_available and all(edge_evidence_mask):
        profile = "dual_native"
    elif v29_available:
        profile = "standard19_native"
    elif all(edge_evidence_mask):
        profile = "tcp22_native"
    else:
        profile = "partial_masked"

    parent_id = str(parent["artifact_id"])
    views = {
        "v29_reference": _view(
            artifact_ref=v29_tensor_ref,
            coordinate="standard19_electrode_node",
            unit_count=19,
            sample_interval=(0, 12000),
            time_interval_seconds=(-12.0, 48.0),
            parent_artifact_id=parent_id,
            preprocessing_scope="frozen_canonical_v29_car19_full_window",
            context_artifact_dependency=None,
            unit_observed_mask=node_mask,
        ),
        "tcp22_context": _view(
            artifact_ref=tcp22_context_tensor_ref,
            coordinate="signed_tcp22_bipolar_edge",
            unit_count=22,
            sample_interval=(0, 12000),
            time_interval_seconds=(-12.0, 48.0),
            parent_artifact_id=parent_id,
            preprocessing_scope="tcp22_context_full_window",
            context_artifact_dependency=None,
            unit_observed_mask=edge_transport_mask,
        ),
        "tcp22_onset": _view(
            artifact_ref=tcp22_onset_tensor_ref,
            coordinate="signed_tcp22_bipolar_edge",
            unit_count=22,
            sample_interval=(2000, 4000),
            time_interval_seconds=(-2.0, 8.0),
            parent_artifact_id=parent_id,
            preprocessing_scope="core_local_frozen_receipt",
            context_artifact_dependency=None,
            unit_observed_mask=edge_transport_mask,
        ),
    }
    edge_support = [
        {
            "edge_unit_id": registry["tcp22_derivations"][index]["unit_id"],
            "support_state": state,
            "orientation_ok": orientation[index],
            "transport_available": edge_transport_mask[index],
            "direct_evidence_eligible": edge_evidence_mask[index],
        }
        for index, state in enumerate(edge_states)
    ]
    body: dict[str, Any] = {
        "schema_version": MONTAGE_DERIVATION_RECEIPT_SCHEMA_VERSION,
        "receipt_id": _PENDING_ID,
        "parent_signal_ref": parent,
        "channel_registry_ref": registry_ref,
        "event_identity_ref": event_identity_ref,
        "reference_observability": reference_observability,
        "clock": {
            "sampling_rate_numerator": 200,
            "sampling_rate_denominator": 1,
            "sample_count": 12000,
            "anchor_sample_index": 2400,
            "analysis_interval_seconds": [-12.0, 48.0],
        },
        "input_profile": profile,
        "views": views,
        "edge_support": edge_support,
        "permissions": {
            "v29_reference_available": v29_available,
            "residual_main_analysis_eligible": (
                v29_available and tcp22_available and any(edge_evidence_mask)
            ),
            "tcp22_standalone_evidence_available": (
                tcp22_available and any(edge_evidence_mask)
            ),
            "interpolated_transport_is_direct_evidence": False,
            "edge_to_endpoint_expansion_authorized": False,
            "onset_source_isolated_from_context_artifact": True,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["receipt_id"] = _ID_PREFIX + canonical_json_sha256(_receipt_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_receipt_hash_source(body))
    return validate_montage_derivation_receipt(
        body,
        channel_registry=registry,
        trusted_event_identity=identity,
    )


def validate_montage_derivation_receipt(
    value: object,
    *,
    channel_registry: Mapping[str, object] | None = None,
    trusted_event_identity: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate clock, sibling ownership, masks and source isolation."""

    if type(value) is not dict or set(value) != _RECEIPT_KEYS:
        raise ValueError("montage derivation receipt fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != MONTAGE_DERIVATION_RECEIPT_SCHEMA_VERSION:
        raise ValueError("montage derivation schema_version drifted")
    parent = validate_artifact_ref(data["parent_signal_ref"])
    if (
        parent["artifact_kind"] != "canonical_signal"
        or parent["content_hash"]["domain"] != "raw_bytes_v1"
    ):
        raise ValueError("montage parent must be a raw canonical_signal")
    registry = validate_channel_registry(
        channel_registry if channel_registry is not None else build_default_channel_registry()
    )
    registry_ref = validate_artifact_ref(data["channel_registry_ref"])
    expected_registry_ref = build_json_artifact_ref(
        registry,
        artifact_kind="channel_registry",
        payload_schema_version=CHANNEL_REGISTRY_SCHEMA_VERSION,
    )
    if registry_ref != expected_registry_ref:
        raise ValueError("montage receipt is bound to a different channel registry")
    event_identity_ref = validate_artifact_ref(data["event_identity_ref"])
    if (
        event_identity_ref["artifact_kind"] != "event_identity"
        or event_identity_ref["content_hash"]["domain"] != "canonical_json_v1"
        or event_identity_ref["payload_schema_version"]
        != EVENT_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError("montage event_identity_ref has the wrong type")
    if trusted_event_identity is not None:
        identity = validate_event_identity(trusted_event_identity)
        if identity["parent_signal_ref"] != parent:
            raise ValueError("trusted event identity belongs to another parent signal")
        expected_identity_ref = build_json_artifact_ref(
            identity,
            artifact_kind="event_identity",
            payload_schema_version=EVENT_IDENTITY_SCHEMA_VERSION,
        )
        if event_identity_ref != expected_identity_ref:
            raise ValueError("montage event identity does not bind trusted content")
    if data["clock"] != {
        "sampling_rate_numerator": 200,
        "sampling_rate_denominator": 1,
        "sample_count": 12000,
        "anchor_sample_index": 2400,
        "analysis_interval_seconds": [-12.0, 48.0],
    }:
        raise ValueError("montage receipt clock drifted from frozen v29 support")

    views = data["views"]
    if type(views) is not dict or set(views) != {
        "v29_reference",
        "tcp22_context",
        "tcp22_onset",
    }:
        raise ValueError("montage sibling view set drifted")
    parent_id = parent["artifact_id"]
    expected_views = {
        "v29_reference": (
            "standard19_electrode_node",
            [19, 12000],
            [0, 12000],
            [-12.0, 48.0],
            "frozen_canonical_v29_car19_full_window",
        ),
        "tcp22_context": (
            "signed_tcp22_bipolar_edge",
            [22, 12000],
            [0, 12000],
            [-12.0, 48.0],
            "tcp22_context_full_window",
        ),
        "tcp22_onset": (
            "signed_tcp22_bipolar_edge",
            [22, 2000],
            [2000, 4000],
            [-2.0, 8.0],
            "core_local_frozen_receipt",
        ),
    }
    for name, expected in expected_views.items():
        row = views[name]
        if type(row) is not dict or set(row) != {
            "artifact_ref",
            "coordinate",
            "shape",
            "sample_interval",
            "time_interval_seconds",
            "source_parent_artifact_id",
            "preprocessing_scope",
            "context_artifact_dependency",
            "unit_observed_mask",
        }:
            raise ValueError(f"{name} view fields drifted")
        if row["artifact_ref"] is not None:
            artifact_ref = validate_artifact_ref(row["artifact_ref"])
            if artifact_ref["artifact_kind"] != "tensor_cache":
                raise ValueError(f"{name} must reference a typed tensor cache")
        if (
            row["coordinate"],
            row["shape"],
            row["sample_interval"],
            row["time_interval_seconds"],
            row["preprocessing_scope"],
        ) != expected:
            raise ValueError(f"{name} view geometry/preprocessing drifted")
        if row["source_parent_artifact_id"] != parent_id:
            raise ValueError(f"{name} does not share the raw parent artifact")
        if row["context_artifact_dependency"] is not None:
            raise ValueError(f"{name} must not depend on a sibling context artifact")
        expected_mask_length = 19 if name == "v29_reference" else 22
        _bool_mask(
            row["unit_observed_mask"],
            expected_mask_length,
            f"{name}.unit_observed_mask",
        )

    support = data["edge_support"]
    if not isinstance(support, list) or len(support) != 22:
        raise ValueError("edge_support must contain 22 rows")
    edge_evidence_mask: list[bool] = []
    edge_transport_mask: list[bool] = []
    edge_states: list[str] = []
    for index, row in enumerate(support):
        if type(row) is not dict or set(row) != {
            "edge_unit_id",
            "support_state",
            "orientation_ok",
            "transport_available",
            "direct_evidence_eligible",
        }:
            raise ValueError(f"edge_support[{index}] fields drifted")
        if row["edge_unit_id"] != registry["tcp22_derivations"][index]["unit_id"]:
            raise ValueError(f"edge_support[{index}] identity/order drifted")
        state = row["support_state"]
        if state not in EDGE_SUPPORT_STATES or type(row["orientation_ok"]) is not bool:
            raise ValueError(f"edge_support[{index}] state/orientation invalid")
        expected_transport = state != "unobserved"
        expected_evidence = state in EVIDENCE_ELIGIBLE_EDGE_STATES and row["orientation_ok"]
        if row["transport_available"] is not expected_transport:
            raise ValueError(f"edge_support[{index}] transport permission drifted")
        if row["direct_evidence_eligible"] is not expected_evidence:
            raise ValueError(f"edge_support[{index}] evidence permission drifted")
        edge_transport_mask.append(expected_transport)
        edge_evidence_mask.append(expected_evidence)
        edge_states.append(str(state))

    context_mask = _bool_mask(
        views["tcp22_context"]["unit_observed_mask"],
        22,
        "tcp22_context.unit_observed_mask",
    )
    onset_mask = _bool_mask(
        views["tcp22_onset"]["unit_observed_mask"],
        22,
        "tcp22_onset.unit_observed_mask",
    )
    if context_mask != edge_transport_mask or onset_mask != edge_transport_mask:
        raise ValueError(
            "TCP22 sibling masks must exactly replay edge transport support"
        )

    reference_binding = data["reference_observability"]
    if reference_binding is None:
        reference_receipt = None
    else:
        if type(reference_binding) is not dict or set(reference_binding) != {
            "artifact_ref",
            "receipt",
        }:
            raise ValueError("reference_observability binding fields drifted")
        reference_receipt = reference_binding["receipt"]
    expected_reference_binding = _reference_observability_binding(
        reference_receipt,
        parent_content_sha256=str(parent["content_hash"]["sha256"]),
        edge_states=edge_states,
        registry=registry,
    )
    if reference_binding != expected_reference_binding:
        raise ValueError("reference_observability binding does not replay")

    permissions = data["permissions"]
    if type(permissions) is not dict or set(permissions) != {
        "v29_reference_available",
        "residual_main_analysis_eligible",
        "tcp22_standalone_evidence_available",
        "interpolated_transport_is_direct_evidence",
        "edge_to_endpoint_expansion_authorized",
        "onset_source_isolated_from_context_artifact",
    }:
        raise ValueError("montage permissions fields drifted")
    for key, item in permissions.items():
        if type(item) is not bool:
            raise TypeError(f"permissions.{key} must be boolean")
    v29_available = (
        views["v29_reference"]["artifact_ref"] is not None
        and all(views["v29_reference"]["unit_observed_mask"])
    )
    context_ref = views["tcp22_context"]["artifact_ref"]
    onset_ref = views["tcp22_onset"]["artifact_ref"]
    if (context_ref is None) != (onset_ref is None):
        raise ValueError("TCP22 context/onset materialization is incomplete")
    if context_ref is not None and context_ref["artifact_id"] == onset_ref["artifact_id"]:
        raise ValueError("TCP22 onset must be a distinct source-isolated artifact")
    tcp_materialized = context_ref is not None and onset_ref is not None
    if any(edge_transport_mask) != tcp_materialized:
        raise ValueError("TCP22 materialization and edge support disagree")
    tcp_evidence_available = tcp_materialized and any(edge_evidence_mask)
    if permissions != {
        "v29_reference_available": v29_available,
        "residual_main_analysis_eligible": v29_available and tcp_evidence_available,
        "tcp22_standalone_evidence_available": tcp_evidence_available,
        "interpolated_transport_is_direct_evidence": False,
        "edge_to_endpoint_expansion_authorized": False,
        "onset_source_isolated_from_context_artifact": True,
    }:
        raise ValueError("montage permissions do not replay from the views")
    expected_profile = (
        "dual_native"
        if v29_available and all(edge_evidence_mask)
        else "standard19_native"
        if v29_available
        else "tcp22_native"
        if all(edge_evidence_mask)
        else "partial_masked"
    )
    if data["input_profile"] != expected_profile:
        raise ValueError("input_profile does not replay from view support")

    expected_id = _ID_PREFIX + canonical_json_sha256(_receipt_id_source(data))[:24]
    if data["receipt_id"] != expected_id:
        raise ValueError("montage receipt_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(_receipt_hash_source(data)):
        raise ValueError("montage receipt_sha256 does not bind its content")
    return data


__all__ = [
    "MONTAGE_DERIVATION_RECEIPT_SCHEMA_VERSION",
    "TCP22_ROUTING_SCHEMA_VERSION",
    "EDGE_SUPPORT_STATES",
    "EVIDENCE_ELIGIBLE_EDGE_STATES",
    "route_tcp22_source_labels",
    "validate_tcp22_routing",
    "materialize_routed_tcp22",
    "derive_tcp22_from_common_reference",
    "build_montage_derivation_receipt",
    "validate_montage_derivation_receipt",
]
