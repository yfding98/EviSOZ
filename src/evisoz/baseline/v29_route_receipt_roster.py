"""Patient-indexed roster of frozen canonical-v29 route receipts.

One :mod:`v29_route_receipt` seals one typed patient decision.  A reference
cache can contain several patient units, or several event units owned by
several patients, so a single receipt reference is insufficient.  This
module provides the missing batch-level join: roster-local patient indices
are contiguous, every row binds one typed identity to one route-receipt
ArtifactRef, and every receipt must share the same frozen resource config.

The serialized roster intentionally contains no raw patient identifier.
Structural validation can be performed from the roster alone.  Consumers
must additionally pass the referenced route-receipt payloads (and the frozen
resource-config bytes) before using model outputs; the numerical cache layer
can then cross-check its unit-to-patient map against these roster-local
indices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
from typing import Any

from src.evisoz.baseline.frozen_v29 import (
    FROZEN_FIVE_FOLD_ROUTE,
    HISTORICAL_PUBLIC_ROUTE,
    METHOD_ID,
    RESOURCE_CONFIG_SCHEMA_VERSION,
)
from src.evisoz.baseline.v29_route_receipt import (
    RESOURCE_CONFIG_ARTIFACT_KIND,
    ROUTE_RECEIPT_ARTIFACT_KIND,
    V29_ROUTE_RECEIPT_SCHEMA_VERSION,
    validate_v29_route_receipt,
)
from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)


V29_ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION = "evisoz_v29_route_receipt_roster_v1"
ROUTE_RECEIPT_ROSTER_ARTIFACT_KIND = "v29_route_receipt_roster"

_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-V29-ROSTER-"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")

_ROUTE_UNIT_KIND = {
    HISTORICAL_PUBLIC_ROUTE: "patient",
    FROZEN_FIVE_FOLD_ROUTE: "event",
}
_TOP_KEYS = {
    "schema_version",
    "roster_id",
    "method_id",
    "route",
    "unit_kind",
    "patient_count",
    "resource_config_ref",
    "resource_registry_projection_sha256",
    "rows",
    "index_contract",
    "replay_contract",
    "receipt_sha256",
}
_ROW_KEYS = {
    "patient_index",
    "identity_namespace",
    "identity_sha256",
    "route_receipt_ref",
}
_INDEX_CONTRACT = {
    "patient_index_scope": "roster_local",
    "ordering": "input_route_receipt_order",
    "continuous_zero_based": True,
    "unique_typed_identity_required": True,
    "raw_patient_identifiers_stored": False,
}
_REPLAY_CONTRACT = {
    "route_receipt_content_replay_required_before_use": True,
    "resource_config_content_replay_required_before_use": True,
    "route_layer_training_authorized": False,
}


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["roster_id"] = _PENDING_ID
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _namespace(value: object, context: str) -> str:
    if not isinstance(value, str) or _NAMESPACE_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a stable typed-identity namespace")
    return value


def _route_and_unit_kind(route: object, unit_kind: object) -> tuple[str, str]:
    if route not in _ROUTE_UNIT_KIND:
        raise ValueError("route receipt roster contains an unsupported route")
    expected = _ROUTE_UNIT_KIND[str(route)]
    if unit_kind != expected:
        raise ValueError(f"route {route!r} requires unit_kind={expected!r}")
    return str(route), expected


def _typed_resource_config_ref(value: object) -> dict[str, Any]:
    ref = validate_artifact_ref(value)
    if (
        ref["artifact_kind"] != RESOURCE_CONFIG_ARTIFACT_KIND
        or ref["media_type"] != "application/json"
        or ref["content_hash"]["domain"] != "raw_bytes_v1"
        or ref["payload_schema_version"] != RESOURCE_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError("route receipt roster resource_config_ref type/domain drifted")
    return ref


def _typed_route_receipt_ref(value: object, *, context: str) -> dict[str, Any]:
    ref = validate_artifact_ref(value)
    if (
        ref["artifact_kind"] != ROUTE_RECEIPT_ARTIFACT_KIND
        or ref["media_type"] != "application/json"
        or ref["content_hash"]["domain"] != "canonical_json_v1"
        or ref["payload_schema_version"] != V29_ROUTE_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError(f"{context} type/domain drifted")
    return ref


def _route_receipt_ref(receipt: Mapping[str, object]) -> dict[str, Any]:
    return build_json_artifact_ref(
        receipt,
        artifact_kind=ROUTE_RECEIPT_ARTIFACT_KIND,
        payload_schema_version=V29_ROUTE_RECEIPT_SCHEMA_VERSION,
    )


def _validated_receipt_payloads(value: object) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("route_receipts must be an ordered sequence")
    if len(value) < 1:
        raise ValueError("route_receipts must contain at least one patient")
    return [validate_v29_route_receipt(receipt) for receipt in value]


def build_v29_route_receipt_roster(
    *,
    route_receipts: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Build one content-addressed roster from ordered patient receipts.

    ``patient_index`` is local to this roster and is assigned from the input
    sequence order.  It is deliberately distinct from the historical-public
    index stored inside a public route decision.
    """

    receipts = _validated_receipt_payloads(route_receipts)
    first = receipts[0]
    route, unit_kind = _route_and_unit_kind(
        first["decision"]["route"],
        first["decision"]["unit_kind"],
    )
    resource_config_ref = _typed_resource_config_ref(first["resource_config_ref"])
    registry_projection_sha256 = _sha256(
        first["resource_registry_projection_sha256"],
        "resource_registry_projection_sha256",
    )

    rows: list[dict[str, object]] = []
    for patient_index, receipt in enumerate(receipts):
        receipt_route, receipt_unit_kind = _route_and_unit_kind(
            receipt["decision"]["route"],
            receipt["decision"]["unit_kind"],
        )
        if receipt_route != route or receipt_unit_kind != unit_kind:
            raise ValueError("route receipt roster cannot mix routes or unit kinds")
        if receipt["method_id"] != METHOD_ID:
            raise ValueError("route receipt roster method_id drifted")
        if receipt["resource_config_ref"] != resource_config_ref:
            raise ValueError("route receipts do not share one resource config")
        if (
            receipt["resource_registry_projection_sha256"]
            != registry_projection_sha256
        ):
            raise ValueError("route receipts do not share one resource registry")
        rows.append(
            {
                "patient_index": patient_index,
                "identity_namespace": _namespace(
                    receipt["decision"]["identity_namespace"],
                    "decision.identity_namespace",
                ),
                "identity_sha256": _sha256(
                    receipt["decision"]["identity_sha256"],
                    "decision.identity_sha256",
                ),
                "route_receipt_ref": _route_receipt_ref(receipt),
            }
        )

    body: dict[str, Any] = {
        "schema_version": V29_ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION,
        "roster_id": _PENDING_ID,
        "method_id": METHOD_ID,
        "route": route,
        "unit_kind": unit_kind,
        "patient_count": len(rows),
        "resource_config_ref": resource_config_ref,
        "resource_registry_projection_sha256": registry_projection_sha256,
        "rows": rows,
        "index_contract": deepcopy(_INDEX_CONTRACT),
        "replay_contract": deepcopy(_REPLAY_CONTRACT),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["roster_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_v29_route_receipt_roster(body, route_receipts=receipts)


def validate_v29_route_receipt_roster(
    value: object,
    *,
    route_receipts: Sequence[Mapping[str, object]] | None = None,
    resource_config_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate the roster and optionally replay every referenced payload.

    Passing ``route_receipts`` checks row order, typed identity, route,
    resource authority and each canonical-JSON ArtifactRef.  Passing
    ``resource_config_bytes`` additionally verifies the shared raw resource
    config.  Consumers must perform both checks before exposing cached model
    outputs, as recorded in ``replay_contract``.
    """

    if type(value) is not dict or set(value) != _TOP_KEYS:
        raise ValueError("v29 route receipt roster fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != V29_ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION:
        raise ValueError("v29 route receipt roster schema_version drifted")
    if data["method_id"] != METHOD_ID:
        raise ValueError("v29 route receipt roster method_id drifted")
    route, unit_kind = _route_and_unit_kind(data["route"], data["unit_kind"])
    patient_count = data["patient_count"]
    if (
        isinstance(patient_count, bool)
        or not isinstance(patient_count, int)
        or patient_count < 1
    ):
        raise TypeError("patient_count must be an integer >= 1")

    resource_config_ref = _typed_resource_config_ref(data["resource_config_ref"])
    data["resource_config_ref"] = resource_config_ref
    registry_projection_sha256 = _sha256(
        data["resource_registry_projection_sha256"],
        "resource_registry_projection_sha256",
    )
    rows_value = data["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != patient_count:
        raise ValueError("route receipt roster rows must match patient_count")

    identities: set[str] = set()
    receipt_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for expected_index, value_row in enumerate(rows_value):
        if type(value_row) is not dict or set(value_row) != _ROW_KEYS:
            raise ValueError(
                f"route receipt roster rows[{expected_index}] fields drifted"
            )
        row = deepcopy(value_row)
        if row["patient_index"] != expected_index or isinstance(
            row["patient_index"], bool
        ):
            raise ValueError("patient_index must be zero-based, continuous and ordered")
        row["identity_namespace"] = _namespace(
            row["identity_namespace"],
            f"rows[{expected_index}].identity_namespace",
        )
        row["identity_sha256"] = _sha256(
            row["identity_sha256"],
            f"rows[{expected_index}].identity_sha256",
        )
        row["route_receipt_ref"] = _typed_route_receipt_ref(
            row["route_receipt_ref"],
            context=f"rows[{expected_index}].route_receipt_ref",
        )
        if row["identity_sha256"] in identities:
            raise ValueError("route receipt roster repeats a typed patient identity")
        identities.add(row["identity_sha256"])
        artifact_id = row["route_receipt_ref"]["artifact_id"]
        if artifact_id in receipt_ids:
            raise ValueError("route receipt roster repeats a route receipt ArtifactRef")
        receipt_ids.add(artifact_id)
        rows.append(row)
    data["rows"] = rows

    if data["index_contract"] != _INDEX_CONTRACT:
        raise ValueError("route receipt roster index contract drifted")
    if data["replay_contract"] != _REPLAY_CONTRACT:
        raise ValueError("route receipt roster replay contract drifted")

    if route_receipts is not None:
        receipts = _validated_receipt_payloads(route_receipts)
        if len(receipts) != patient_count:
            raise ValueError("route receipt replay count disagrees with roster")
        for patient_index, (row, receipt) in enumerate(zip(rows, receipts)):
            decision = receipt["decision"]
            if (
                receipt["method_id"] != METHOD_ID
                or decision["route"] != route
                or decision["unit_kind"] != unit_kind
            ):
                raise ValueError(
                    f"route receipt rows[{patient_index}] route/method does not replay"
                )
            if (
                decision["identity_namespace"] != row["identity_namespace"]
                or decision["identity_sha256"] != row["identity_sha256"]
            ):
                raise ValueError(
                    f"route receipt rows[{patient_index}] typed identity does not replay"
                )
            if receipt["resource_config_ref"] != resource_config_ref:
                raise ValueError(
                    f"route receipt rows[{patient_index}] resource config does not replay"
                )
            if (
                receipt["resource_registry_projection_sha256"]
                != registry_projection_sha256
            ):
                raise ValueError(
                    f"route receipt rows[{patient_index}] resource registry does not replay"
                )
            verify_artifact_content(row["route_receipt_ref"], receipt)

    if resource_config_bytes is not None:
        if not isinstance(resource_config_bytes, bytes):
            raise TypeError("resource_config_bytes must be immutable bytes")
        verify_artifact_content(resource_config_ref, resource_config_bytes)

    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["roster_id"] != expected_id:
        raise ValueError("v29 route receipt roster_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("v29 route receipt roster receipt_sha256 does not bind its content")
    return data


__all__ = [
    "V29_ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION",
    "ROUTE_RECEIPT_ROSTER_ARTIFACT_KIND",
    "build_v29_route_receipt_roster",
    "validate_v29_route_receipt_roster",
]
