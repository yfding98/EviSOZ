"""Content-addressed route receipt for the frozen canonical v29 reference.

The route resolver in :mod:`src.evisoz.baseline.frozen_v29` intentionally
returns an in-memory decision.  This module turns that already-resolved
decision into an immutable artifact while replaying its public-roster and
frozen-resource identities.  It never invokes a model and never grants a
training permission.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any, Mapping

from src.evisoz.data.artifact_ref import (
    build_raw_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from src.evisoz.baseline.frozen_v29 import (
    CALLER_ABSENCE_PROOF_KIND,
    DEVELOPMENT_ONLY,
    EVALUATOR_ONLY,
    FROZEN_MEMBER_PROOF_KIND,
    FROZEN_MEMBER_RELATION,
    FROZEN_FIVE_FOLD_ROUTE,
    FROZEN_PUBLIC_IDENTITY_NAMESPACE,
    HISTORICAL_PUBLIC_ROUTE,
    INFERENCE_ONLY,
    METHOD_ID,
    N_FOLDS,
    P0_LINKAGE_PROOF_ARTIFACT_KIND,
    P0_LINKAGE_PROOF_SCHEMA_VERSION,
    PROVEN_ABSENT_RELATION,
    PUBLIC_PATIENT_COUNT,
    RESOURCE_CONFIG_SCHEMA_VERSION,
    FrozenV29ResourceRegistry,
    PublicV29RosterIndex,
    V29RouteDecision,
    validate_frozen_v29_resource_registry,
    validate_public_v29_roster_index,
    validate_v29_route_decision,
)


V29_ROUTE_RECEIPT_SCHEMA_VERSION = "evisoz_v29_route_receipt_v1"
ROUTE_RECEIPT_ARTIFACT_KIND = "v29_route_receipt"
RESOURCE_CONFIG_ARTIFACT_KIND = "v29_frozen_resource_config"

_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-V29-ROUTE-"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_TOP_KEYS = {
    "schema_version",
    "receipt_id",
    "method_id",
    "patient_id_sha256",
    "decision",
    "resource_config_ref",
    "resource_registry_projection_sha256",
    "public_roster_binding",
    "safety_contract",
    "receipt_sha256",
}
_DECISION_KEYS = {
    "identity_namespace",
    "identity_sha256",
    "public_roster_relation",
    "public_roster_relation_proof_kind",
    "public_roster_relation_proof_sha256",
    "public_roster_relation_proof_ref",
    "public_roster_relation_sha256",
    "route",
    "unit_kind",
    "fold_indices",
    "public_patient_index",
    "historical_source_role",
    "access_role",
    "historical_development_eligible",
    "route_layer_training_authorized",
}
_ROSTER_KEYS = {
    "patient_count",
    "patient_order_sha256",
    "roster_projection_sha256",
    "membership_state",
    "member_patient_index",
    "member_held_out_fold",
}


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_id"] = _PENDING_ID
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _patient_sha256(identity_namespace: str, patient_id: str) -> str:
    return hashlib.sha256(
        b"evisoz-v29-route-patient-v2\x00"
        + identity_namespace.encode("utf-8")
        + b"\x00"
        + patient_id.encode("utf-8")
    ).hexdigest()


def _roster_projection(public_index: PublicV29RosterIndex) -> dict[str, object]:
    rows = []
    for patient_id in public_index.patient_ids:
        binding = public_index.require(patient_id)
        rows.append(
            {
                "patient_id": patient_id,
                "patient_index": binding.patient_index,
                "held_out_fold": binding.held_out_fold,
                "source_role": binding.source_role,
                "access_role": binding.access_role,
            }
        )
    return {
        "patient_ids": list(public_index.patient_ids),
        "rows": rows,
        "fold_counts": {
            str(key): int(value) for key, value in public_index.fold_counts.items()
        },
        "source_role_counts": dict(public_index.source_role_counts),
        "resource_config_sha256": public_index.resource_config_sha256,
        "resource_registry_projection_sha256": (
            public_index.resource_registry_projection_sha256
        ),
        "authority_sha256": public_index.authority_sha256,
    }


def _resource_config_ref(registry: FrozenV29ResourceRegistry) -> dict[str, Any]:
    payload = registry.config_bytes
    if hashlib.sha256(payload).hexdigest() != registry.config_sha256:
        raise ValueError("frozen resource config bytes drifted from loaded registry")
    return build_raw_artifact_ref(
        payload,
        artifact_kind=RESOURCE_CONFIG_ARTIFACT_KIND,
        media_type="application/json",
        payload_schema_version=RESOURCE_CONFIG_SCHEMA_VERSION,
    )


def _trusted_fields(
    *,
    decision: V29RouteDecision,
    public_index: PublicV29RosterIndex,
    resource_registry: FrozenV29ResourceRegistry,
) -> dict[str, object]:
    if not isinstance(decision, V29RouteDecision):
        raise TypeError("decision must be a resolved V29RouteDecision")
    if not isinstance(public_index, PublicV29RosterIndex):
        raise TypeError("public_index must be a PublicV29RosterIndex")
    if not isinstance(resource_registry, FrozenV29ResourceRegistry):
        raise TypeError("resource_registry must be a FrozenV29ResourceRegistry")
    if decision.route_layer_training_authorized is not False:
        raise ValueError("the frozen v29 route layer must never authorize training")
    trusted_registry = validate_frozen_v29_resource_registry(resource_registry)
    trusted_index = validate_public_v29_roster_index(
        public_index,
        trusted_registry,
    )
    trusted_decision = validate_v29_route_decision(
        decision,
        trusted_index,
        trusted_registry,
    )
    if trusted_decision.route_layer_training_authorized is not False:
        raise ValueError("the frozen v29 route layer must never authorize training")

    if trusted_decision.route == HISTORICAL_PUBLIC_ROUTE:
        member = trusted_index.require(trusted_decision.patient_id)
        membership_state = "present"
        member_index: int | None = member.patient_index
        member_fold: int | None = member.held_out_fold
    elif trusted_decision.route == FROZEN_FIVE_FOLD_ROUTE:
        # Raw patient strings are deliberately irrelevant here.  A private
        # identity may share a string with a frozen-public patient and must
        # still remain on the proven-absent, five-fold event route.
        membership_state = "absent"
        member_index = None
        member_fold = None
    else:  # pragma: no cover - validate_v29_route_decision closes this
        raise ValueError("decision contains an unsupported frozen v29 route")

    roster_projection = _roster_projection(trusted_index)
    decision_payload = {
        "identity_namespace": trusted_decision.identity_namespace,
        "identity_sha256": trusted_decision.identity_sha256,
        "public_roster_relation": trusted_decision.public_roster_relation,
        "public_roster_relation_proof_kind": (
            trusted_decision.public_roster_relation_proof_kind
        ),
        "public_roster_relation_proof_sha256": (
            trusted_decision.public_roster_relation_proof_sha256
        ),
        "public_roster_relation_proof_ref": (
            deepcopy(dict(trusted_decision.public_roster_relation_proof_ref))
            if trusted_decision.public_roster_relation_proof_ref is not None
            else None
        ),
        "public_roster_relation_sha256": (
            trusted_decision.public_roster_relation_sha256
        ),
        "route": trusted_decision.route,
        "unit_kind": trusted_decision.unit_kind,
        "fold_indices": list(trusted_decision.fold_indices),
        "public_patient_index": trusted_decision.public_patient_index,
        "historical_source_role": trusted_decision.historical_source_role,
        "access_role": trusted_decision.access_role,
        "historical_development_eligible": (
            trusted_decision.historical_development_eligible
        ),
        "route_layer_training_authorized": False,
    }
    return {
        "method_id": METHOD_ID,
        "patient_id_sha256": _patient_sha256(
            trusted_decision.identity_namespace,
            trusted_decision.patient_id,
        ),
        "decision": decision_payload,
        "resource_config_ref": _resource_config_ref(trusted_registry),
        "resource_registry_projection_sha256": (
            trusted_index.resource_registry_projection_sha256
        ),
        "public_roster_binding": {
            "patient_count": len(trusted_index.patient_ids),
            "patient_order_sha256": canonical_json_sha256(
                list(trusted_index.patient_ids)
            ),
            "roster_projection_sha256": canonical_json_sha256(roster_projection),
            "membership_state": membership_state,
            "member_patient_index": member_index,
            "member_held_out_fold": member_fold,
        },
        "safety_contract": {
            "route_layer_is_training_authority": False,
            "historical_source_role_can_enable_loss": False,
            "training_requires_independent_split_and_field_release": True,
            "model_execution_performed": False,
            "typed_patient_identity_required": True,
            "unknown_public_roster_relation_fails_closed": True,
            "non_public_route_requires_proven_absence": True,
            "frozen_public_route_requires_authoritative_membership": True,
        },
    }


def build_v29_route_receipt(
    *,
    decision: V29RouteDecision,
    public_index: PublicV29RosterIndex,
    resource_registry: FrozenV29ResourceRegistry,
) -> dict[str, Any]:
    """Seal a previously resolved route against its frozen authorities."""

    trusted = _trusted_fields(
        decision=decision,
        public_index=public_index,
        resource_registry=resource_registry,
    )
    body: dict[str, Any] = {
        "schema_version": V29_ROUTE_RECEIPT_SCHEMA_VERSION,
        "receipt_id": _PENDING_ID,
        **trusted,
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["receipt_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    # Trusted inputs were replayed by _trusted_fields; the final pass closes
    # the self-contained receipt hashes without reopening any source path.
    return validate_v29_route_receipt(body)


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identity_namespace(value: object) -> str:
    if not isinstance(value, str) or _NAMESPACE_RE.fullmatch(value) is None:
        raise ValueError("decision.identity_namespace must be a stable namespace")
    return value


def _relation_sha256(
    *,
    identity_sha256: str,
    state: str,
    proof_kind: str,
    proof_sha256: str,
    proof_ref: Mapping[str, object] | None,
) -> str:
    return canonical_json_sha256(
        {
            "domain": "evisoz_v29_public_roster_relation_v1",
            "identity_sha256": identity_sha256,
            "state": state,
            "proof_kind": proof_kind,
            "proof_sha256": proof_sha256,
            "proof_ref": deepcopy(dict(proof_ref)) if proof_ref is not None else None,
        }
    )


def validate_v29_route_receipt(
    value: object,
    *,
    decision: V29RouteDecision | None = None,
    public_index: PublicV29RosterIndex | None = None,
    resource_registry: FrozenV29ResourceRegistry | None = None,
) -> dict[str, Any]:
    """Validate a closed receipt and optionally replay all trusted inputs."""

    if type(value) is not dict or set(value) != _TOP_KEYS:
        raise ValueError("v29 route receipt fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != V29_ROUTE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("v29 route receipt schema_version drifted")
    if data["method_id"] != METHOD_ID:
        raise ValueError("v29 route receipt method_id drifted")
    _sha256(data["patient_id_sha256"], "patient_id_sha256")
    _sha256(
        data["resource_registry_projection_sha256"],
        "resource_registry_projection_sha256",
    )
    resource_ref = validate_artifact_ref(data["resource_config_ref"])
    if (
        resource_ref["artifact_kind"] != RESOURCE_CONFIG_ARTIFACT_KIND
        or resource_ref["media_type"] != "application/json"
        or resource_ref["content_hash"]["domain"] != "raw_bytes_v1"
        or resource_ref["payload_schema_version"] != RESOURCE_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError("route receipt resource config ArtifactRef drifted")
    data["resource_config_ref"] = resource_ref

    decision_row = data["decision"]
    if type(decision_row) is not dict or set(decision_row) != _DECISION_KEYS:
        raise ValueError("v29 route decision fields drifted")
    if decision_row["route_layer_training_authorized"] is not False:
        raise ValueError("v29 route receipt cannot authorize training")
    namespace = _identity_namespace(decision_row["identity_namespace"])
    identity_sha256 = _sha256(
        decision_row["identity_sha256"],
        "decision.identity_sha256",
    )
    relation = decision_row["public_roster_relation"]
    proof_kind = decision_row["public_roster_relation_proof_kind"]
    if not isinstance(relation, str) or not isinstance(proof_kind, str):
        raise TypeError("route receipt relation and proof kind must be strings")
    proof_sha256 = _sha256(
        decision_row["public_roster_relation_proof_sha256"],
        "decision.public_roster_relation_proof_sha256",
    )
    proof_ref_value = decision_row["public_roster_relation_proof_ref"]
    if proof_ref_value is None:
        proof_ref = None
    else:
        proof_ref = validate_artifact_ref(proof_ref_value)
        if (
            proof_ref["artifact_kind"] != P0_LINKAGE_PROOF_ARTIFACT_KIND
            or proof_ref["media_type"] != "application/json"
            or proof_ref["content_hash"]["domain"] != "canonical_json_v1"
            or proof_ref["payload_schema_version"]
            != P0_LINKAGE_PROOF_SCHEMA_VERSION
        ):
            raise ValueError("route receipt P0 linkage proof ArtifactRef drifted")
        decision_row["public_roster_relation_proof_ref"] = proof_ref
    relation_sha256 = _sha256(
        decision_row["public_roster_relation_sha256"],
        "decision.public_roster_relation_sha256",
    )
    if relation_sha256 != _relation_sha256(
        identity_sha256=identity_sha256,
        state=relation,
        proof_kind=proof_kind,
        proof_sha256=proof_sha256,
        proof_ref=proof_ref,
    ):
        raise ValueError("route receipt public roster relation digest drifted")
    route = decision_row["route"]
    folds = decision_row["fold_indices"]
    if not isinstance(folds, list) or any(type(value) is not int for value in folds):
        raise TypeError("route fold_indices must be an integer array")
    if route == HISTORICAL_PUBLIC_ROUTE:
        if (
            namespace != FROZEN_PUBLIC_IDENTITY_NAMESPACE
            or relation != FROZEN_MEMBER_RELATION
            or proof_kind != FROZEN_MEMBER_PROOF_KIND
            or proof_ref is not None
            or decision_row["unit_kind"] != "patient"
            or len(folds) != 1
            or folds[0] not in range(N_FOLDS)
            or isinstance(decision_row["public_patient_index"], bool)
            or not isinstance(decision_row["public_patient_index"], int)
            or decision_row["public_patient_index"] < 0
            or decision_row["historical_source_role"]
            not in {"source_train", "source_dev", "source_eval"}
            or decision_row["access_role"]
            not in {DEVELOPMENT_ONLY, EVALUATOR_ONLY}
        ):
            raise ValueError("public held-fold route decision is invalid")
        expected_access = (
            EVALUATOR_ONLY
            if decision_row["historical_source_role"] == "source_eval"
            else DEVELOPMENT_ONLY
        )
        if decision_row["access_role"] != expected_access:
            raise ValueError("public source role/access relation drifted")
        expected_eligible = expected_access == DEVELOPMENT_ONLY
    elif route == FROZEN_FIVE_FOLD_ROUTE:
        if (
            namespace == FROZEN_PUBLIC_IDENTITY_NAMESPACE
            or relation != PROVEN_ABSENT_RELATION
            or proof_kind != CALLER_ABSENCE_PROOF_KIND
            or proof_ref is None
            or proof_sha256 != proof_ref["ref_sha256"]
            or decision_row["unit_kind"] != "event"
            or folds != list(range(N_FOLDS))
            or decision_row["public_patient_index"] is not None
            or decision_row["historical_source_role"]
            not in {None, "source_train", "source_dev", "source_eval"}
            or decision_row["access_role"]
            not in {DEVELOPMENT_ONLY, EVALUATOR_ONLY, INFERENCE_ONLY}
        ):
            raise ValueError("five-fold new-event route decision is invalid")
        if decision_row["historical_source_role"] == "source_eval":
            expected_access = EVALUATOR_ONLY
        elif decision_row["historical_source_role"] in {
            "source_train",
            "source_dev",
        }:
            expected_access = DEVELOPMENT_ONLY
        else:
            expected_access = INFERENCE_ONLY
        if decision_row["access_role"] != expected_access:
            raise ValueError("new-event source role/access relation drifted")
        expected_eligible = expected_access == DEVELOPMENT_ONLY
    else:
        raise ValueError("v29 route receipt route is unsupported")
    if decision_row["historical_development_eligible"] is not expected_eligible:
        raise ValueError("historical development eligibility drifted")

    roster = data["public_roster_binding"]
    if type(roster) is not dict or set(roster) != _ROSTER_KEYS:
        raise ValueError("public roster binding fields drifted")
    if roster["patient_count"] != PUBLIC_PATIENT_COUNT:
        raise ValueError("public roster patient count drifted")
    _sha256(roster["patient_order_sha256"], "patient_order_sha256")
    _sha256(roster["roster_projection_sha256"], "roster_projection_sha256")
    if route == HISTORICAL_PUBLIC_ROUTE:
        if (
            roster["membership_state"] != "present"
            or roster["member_patient_index"]
            != decision_row["public_patient_index"]
            or roster["member_patient_index"] not in range(PUBLIC_PATIENT_COUNT)
            or roster["member_held_out_fold"] != folds[0]
        ):
            raise ValueError("public route roster membership binding drifted")
    elif (
        roster["membership_state"] != "absent"
        or roster["member_patient_index"] is not None
        or roster["member_held_out_fold"] is not None
    ):
        raise ValueError("new-event route must bind absence from the public roster")

    if data["safety_contract"] != {
        "route_layer_is_training_authority": False,
        "historical_source_role_can_enable_loss": False,
        "training_requires_independent_split_and_field_release": True,
        "model_execution_performed": False,
        "typed_patient_identity_required": True,
        "unknown_public_roster_relation_fails_closed": True,
        "non_public_route_requires_proven_absence": True,
        "frozen_public_route_requires_authoritative_membership": True,
    }:
        raise ValueError("v29 route safety contract drifted")

    supplied = (decision, public_index, resource_registry)
    if any(item is not None for item in supplied):
        if any(item is None for item in supplied):
            raise ValueError("trusted route replay requires decision, roster and registry")
        expected = _trusted_fields(
            decision=decision,
            public_index=public_index,
            resource_registry=resource_registry,
        )
        for field, expected_value in expected.items():
            if data[field] != expected_value:
                raise ValueError(f"v29 route receipt {field} does not replay")
        verify_artifact_content(
            resource_ref,
            resource_registry.config_bytes,
        )

    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["receipt_id"] != expected_id:
        raise ValueError("v29 route receipt_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("v29 route receipt_sha256 does not bind its content")
    return data


__all__ = [
    "V29_ROUTE_RECEIPT_SCHEMA_VERSION",
    "ROUTE_RECEIPT_ARTIFACT_KIND",
    "RESOURCE_CONFIG_ARTIFACT_KIND",
    "build_v29_route_receipt",
    "validate_v29_route_receipt",
]
