"""Atomic wrapper-to-cache materialization and replay for frozen v29.

The low-level cache contract can validate numerical tensors, but it does not
by itself prove which typed patients own the rows or which per-patient route
receipts authorized their frozen inference route.  This module closes that
join.  Production callers materialize through one of the two route-specific
builders and must reopen through :func:`open_v29_cache_for_use`; no p0/z0
tensor is returned before registry, roster, route, identity, montage and
tensor-content replay have all succeeded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from numbers import Integral
import re
from types import MappingProxyType
from typing import Any

import torch

from src.evisoz.baseline.frozen_v29 import (
    FROZEN_FIVE_FOLD_ROUTE,
    HISTORICAL_PUBLIC_ROUTE,
    FrozenV29ResourceRegistry,
    PublicV29RosterIndex,
    V29PatientIdentity,
    V29PublicRosterRelation,
    V29RouteDecision,
    build_frozen_public_member_relation,
    load_v29_inference_states,
    replay_public_oof_rows,
    resolve_v29_route,
    v29_patient_identity_sha256,
    validate_frozen_v29_resource_registry,
    validate_public_v29_roster_index,
)
from src.evisoz.baseline.v29_cache import (
    C18_TO_STANDARD19,
    EVENT_MEAN_ROUTE,
    PREDICTION_RESOURCE_IDS,
    PUBLIC_OOF_ROUTE,
    build_v29_reference_cache,
    canonical_tensor_bytes,
    hard_bypass_p0,
    tensor_sha256,
    validate_v29_reference_cache,
)
from src.evisoz.baseline.v29_route_receipt import (
    build_v29_route_receipt,
    validate_v29_route_receipt,
)
from src.evisoz.baseline.v29_route_receipt_roster import (
    ROUTE_RECEIPT_ROSTER_ARTIFACT_KIND,
    V29_ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION,
    build_v29_route_receipt_roster,
    validate_v29_route_receipt_roster,
)
from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    verify_artifact_content,
)
from src.soz.v29_long_recording_inference import infer_v29_probabilities


PATIENT_IDENTITY_PAYLOAD_SCHEMA_VERSION = (
    "evisoz_v29_patient_identity_roster_projection_v1"
)
EVENT_IDENTITY_PAYLOAD_SCHEMA_VERSION = (
    "evisoz_v29_event_identity_roster_projection_v1"
)
MONTAGE_DERIVATION_PAYLOAD_SCHEMA_VERSION = (
    "evisoz_v29_montage_derivation_roster_projection_v1"
)
MATERIALIZER_VERSION = "evisoz_v29_wrapper_cache_adapter_v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


@dataclass(frozen=True)
class V29MaterializedCache:
    cache: Mapping[str, Any]
    tensor_payloads: Mapping[str, torch.Tensor]
    route_receipt_roster: Mapping[str, Any]
    route_receipts: tuple[Mapping[str, Any], ...]
    identity_payload: Mapping[str, Any]
    event_identity_payload: Mapping[str, Any] | None
    montage_derivation_payload: Mapping[str, Any] | None


@dataclass(frozen=True)
class _CanonicalTensorSnapshot:
    """Immutable canonical bytes for one replayed tensor."""

    payload: bytes = field(repr=False)
    dtype: torch.dtype
    shape: tuple[int, ...]
    sha256: str

    @classmethod
    def from_tensor(cls, value: torch.Tensor) -> _CanonicalTensorSnapshot:
        tensor = value.detach().cpu().contiguous()
        return cls(
            payload=canonical_tensor_bytes(tensor),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            sha256=tensor_sha256(tensor),
        )

    def checkout(self) -> torch.Tensor:
        """Decode a fresh tensor whose mutation cannot alter this snapshot."""

        header_length = int.from_bytes(self.payload[:8], byteorder="big")
        raw_length_offset = 8 + header_length
        raw_length = int.from_bytes(
            self.payload[raw_length_offset : raw_length_offset + 8],
            byteorder="big",
        )
        payload_start = raw_length_offset + 8
        payload_end = payload_start + raw_length
        if payload_end != len(self.payload):  # pragma: no cover - built internally
            raise RuntimeError("canonical tensor snapshot envelope drifted")
        tensor = torch.frombuffer(
            bytearray(self.payload[payload_start:payload_end]),
            dtype=self.dtype,
        ).clone()
        tensor = tensor.reshape(self.shape).contiguous()
        if (
            canonical_tensor_bytes(tensor) != self.payload
            or tensor_sha256(tensor) != self.sha256
        ):  # pragma: no cover - immutable internal bytes
            raise RuntimeError("canonical tensor snapshot failed checkout replay")
        return tensor


@dataclass(frozen=True)
class OpenedV29Cache:
    """Verified cache handle with clone-only tensor checkout.

    No mutable replayed tensor is stored in or exposed by this object.  The
    private snapshots contain immutable canonical bytes; every checkout
    decodes a new tensor.  Consequently mutating a checkout cannot change a
    later checkout or its recorded digest.
    """

    cache: Mapping[str, Any]
    route_decisions: tuple[V29RouteDecision, ...]
    unit_access_roles: tuple[str, ...]
    _tensor_snapshots: Mapping[str, _CanonicalTensorSnapshot] = field(
        repr=False,
    )
    _alpha_zero_hard_bypass: bool = field(repr=False)
    _selected_residual: Any = field(repr=False, compare=False)
    _selected_tensor_snapshot: _CanonicalTensorSnapshot | None = field(
        repr=False,
        compare=False,
    )

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self._tensor_snapshots)

    @property
    def alpha_zero_hard_bypass(self) -> bool:
        return self._alpha_zero_hard_bypass

    def tensor_sha256(self, name: str) -> str:
        try:
            return self._tensor_snapshots[name].sha256
        except KeyError as exc:
            raise KeyError(f"unknown opened v29 tensor: {name!r}") from exc

    def checkout_tensor(self, name: str) -> torch.Tensor:
        try:
            snapshot = self._tensor_snapshots[name]
        except KeyError as exc:
            raise KeyError(f"unknown opened v29 tensor: {name!r}") from exc
        return snapshot.checkout()

    def checkout_p0(self) -> torch.Tensor:
        return self.checkout_tensor("p0_c18")

    def checkout_z0(self) -> torch.Tensor:
        return self.checkout_tensor("z0_c18")

    def checkout_selected(self) -> Any:
        if self._alpha_zero_hard_bypass:
            return self.checkout_p0()
        if self._selected_tensor_snapshot is not None:
            return self._selected_tensor_snapshot.checkout()
        return self._selected_residual


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a stable identifier")
    return value


def _trusted_authorities(
    registry: FrozenV29ResourceRegistry,
    public_index: PublicV29RosterIndex,
) -> tuple[FrozenV29ResourceRegistry, PublicV29RosterIndex]:
    trusted_registry = validate_frozen_v29_resource_registry(registry)
    trusted_index = validate_public_v29_roster_index(
        public_index,
        trusted_registry,
    )
    return trusted_registry, trusted_index


def _validate_patient_identities(
    identities: object,
) -> tuple[V29PatientIdentity, ...]:
    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise TypeError("patient_identities must be an ordered sequence")
    rows = tuple(identities)
    if not rows or any(not isinstance(value, V29PatientIdentity) for value in rows):
        raise TypeError("patient_identities must contain V29PatientIdentity values")
    digests = tuple(v29_patient_identity_sha256(value) for value in rows)
    if len(set(digests)) != len(digests):
        raise ValueError("patient_identities must be unique typed identities")
    return rows


def _patient_rows(
    identities: Sequence[V29PatientIdentity],
) -> list[dict[str, object]]:
    return [
        {
            "patient_index": patient_index,
            "identity_namespace": identity.namespace,
            "patient_id": identity.patient_id,
            "identity_sha256": v29_patient_identity_sha256(identity),
        }
        for patient_index, identity in enumerate(identities)
    ]


def _public_identity_payload(
    identities: Sequence[V29PatientIdentity],
) -> dict[str, object]:
    return {
        "schema_version": PATIENT_IDENTITY_PAYLOAD_SCHEMA_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "route": PUBLIC_OOF_ROUTE,
        "patients": _patient_rows(identities),
    }


def _event_identity_payload(
    identities: Sequence[V29PatientIdentity],
    event_ids: Sequence[str],
    ownership: torch.Tensor,
) -> dict[str, object]:
    patient_rows = _patient_rows(identities)
    events = []
    for unit_index, (event_id, patient_index) in enumerate(
        zip(event_ids, ownership.tolist())
    ):
        events.append(
            {
                "unit_index": unit_index,
                "event_id": _identifier(event_id, f"event_ids[{unit_index}]"),
                "patient_index": int(patient_index),
                "identity_sha256": patient_rows[int(patient_index)][
                    "identity_sha256"
                ],
            }
        )
    return {
        "schema_version": EVENT_IDENTITY_PAYLOAD_SCHEMA_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "route": EVENT_MEAN_ROUTE,
        "patients": patient_rows,
        "events": events,
    }


def _parse_identity_payload(
    payload: object,
    *,
    route: str,
) -> tuple[tuple[V29PatientIdentity, ...], torch.Tensor]:
    if type(payload) is not dict:
        raise TypeError("identity_payload must be an object")
    expected_keys = {
        "schema_version",
        "materializer_version",
        "route",
        "patients",
    }
    if route == EVENT_MEAN_ROUTE:
        expected_keys.add("events")
    if set(payload) != expected_keys:
        raise ValueError("identity_payload fields drifted")
    expected_schema = (
        PATIENT_IDENTITY_PAYLOAD_SCHEMA_VERSION
        if route == PUBLIC_OOF_ROUTE
        else EVENT_IDENTITY_PAYLOAD_SCHEMA_VERSION
    )
    if (
        payload["schema_version"] != expected_schema
        or payload["materializer_version"] != MATERIALIZER_VERSION
        or payload["route"] != route
    ):
        raise ValueError("identity_payload route/schema drifted")
    patient_values = payload["patients"]
    if not isinstance(patient_values, list) or not patient_values:
        raise ValueError("identity_payload must contain patients")
    identities: list[V29PatientIdentity] = []
    identity_hashes: list[str] = []
    for patient_index, row in enumerate(patient_values):
        if type(row) is not dict or set(row) != {
            "patient_index",
            "identity_namespace",
            "patient_id",
            "identity_sha256",
        }:
            raise ValueError("identity_payload patient fields drifted")
        if row["patient_index"] != patient_index or isinstance(
            row["patient_index"], bool
        ):
            raise ValueError("identity_payload patient indices must be continuous")
        identity = V29PatientIdentity(
            namespace=row["identity_namespace"],
            patient_id=row["patient_id"],
        )
        digest = v29_patient_identity_sha256(identity)
        if row["identity_sha256"] != digest:
            raise ValueError("identity_payload typed identity digest drifted")
        identities.append(identity)
        identity_hashes.append(digest)
    if len(set(identity_hashes)) != len(identity_hashes):
        raise ValueError("identity_payload repeats a typed patient")

    if route == PUBLIC_OOF_ROUTE:
        ownership = torch.arange(len(identities), dtype=torch.int64)
    else:
        event_values = payload["events"]
        if not isinstance(event_values, list) or not event_values:
            raise ValueError("event identity payload must contain events")
        ownership_values: list[int] = []
        event_ids: set[str] = set()
        for unit_index, row in enumerate(event_values):
            if type(row) is not dict or set(row) != {
                "unit_index",
                "event_id",
                "patient_index",
                "identity_sha256",
            }:
                raise ValueError("event identity payload row fields drifted")
            if row["unit_index"] != unit_index or isinstance(row["unit_index"], bool):
                raise ValueError("event unit indices must be continuous")
            event_id = _identifier(row["event_id"], "event identity event_id")
            if event_id in event_ids:
                raise ValueError("event identity payload repeats an event")
            event_ids.add(event_id)
            patient_index = row["patient_index"]
            if (
                isinstance(patient_index, bool)
                or not isinstance(patient_index, int)
                or patient_index not in range(len(identities))
            ):
                raise ValueError("event identity patient_index is out of range")
            if row["identity_sha256"] != identity_hashes[patient_index]:
                raise ValueError("event ownership typed identity digest drifted")
            ownership_values.append(patient_index)
        ownership = torch.tensor(ownership_values, dtype=torch.int64)
        if not torch.equal(
            torch.unique(ownership, sorted=True),
            torch.arange(len(identities), dtype=torch.int64),
        ):
            raise ValueError("every patient must own at least one event")
    return tuple(identities), ownership


def _identity_ref(payload: Mapping[str, object], *, event: bool) -> dict[str, Any]:
    return build_json_artifact_ref(
        payload,
        artifact_kind="event_unit_roster" if event else "patient_unit_roster",
        payload_schema_version=(
            EVENT_IDENTITY_PAYLOAD_SCHEMA_VERSION
            if event
            else PATIENT_IDENTITY_PAYLOAD_SCHEMA_VERSION
        ),
    )


def _event_identity_ref(payload: Mapping[str, object]) -> dict[str, Any]:
    return build_json_artifact_ref(
        payload,
        artifact_kind="event_identity_roster",
        payload_schema_version=EVENT_IDENTITY_PAYLOAD_SCHEMA_VERSION,
    )


def _montage_ref(payload: Mapping[str, object]) -> dict[str, Any]:
    return build_json_artifact_ref(
        payload,
        artifact_kind="montage_derivation_roster",
        payload_schema_version=MONTAGE_DERIVATION_PAYLOAD_SCHEMA_VERSION,
    )


def _route_roster_ref(payload: Mapping[str, object]) -> dict[str, Any]:
    return build_json_artifact_ref(
        payload,
        artifact_kind=ROUTE_RECEIPT_ROSTER_ARTIFACT_KIND,
        payload_schema_version=V29_ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION,
    )


def _freeze_tensor_mapping(
    tensors: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    return MappingProxyType(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in tensors.items()
        }
    )


def _snapshot_tensor_mapping(
    tensors: Mapping[str, torch.Tensor],
) -> Mapping[str, _CanonicalTensorSnapshot]:
    return MappingProxyType(
        {
            name: _CanonicalTensorSnapshot.from_tensor(value)
            for name, value in tensors.items()
        }
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_freeze_decision(decision: V29RouteDecision) -> V29RouteDecision:
    proof_ref = decision.public_roster_relation_proof_ref
    return replace(
        decision,
        public_roster_relation_proof_ref=(
            _deep_freeze(proof_ref) if proof_ref is not None else None
        ),
    )


def _ownership_tensor(value: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Return an exact int64 ownership vector without lossy coercion."""

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError("unit_to_patient_index must be a dense strided tensor")
        if value.dtype != torch.int64:
            raise TypeError("unit_to_patient_index tensor must have dtype int64")
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("unit_to_patient_index must be an ordered sequence")
    rows = list(value)
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in rows):
        raise TypeError("unit_to_patient_index values must be exact integers")
    return torch.tensor([int(item) for item in rows], dtype=torch.int64)


def _bundle(
    *,
    cache: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    route_roster: Mapping[str, Any],
    route_receipts: Sequence[Mapping[str, Any]],
    identity_payload: Mapping[str, Any],
    event_identity_payload: Mapping[str, Any] | None,
    montage_payload: Mapping[str, Any] | None,
) -> V29MaterializedCache:
    return V29MaterializedCache(
        cache=MappingProxyType(deepcopy(dict(cache))),
        tensor_payloads=_freeze_tensor_mapping(tensors),
        route_receipt_roster=MappingProxyType(deepcopy(dict(route_roster))),
        route_receipts=tuple(
            MappingProxyType(deepcopy(dict(receipt))) for receipt in route_receipts
        ),
        identity_payload=MappingProxyType(deepcopy(dict(identity_payload))),
        event_identity_payload=(
            MappingProxyType(deepcopy(dict(event_identity_payload)))
            if event_identity_payload is not None
            else None
        ),
        montage_derivation_payload=(
            MappingProxyType(deepcopy(dict(montage_payload)))
            if montage_payload is not None
            else None
        ),
    )


def materialize_public_v29_cache(
    patient_identities: Sequence[V29PatientIdentity],
    *,
    registry: FrozenV29ResourceRegistry,
    public_index: PublicV29RosterIndex,
) -> V29MaterializedCache:
    """Atomically materialize held-fold public rows for ordered patients."""

    trusted_registry, trusted_index = _trusted_authorities(registry, public_index)
    identities = _validate_patient_identities(patient_identities)
    relations = tuple(
        build_frozen_public_member_relation(identity, trusted_index, trusted_registry)
        for identity in identities
    )
    decisions = tuple(
        resolve_v29_route(
            identity,
            trusted_index,
            trusted_registry,
            public_roster_relation=relation,
            requested_route=HISTORICAL_PUBLIC_ROUTE,
        )
        for identity, relation in zip(identities, relations)
    )
    receipts = tuple(
        build_v29_route_receipt(
            decision=decision,
            public_index=trusted_index,
            resource_registry=trusted_registry,
        )
        for decision in decisions
    )
    route_roster = build_v29_route_receipt_roster(route_receipts=receipts)
    replay = replay_public_oof_rows(
        trusted_registry,
        trusted_index,
        identities,
        relations,
    )
    indices = torch.tensor(C18_TO_STANDARD19, dtype=torch.long)
    p_h = replay["p_h_node"].index_select(1, indices).contiguous()
    p_d = replay["p_d_node"].index_select(1, indices).contiguous()
    p0 = replay["p0_node"].index_select(1, indices).contiguous()
    route_fold_mask = torch.zeros((len(identities), 5), dtype=torch.bool)
    route_fold_mask[
        torch.arange(len(identities)), replay["held_out_folds"]
    ] = True
    tensors = {
        "p_h_c18": p_h,
        "p_d_c18": p_d,
        "p0_c18": p0,
        "p0_node": replay["p0_node"].contiguous(),
        "z0_c18": torch.log(p0),
        "candidate_mask": replay["candidate_mask_node"].contiguous(),
        "route_fold_mask": route_fold_mask,
        "unit_to_patient_index": torch.arange(len(identities), dtype=torch.int64),
    }
    identity_payload = _public_identity_payload(identities)
    cache = build_v29_reference_cache(
        route=PUBLIC_OOF_ROUTE,
        unit_kind="patient",
        tensor_payloads=tensors,
        identity_ref=_identity_ref(identity_payload, event=False),
        route_receipt_roster_ref=_route_roster_ref(route_roster),
        resource_manifest_ref=route_roster["resource_config_ref"],
        prediction_resource_ids=PREDICTION_RESOURCE_IDS[PUBLIC_OOF_ROUTE],
    )
    return _bundle(
        cache=cache,
        tensors=tensors,
        route_roster=route_roster,
        route_receipts=receipts,
        identity_payload=identity_payload,
        event_identity_payload=None,
        montage_payload=None,
    )


def materialize_event_v29_cache(
    patient_identities: Sequence[V29PatientIdentity],
    public_roster_relations: Sequence[V29PublicRosterRelation],
    event_ids: Sequence[str],
    unit_to_patient_index: Sequence[int] | torch.Tensor,
    h_features: torch.Tensor,
    phase_features: torch.Tensor,
    montage_derivation_payload: Mapping[str, object],
    *,
    registry: FrozenV29ResourceRegistry,
    public_index: PublicV29RosterIndex,
    official_source_roles: Sequence[str | None] | None = None,
) -> V29MaterializedCache:
    """Materialize a quarantined five-fold event bundle for numerical audits.

    The production open gate deliberately rejects this route until the bundle
    carries validated per-event EventIdentity/MontageDerivationReceipt rows and
    content-addressed H/phase input receipts.  Callers must not consume the
    returned prediction tensors as a deployed v29 reference.
    """

    trusted_registry, trusted_index = _trusted_authorities(registry, public_index)
    identities = _validate_patient_identities(patient_identities)
    if isinstance(public_roster_relations, (str, bytes)) or not isinstance(
        public_roster_relations, Sequence
    ):
        raise TypeError("public_roster_relations must be an ordered sequence")
    relations = tuple(public_roster_relations)
    if len(relations) != len(identities) or any(
        not isinstance(value, V29PublicRosterRelation) for value in relations
    ):
        raise ValueError("public_roster_relations must align with patients")
    if official_source_roles is None:
        roles = (None,) * len(identities)
    else:
        if isinstance(official_source_roles, (str, bytes)) or not isinstance(
            official_source_roles, Sequence
        ):
            raise TypeError("official_source_roles must be an ordered sequence")
        roles = tuple(official_source_roles)
        if len(roles) != len(identities):
            raise ValueError("official_source_roles must align with patients")
    decisions = tuple(
        resolve_v29_route(
            identity,
            trusted_index,
            trusted_registry,
            public_roster_relation=relation,
            requested_route=FROZEN_FIVE_FOLD_ROUTE,
            official_source_role=role,
        )
        for identity, relation, role in zip(identities, relations, roles)
    )
    receipts = tuple(
        build_v29_route_receipt(
            decision=decision,
            public_index=trusted_index,
            resource_registry=trusted_registry,
        )
        for decision in decisions
    )
    route_roster = build_v29_route_receipt_roster(route_receipts=receipts)

    ownership = _ownership_tensor(unit_to_patient_index)
    if ownership.ndim != 1 or ownership.numel() < 1:
        raise ValueError("unit_to_patient_index must be a non-empty vector")
    if bool((ownership < 0).any()) or bool((ownership >= len(identities)).any()):
        raise ValueError("unit_to_patient_index contains an out-of-range patient")
    if not torch.equal(
        torch.unique(ownership, sorted=True),
        torch.arange(len(identities), dtype=torch.int64),
    ):
        raise ValueError("every patient must own at least one event")
    if isinstance(event_ids, (str, bytes)) or not isinstance(event_ids, Sequence):
        raise TypeError("event_ids must be an ordered sequence")
    event_id_values = tuple(
        _identifier(value, f"event_ids[{index}]")
        for index, value in enumerate(event_ids)
    )
    if len(event_id_values) != ownership.numel() or len(set(event_id_values)) != len(
        event_id_values
    ):
        raise ValueError("event_ids must be unique and align with ownership")
    if type(montage_derivation_payload) is not dict:
        raise TypeError("montage_derivation_payload must be an object")

    direct_states, h_states = load_v29_inference_states(trusted_registry)
    output = infer_v29_probabilities(
        h_features,
        phase_features,
        direct_states=direct_states,
        h_states=h_states,
    )
    if output["portable_equal_probability"].shape[0] != ownership.numel():
        raise ValueError("event feature count does not align with event identities")
    indices = torch.tensor(C18_TO_STANDARD19, dtype=torch.long)
    p_h_fold = output["h_only_fold_probability"].index_select(2, indices).contiguous()
    p_d_fold = output["rank1_direct_fold_probability"].index_select(2, indices).contiguous()
    p_equal_fold = output["portable_equal_fold_probability"].index_select(
        2, indices
    ).contiguous()
    p_h = p_h_fold.mean(dim=1).contiguous()
    p_d = p_d_fold.mean(dim=1).contiguous()
    p0 = p_equal_fold.mean(dim=1).contiguous()
    tensors = {
        "p_h_c18": p_h,
        "p_d_c18": p_d,
        "p0_c18": p0,
        "p0_node": output["portable_equal_probability"].contiguous(),
        "z0_c18": torch.log(p0),
        "candidate_mask": output["candidate_mask"].contiguous(),
        "route_fold_mask": torch.ones((ownership.numel(), 5), dtype=torch.bool),
        "unit_to_patient_index": ownership,
        "p_h_fold_c18": p_h_fold,
        "p_d_fold_c18": p_d_fold,
        "p_equal_fold_c18": p_equal_fold,
    }
    identity_payload = _event_identity_payload(
        identities,
        event_id_values,
        ownership,
    )
    event_payload = deepcopy(identity_payload)
    montage_payload = deepcopy(montage_derivation_payload)
    cache = build_v29_reference_cache(
        route=EVENT_MEAN_ROUTE,
        unit_kind="event",
        tensor_payloads=tensors,
        identity_ref=_identity_ref(identity_payload, event=True),
        event_identity_ref=_event_identity_ref(event_payload),
        montage_derivation_receipt_ref=_montage_ref(montage_payload),
        route_receipt_roster_ref=_route_roster_ref(route_roster),
        resource_manifest_ref=route_roster["resource_config_ref"],
        prediction_resource_ids=PREDICTION_RESOURCE_IDS[EVENT_MEAN_ROUTE],
    )
    return _bundle(
        cache=cache,
        tensors=tensors,
        route_roster=route_roster,
        route_receipts=receipts,
        identity_payload=identity_payload,
        event_identity_payload=event_payload,
        montage_payload=montage_payload,
    )


def _relation_from_receipt(receipt: Mapping[str, Any]) -> V29PublicRosterRelation:
    decision = receipt["decision"]
    return V29PublicRosterRelation(
        identity_sha256=decision["identity_sha256"],
        state=decision["public_roster_relation"],
        proof_kind=decision["public_roster_relation_proof_kind"],
        proof_sha256=decision["public_roster_relation_proof_sha256"],
        proof_ref=decision["public_roster_relation_proof_ref"],
        relation_sha256=decision["public_roster_relation_sha256"],
    )


def _assert_equal(name: str, observed: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(observed.detach().cpu(), expected.detach().cpu()):
        raise ValueError(f"{name} does not replay from the frozen wrapper")


def open_v29_cache_for_use(
    cache: Mapping[str, Any],
    *,
    tensor_payloads: Mapping[str, torch.Tensor],
    route_receipt_roster: Mapping[str, Any],
    route_receipts: Sequence[Mapping[str, Any]],
    identity_payload: Mapping[str, Any],
    event_identity_payload: Mapping[str, Any] | None,
    montage_derivation_payload: Mapping[str, Any] | None,
    registry: FrozenV29ResourceRegistry,
    public_index: PublicV29RosterIndex,
    alpha: float = 0.0,
    residual_supplier: Callable[[], Any] | None = None,
    h_features: torch.Tensor | None = None,
    phase_features: torch.Tensor | None = None,
) -> OpenedV29Cache:
    """Replay every available authority before exposing public p0 or z0.

    The public held-fold route is production-open after complete replay.  The
    event route is currently fail-closed because its semantic input receipts
    are not yet represented by this adapter contract.
    """

    if not isinstance(cache, Mapping):
        raise TypeError("cache must be a mapping")
    trusted_registry, trusted_index = _trusted_authorities(registry, public_index)
    structural = validate_v29_reference_cache(dict(cache))
    route = structural["route"]

    # The event materializer remains useful for quarantined numerical audits,
    # but its current projection payload does not yet contain validated
    # EventIdentity/MontageDerivationReceipt rows or content-addressed H/phase
    # feature inputs.  Consequently no event p0/z0 may leave this production
    # open gate, including the otherwise hard-bypassed alpha=0 path.
    if route == EVENT_MEAN_ROUTE:
        raise RuntimeError(
            "event v29 cache consumption is disabled until EventIdentity, "
            "MontageDerivationReceipt and H/phase input receipts are replayable"
        )

    resource_ref = structural["bindings"]["resource_manifest_ref"]
    verify_artifact_content(resource_ref, trusted_registry.config_bytes)
    for resource_id in structural["bindings"]["prediction_resource_ids"]:
        trusted_registry.require(resource_id)

    receipt_payloads = tuple(deepcopy(dict(value)) for value in route_receipts)
    roster = validate_v29_route_receipt_roster(
        dict(route_receipt_roster),
        route_receipts=receipt_payloads,
        resource_config_bytes=trusted_registry.config_bytes,
    )
    verify_artifact_content(
        structural["bindings"]["route_receipt_roster_ref"],
        roster,
    )
    if (
        roster["resource_config_ref"] != resource_ref
        or roster["route"] != route
        or roster["patient_count"] != structural["patient_count"]
    ):
        raise ValueError("route receipt roster does not bind the cache authority")

    identities, identity_ownership = _parse_identity_payload(
        dict(identity_payload),
        route=route,
    )
    verify_artifact_content(
        structural["bindings"]["identity_ref"],
        dict(identity_payload),
    )
    if len(identities) != roster["patient_count"]:
        raise ValueError("identity payload patient count disagrees with route roster")
    if len(receipt_payloads) != len(identities):
        raise ValueError("one route receipt is required per typed patient")

    decisions: list[V29RouteDecision] = []
    relations: list[V29PublicRosterRelation] = []
    for patient_index, (identity, row, receipt_value) in enumerate(
        zip(identities, roster["rows"], receipt_payloads)
    ):
        receipt = validate_v29_route_receipt(dict(receipt_value))
        identity_sha = v29_patient_identity_sha256(identity)
        if (
            row["patient_index"] != patient_index
            or row["identity_namespace"] != identity.namespace
            or row["identity_sha256"] != identity_sha
        ):
            raise ValueError("typed patient order does not replay from route roster")
        relation = _relation_from_receipt(receipt)
        source_role = receipt["decision"]["historical_source_role"]
        decision = resolve_v29_route(
            identity,
            trusted_index,
            trusted_registry,
            public_roster_relation=relation,
            requested_route=route,
            official_source_role=source_role,
        )
        validate_v29_route_receipt(
            receipt,
            decision=decision,
            public_index=trusted_index,
            resource_registry=trusted_registry,
        )
        verify_artifact_content(row["route_receipt_ref"], receipt)
        decisions.append(decision)
        relations.append(relation)

    if route == PUBLIC_OOF_ROUTE:
        if event_identity_payload is not None or montage_derivation_payload is not None:
            raise ValueError("public cache cannot replay event or montage payloads")
    else:
        if event_identity_payload is None or montage_derivation_payload is None:
            raise ValueError("event cache requires event identity and montage payloads")
        if dict(event_identity_payload) != dict(identity_payload):
            raise ValueError("event unit and event identity payloads must agree")
        verify_artifact_content(
            structural["bindings"]["event_identity_ref"],
            dict(event_identity_payload),
        )
        verify_artifact_content(
            structural["bindings"]["montage_derivation_receipt_ref"],
            dict(montage_derivation_payload),
        )

    validated_cache = validate_v29_reference_cache(
        structural,
        tensor_payloads=tensor_payloads,
    )
    ownership = tensor_payloads["unit_to_patient_index"].detach().cpu()
    if not torch.equal(ownership, identity_ownership):
        raise ValueError("cache ownership does not replay from identity payload")

    if route == PUBLIC_OOF_ROUTE:
        expected_fold_mask = torch.zeros_like(tensor_payloads["route_fold_mask"])
        expected_fold_mask[
            torch.arange(len(decisions)),
            torch.tensor([decision.fold_indices[0] for decision in decisions]),
        ] = True
        if not torch.equal(
            tensor_payloads["route_fold_mask"].detach().cpu(), expected_fold_mask
        ):
            raise ValueError("public held-fold mask disagrees with route receipts")
        replay = replay_public_oof_rows(
            trusted_registry,
            trusted_index,
            identities,
            relations,
        )
        indices = torch.tensor(C18_TO_STANDARD19, dtype=torch.long)
        _assert_equal(
            "p_h_c18",
            tensor_payloads["p_h_c18"],
            replay["p_h_node"].index_select(1, indices),
        )
        _assert_equal(
            "p_d_c18",
            tensor_payloads["p_d_c18"],
            replay["p_d_node"].index_select(1, indices),
        )
        _assert_equal(
            "p0_c18",
            tensor_payloads["p0_c18"],
            replay["p0_node"].index_select(1, indices),
        )
        _assert_equal("p0_node", tensor_payloads["p0_node"], replay["p0_node"])
    else:
        direct_states, h_states = load_v29_inference_states(trusted_registry)
        if (h_features is None) != (phase_features is None):
            raise ValueError("strict event replay requires both H and phase features")
        if h_features is not None and phase_features is not None:
            output = infer_v29_probabilities(
                h_features,
                phase_features,
                direct_states=direct_states,
                h_states=h_states,
            )
            indices = torch.tensor(C18_TO_STANDARD19, dtype=torch.long)
            _assert_equal(
                "p_h_fold_c18",
                tensor_payloads["p_h_fold_c18"],
                output["h_only_fold_probability"].index_select(2, indices),
            )
            _assert_equal(
                "p_d_fold_c18",
                tensor_payloads["p_d_fold_c18"],
                output["rank1_direct_fold_probability"].index_select(2, indices),
            )
            _assert_equal(
                "p_equal_fold_c18",
                tensor_payloads["p_equal_fold_c18"],
                output["portable_equal_fold_probability"].index_select(2, indices),
            )
            _assert_equal(
                "p0_node",
                tensor_payloads["p0_node"],
                output["portable_equal_probability"],
            )

    replayed_tensors = _freeze_tensor_mapping(tensor_payloads)
    replayed_p0 = replayed_tensors["p0_c18"]
    selected = hard_bypass_p0(
        replayed_p0,
        alpha=alpha,
        residual_supplier=residual_supplier,
    )
    alpha_zero_hard_bypass = float(alpha) == 0.0
    if alpha_zero_hard_bypass and selected is not replayed_p0:
        raise RuntimeError("alpha=0 did not hard-bypass to the replayed p0 tensor")
    selected_tensor_snapshot = (
        _CanonicalTensorSnapshot.from_tensor(selected)
        if not alpha_zero_hard_bypass and isinstance(selected, torch.Tensor)
        else None
    )
    selected_residual = (
        selected
        if not alpha_zero_hard_bypass and selected_tensor_snapshot is None
        else None
    )
    tensor_snapshots = _snapshot_tensor_mapping(replayed_tensors)
    unit_access_roles = tuple(
        decisions[int(patient_index)].access_role for patient_index in ownership.tolist()
    )
    return OpenedV29Cache(
        cache=_deep_freeze(validated_cache),
        route_decisions=tuple(_deep_freeze_decision(value) for value in decisions),
        unit_access_roles=unit_access_roles,
        _tensor_snapshots=tensor_snapshots,
        _alpha_zero_hard_bypass=alpha_zero_hard_bypass,
        _selected_residual=selected_residual,
        _selected_tensor_snapshot=selected_tensor_snapshot,
    )


__all__ = [
    "PATIENT_IDENTITY_PAYLOAD_SCHEMA_VERSION",
    "EVENT_IDENTITY_PAYLOAD_SCHEMA_VERSION",
    "MONTAGE_DERIVATION_PAYLOAD_SCHEMA_VERSION",
    "MATERIALIZER_VERSION",
    "V29MaterializedCache",
    "OpenedV29Cache",
    "materialize_public_v29_cache",
    "materialize_event_v29_cache",
    "open_v29_cache_for_use",
]
