"""Content-closed cache contract for the frozen canonical v29 reference.

This module does not load or execute LaBraM, H, or D.  It only seals and
replays the tensors emitted by that frozen pipeline.  The two supported
routes are intentionally disjoint:

``historical_public_oof_held_fold``
    One row per public patient.  ``route_fold_mask`` selects exactly the
    patient's held-out fold.

``frozen_five_fold_event_mean``
    One row per newly materialized event.  All five frozen folds participate
    and ``unit_to_patient_index`` records the event-to-patient ownership.

The native probability coordinate is C18 (Standard19 without PZ).  Expansion
to Standard19 inserts one zero at PZ and performs no reordering, smoothing, or
renormalization.  The artifact reference for ``p0_c18`` is consequently the
baseline source of truth; callers using ``alpha == 0`` must return it directly
instead of performing a log/softmax round trip.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import math
import struct
from typing import Any, TypeVar

import torch

from src.evisoz.data.artifact_ref import (
    build_raw_artifact_ref,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_bytes,
    validate_artifact_ref,
    verify_artifact_content,
)
from src.soz.geometry import STANDARD_19
from src.soz.v11_reasoner import V11_CANDIDATE_MASK


V29_REFERENCE_CACHE_SCHEMA_VERSION = "evisoz_v29_reference_cache_v1"
TENSOR_ENCODING = "evisoz_tensor_bytes_v1"
TENSOR_MEDIA_TYPE = "application/vnd.evisoz.tensor"

PUBLIC_OOF_ROUTE = "historical_public_oof_held_fold"
EVENT_MEAN_ROUTE = "frozen_five_fold_event_mean"
ROUTE_RECEIPT_SCHEMA_VERSION = "evisoz_v29_route_receipt_v1"
ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION = "evisoz_v29_route_receipt_roster_v1"
FROZEN_RESOURCE_CONFIG_SCHEMA_VERSION = "evisoz_v29_frozen_resources_v1"
ROUTE_RECEIPT_ROSTER_ARTIFACT_KIND = "v29_route_receipt_roster"
ROUTE_UNIT_KIND = {
    PUBLIC_OOF_ROUTE: "patient",
    EVENT_MEAN_ROUTE: "event",
}
PREDICTION_RESOURCE_IDS = {
    PUBLIC_OOF_ROUTE: ("public_v29_oof",),
    EVENT_MEAN_ROUTE: ("direct_fold_states", "h_fold_states"),
}

N_FOLDS = 5
PZ_INDEX = STANDARD_19.index("PZ")
C18_ORDER = tuple(channel for channel in STANDARD_19 if channel != "PZ")
C18_TO_STANDARD19 = tuple(
    index for index, channel in enumerate(STANDARD_19) if channel != "PZ"
)

CORE_TENSOR_NAMES = (
    "p_h_c18",
    "p_d_c18",
    "p0_c18",
    "p0_node",
    "z0_c18",
    "candidate_mask",
    "route_fold_mask",
    "unit_to_patient_index",
)
EVENT_FOLD_TENSOR_NAMES = (
    "p_h_fold_c18",
    "p_d_fold_c18",
    "p_equal_fold_c18",
)
_FLOAT_TENSORS = {
    "p_h_c18",
    "p_d_c18",
    "p0_c18",
    "p0_node",
    "z0_c18",
    *EVENT_FOLD_TENSOR_NAMES,
}
_EXPECTED_DTYPES = {
    **{name: "float32" for name in _FLOAT_TENSORS},
    "candidate_mask": "bool",
    "route_fold_mask": "bool",
    "unit_to_patient_index": "int64",
}
_TORCH_DTYPES = {
    "float32": torch.float32,
    "bool": torch.bool,
    "int64": torch.int64,
}

_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-V29-"
_TOP_KEYS = {
    "schema_version",
    "cache_id",
    "route",
    "unit_kind",
    "unit_count",
    "patient_count",
    "fold_count",
    "channel_contract",
    "bindings",
    "tensor_receipts",
    "numerical_contract",
    "receipt_sha256",
}
_BINDING_KEYS = {
    "identity_ref",
    "event_identity_ref",
    "montage_derivation_receipt_ref",
    "route_receipt_roster_ref",
    "resource_manifest_ref",
    "prediction_resource_ids",
}
_TENSOR_RECEIPT_KEYS = {
    "artifact_ref",
    "encoding",
    "dtype",
    "shape",
    "byte_length",
    "tensor_sha256",
}


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["cache_id"] = _PENDING_ID
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _channel_contract() -> dict[str, object]:
    return {
        "native_coordinate": "standard19_without_pz_c18",
        "c18_order": list(C18_ORDER),
        "standard19_order": list(STANDARD_19),
        "c18_to_standard19_indices": list(C18_TO_STANDARD19),
        "inserted_node": "PZ",
        "inserted_node_index": PZ_INDEX,
        "inserted_probability": 0.0,
        "inserted_candidate_mask": False,
        "projection_operation": "insert_pz_zero_only_no_reorder_or_smoothing",
    }


def _numerical_contract(route: str) -> dict[str, object]:
    if route == PUBLIC_OOF_ROUTE:
        fusion = {
            "operation_order": "held_fold_h_d_probability_fusion",
            "h_weight": 0.5,
            "d_weight": 0.5,
            "equation": "p0_c18=0.5*p_h_c18+0.5*p_d_c18",
            "validation": "torch_equal_elementwise",
        }
    elif route == EVENT_MEAN_ROUTE:
        fusion = {
            "operation_order": "per_fold_h_d_fusion_then_five_fold_mean",
            "h_weight": 0.5,
            "d_weight": 0.5,
            "fold_equation": (
                "p_equal_fold_c18=0.5*p_h_fold_c18+0.5*p_d_fold_c18"
            ),
            "route_equations": [
                "p_h_c18=mean_fold(p_h_fold_c18)",
                "p_d_c18=mean_fold(p_d_fold_c18)",
                "p0_c18=mean_fold(p_equal_fold_c18)",
            ],
            "validation": "torch_equal_elementwise_in_canonical_operation_order",
            "aggregate_then_fuse_is_authoritative": False,
        }
    else:  # pragma: no cover - callers validate route first
        raise ValueError("unsupported v29 route")
    return {
        "fusion": fusion,
        "logit": {
            "equation": "z0_c18=log(p0_c18)",
            "epsilon_added": False,
            "zero_probability_logit": "negative_infinity",
        },
        "hard_bypass": {
            "alpha": 0.0,
            "return_value": "p0_c18_artifact_directly",
            "log_softmax_roundtrip": False,
            "residual_path_evaluated": False,
        },
        "tensor_content_replay_required_before_use": True,
    }


def _dtype_name(value: torch.Tensor) -> str:
    names = {
        torch.float32: "float32",
        torch.bool: "bool",
        torch.int64: "int64",
    }
    try:
        return names[value.dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported v29 cache tensor dtype: {value.dtype}") from exc


def _canonical_tensor(value: object, *, context: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{context} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{context} must be a dense strided tensor")
    if value.requires_grad:
        raise ValueError(f"{context} must be detached from autograd")
    _dtype_name(value)
    return value.detach().cpu().contiguous()


def canonical_tensor_bytes(value: torch.Tensor) -> bytes:
    """Encode one supported tensor with shape/dtype bound into its raw hash."""

    tensor = _canonical_tensor(value, context="tensor")
    dtype = _dtype_name(tensor)
    header = canonical_json_bytes(
        {
            "encoding": TENSOR_ENCODING,
            "dtype": dtype,
            "shape": list(tensor.shape),
            "byte_order": "little",
        }
    )
    # The frozen cache dtypes have unambiguous primitive storage.  Convert to
    # bytes through uint8 after moving to contiguous CPU storage so views,
    # strides and devices cannot alter the digest domain.
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return struct.pack(">Q", len(header)) + header + struct.pack(">Q", len(raw)) + raw


def tensor_sha256(value: torch.Tensor) -> str:
    """Return the v1 content digest for one canonical tensor payload."""

    return sha256_bytes(canonical_tensor_bytes(value))


def _expected_shape(name: str, unit_count: int) -> tuple[int, ...]:
    if name in {"p_h_c18", "p_d_c18", "p0_c18", "z0_c18"}:
        return (unit_count, 18)
    if name == "p0_node":
        return (unit_count, 19)
    if name == "candidate_mask":
        return (19,)
    if name == "route_fold_mask":
        return (unit_count, N_FOLDS)
    if name == "unit_to_patient_index":
        return (unit_count,)
    if name in EVENT_FOLD_TENSOR_NAMES:
        return (unit_count, N_FOLDS, 18)
    raise KeyError(name)


def _tensor_names_for_route(route: str) -> tuple[str, ...]:
    if route == PUBLIC_OOF_ROUTE:
        return CORE_TENSOR_NAMES
    if route == EVENT_MEAN_ROUTE:
        return CORE_TENSOR_NAMES + EVENT_FOLD_TENSOR_NAMES
    raise ValueError("unsupported v29 route")


def project_c18_to_standard19(
    values_c18: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert PZ=0 into the last dimension and return the frozen node mask.

    No value in C18 is modified, reordered, smoothed, or renormalized.
    ``candidate_mask`` is a one-dimensional frozen Standard19 registry mask.
    """

    if not isinstance(values_c18, torch.Tensor) or not values_c18.is_floating_point():
        raise TypeError("C18 values must be a floating-point torch.Tensor")
    if values_c18.ndim < 1 or values_c18.shape[-1] != 18:
        raise ValueError("C18 values must have last dimension 18")
    zero_shape = (*values_c18.shape[:-1], 1)
    pz = torch.zeros(zero_shape, dtype=values_c18.dtype, device=values_c18.device)
    projected = torch.cat(
        (values_c18[..., :PZ_INDEX], pz, values_c18[..., PZ_INDEX:]),
        dim=-1,
    )
    mask = V11_CANDIDATE_MASK.to(device=values_c18.device).clone()
    return projected, mask


_T = TypeVar("_T")


def hard_bypass_p0(
    p0: _T,
    *,
    alpha: float,
    residual_supplier: Callable[[], _T] | None = None,
) -> _T:
    """Return ``p0`` by identity before touching the residual path at alpha=0.

    The generic input may be the p0 tensor or its ArtifactRef.  For non-zero
    alpha this small utility delegates to an explicitly supplied residual
    computation; it never attempts to synthesize residual behavior itself.
    """

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a finite real number")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value):
        raise ValueError("alpha must be finite")
    if alpha_value == 0.0:
        return p0
    if residual_supplier is None:
        raise ValueError("non-zero alpha requires an explicit residual supplier")
    return residual_supplier()


def _validate_route_and_unit_kind(route: object, unit_kind: object) -> tuple[str, str]:
    if route not in ROUTE_UNIT_KIND:
        raise ValueError("unsupported frozen v29 cache route")
    expected = ROUTE_UNIT_KIND[str(route)]
    if unit_kind != expected:
        raise ValueError(f"route {route!r} requires unit_kind={expected!r}")
    return str(route), expected


def _validate_tensor_payloads(
    tensor_payloads: object,
    *,
    route: str,
    unit_count: int | None = None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    expected_names = _tensor_names_for_route(route)
    if not isinstance(tensor_payloads, Mapping) or set(tensor_payloads) != set(
        expected_names
    ):
        raise ValueError("v29 cache tensor payload names drifted")
    tensors = {
        name: _canonical_tensor(tensor_payloads[name], context=name)
        for name in expected_names
    }
    inferred_units = int(tensors["p0_c18"].shape[0]) if tensors["p0_c18"].ndim else 0
    if inferred_units < 1:
        raise ValueError("v29 cache must contain at least one unit")
    if unit_count is not None and inferred_units != unit_count:
        raise ValueError("v29 cache tensor unit count drifted")
    for name, tensor in tensors.items():
        expected_dtype = _TORCH_DTYPES[_EXPECTED_DTYPES[name]]
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"{name} must have dtype {_EXPECTED_DTYPES[name]}, got {_dtype_name(tensor)}"
            )
        if tuple(tensor.shape) != _expected_shape(name, inferred_units):
            raise ValueError(
                f"{name} must have shape {list(_expected_shape(name, inferred_units))}"
            )

    probability_names = ["p_h_c18", "p_d_c18", "p0_c18", "p0_node"]
    if route == EVENT_MEAN_ROUTE:
        probability_names.extend(EVENT_FOLD_TENSOR_NAMES)
    for name in probability_names:
        tensor = tensors[name]
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
        if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
            raise ValueError(f"{name} must contain probabilities in [0,1]")
    for name in ("p_h_c18", "p_d_c18", "p0_c18"):
        if not torch.allclose(
            tensors[name].sum(dim=1),
            torch.ones(inferred_units, dtype=torch.float32),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"{name} rows must sum to one")
    if route == EVENT_MEAN_ROUTE:
        for name in EVENT_FOLD_TENSOR_NAMES:
            if not torch.allclose(
                tensors[name].sum(dim=2),
                torch.ones((inferred_units, N_FOLDS), dtype=torch.float32),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(f"{name} fold rows must sum to one")

    if route == PUBLIC_OOF_ROUTE:
        expected_p0 = 0.5 * tensors["p_h_c18"] + 0.5 * tensors["p_d_c18"]
        if not torch.equal(tensors["p0_c18"], expected_p0):
            raise ValueError(
                "p0_c18 is not the elementwise held-fold H/D probability fusion"
            )
    else:
        expected_equal_fold = (
            0.5 * tensors["p_h_fold_c18"]
            + 0.5 * tensors["p_d_fold_c18"]
        )
        if not torch.equal(tensors["p_equal_fold_c18"], expected_equal_fold):
            raise ValueError(
                "p_equal_fold_c18 is not the elementwise per-fold H/D fusion"
            )
        if not torch.equal(
            tensors["p_h_c18"], tensors["p_h_fold_c18"].mean(dim=1)
        ):
            raise ValueError("p_h_c18 is not the canonical five-fold H mean")
        if not torch.equal(
            tensors["p_d_c18"], tensors["p_d_fold_c18"].mean(dim=1)
        ):
            raise ValueError("p_d_c18 is not the canonical five-fold D mean")
        if not torch.equal(
            tensors["p0_c18"], tensors["p_equal_fold_c18"].mean(dim=1)
        ):
            raise ValueError(
                "p0_c18 is not the canonical mean of per-fold equal H/D fusion"
            )
    expected_node, expected_mask = project_c18_to_standard19(tensors["p0_c18"])
    if not torch.equal(tensors["p0_node"], expected_node):
        raise ValueError("p0_node is not the exact C18 projection with PZ=0")
    if not torch.equal(tensors["candidate_mask"], expected_mask.cpu()):
        raise ValueError("candidate_mask drifted from the frozen Standard19 C18 mask")
    expected_z0 = torch.log(tensors["p0_c18"])
    z0 = tensors["z0_c18"]
    if bool(torch.isnan(z0).any()) or bool(torch.isposinf(z0).any()):
        raise ValueError("z0_c18 may contain only finite values or -inf at zero probability")
    if not torch.equal(z0, expected_z0):
        raise ValueError("z0_c18 must be log(p0_c18) without epsilon")

    fold_mask = tensors["route_fold_mask"]
    if route == PUBLIC_OOF_ROUTE:
        if not torch.equal(
            fold_mask.sum(dim=1),
            torch.ones(inferred_units, dtype=torch.int64),
        ):
            raise ValueError("public OOF route must select exactly one held-out fold per patient")
    elif route == EVENT_MEAN_ROUTE:
        if not bool(fold_mask.all()):
            raise ValueError("new-event route must use all five frozen folds")
    else:  # pragma: no cover - route validated by callers
        raise ValueError("unsupported v29 route")

    ownership = tensors["unit_to_patient_index"]
    if bool((ownership < 0).any()):
        raise ValueError("unit_to_patient_index cannot contain negative indices")
    patient_count = int(ownership.max().item()) + 1
    if route == PUBLIC_OOF_ROUTE:
        expected_ownership = torch.arange(inferred_units, dtype=torch.int64)
        if not torch.equal(ownership, expected_ownership):
            raise ValueError("public patient units require identity unit-to-patient mapping")
        patient_count = inferred_units
    else:
        observed = torch.unique(ownership, sorted=True)
        if not torch.equal(observed, torch.arange(patient_count, dtype=torch.int64)):
            raise ValueError("event ownership patient indices must be contiguous and represented")
    return tensors, inferred_units, patient_count


def _tensor_receipt(value: torch.Tensor) -> dict[str, object]:
    tensor = _canonical_tensor(value, context="tensor receipt")
    payload = canonical_tensor_bytes(tensor)
    artifact_ref = build_raw_artifact_ref(
        payload,
        artifact_kind="tensor_cache",
        media_type=TENSOR_MEDIA_TYPE,
    )
    return {
        "artifact_ref": artifact_ref,
        "encoding": TENSOR_ENCODING,
        "dtype": _dtype_name(tensor),
        "shape": list(tensor.shape),
        "byte_length": len(payload),
        "tensor_sha256": sha256_bytes(payload),
    }


def _optional_ref(value: object, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        return validate_artifact_ref(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a valid ArtifactRef") from exc


def _typed_ref(
    value: object,
    *,
    context: str,
    artifact_kind: str,
    hash_domain: str | None = None,
    media_type: str | None = None,
    payload_schema_version: str | None | object = ...,
) -> dict[str, Any]:
    result = _optional_ref(value, context)
    if result is None:
        raise ValueError(f"{context} is required")
    if result["artifact_kind"] != artifact_kind:
        raise ValueError(f"{context} must have artifact_kind={artifact_kind!r}")
    if hash_domain is not None and result["content_hash"]["domain"] != hash_domain:
        raise ValueError(f"{context} hash domain drifted")
    if media_type is not None and result["media_type"] != media_type:
        raise ValueError(f"{context} media_type drifted")
    if (
        payload_schema_version is not ...
        and result["payload_schema_version"] != payload_schema_version
    ):
        raise ValueError(f"{context} payload_schema_version drifted")
    return result


def _bindings(
    *,
    route: str,
    identity_ref: Mapping[str, object],
    event_identity_ref: Mapping[str, object] | None,
    montage_derivation_receipt_ref: Mapping[str, object] | None,
    route_receipt_roster_ref: Mapping[str, object],
    resource_manifest_ref: Mapping[str, object],
    prediction_resource_ids: Sequence[str],
) -> dict[str, object]:
    identity = _typed_ref(
        identity_ref,
        context="identity_ref",
        artifact_kind=(
            "patient_unit_roster" if route == PUBLIC_OOF_ROUTE else "event_unit_roster"
        ),
    )
    route_ref = _typed_ref(
        route_receipt_roster_ref,
        context="route_receipt_roster_ref",
        artifact_kind=ROUTE_RECEIPT_ROSTER_ARTIFACT_KIND,
        hash_domain="canonical_json_v1",
        media_type="application/json",
        payload_schema_version=ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION,
    )
    resource_ref = _typed_ref(
        resource_manifest_ref,
        context="resource_manifest_ref",
        artifact_kind="v29_frozen_resource_config",
        hash_domain="raw_bytes_v1",
        media_type="application/json",
        payload_schema_version=FROZEN_RESOURCE_CONFIG_SCHEMA_VERSION,
    )
    if route == PUBLIC_OOF_ROUTE:
        if event_identity_ref is not None or montage_derivation_receipt_ref is not None:
            raise ValueError(
                "historical public patient cache cannot carry event/montage bindings"
            )
        event_ref = None
        montage_ref = None
    else:
        event_ref = _typed_ref(
            event_identity_ref,
            context="event_identity_ref",
            artifact_kind="event_identity_roster",
        )
        montage_ref = _typed_ref(
            montage_derivation_receipt_ref,
            context="montage_derivation_receipt_ref",
            artifact_kind="montage_derivation_roster",
        )
    if isinstance(prediction_resource_ids, (str, bytes)) or not isinstance(
        prediction_resource_ids, Sequence
    ):
        raise TypeError("prediction_resource_ids must be an ordered string array")
    resources = tuple(prediction_resource_ids)
    if any(not isinstance(value, str) for value in resources):
        raise TypeError("prediction_resource_ids must contain strings")
    if resources != PREDICTION_RESOURCE_IDS[route]:
        raise ValueError(
            "prediction_resource_ids do not match the frozen route resources"
        )
    return {
        "identity_ref": identity,
        "event_identity_ref": event_ref,
        "montage_derivation_receipt_ref": montage_ref,
        "route_receipt_roster_ref": route_ref,
        "resource_manifest_ref": resource_ref,
        "prediction_resource_ids": list(resources),
    }


def build_v29_reference_cache(
    *,
    route: str,
    unit_kind: str,
    tensor_payloads: Mapping[str, torch.Tensor],
    identity_ref: Mapping[str, object],
    route_receipt_roster_ref: Mapping[str, object],
    resource_manifest_ref: Mapping[str, object],
    prediction_resource_ids: Sequence[str],
    event_identity_ref: Mapping[str, object] | None = None,
    montage_derivation_receipt_ref: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Seal already-materialized frozen v29 tensors without invoking a model."""

    route_value, unit_kind_value = _validate_route_and_unit_kind(route, unit_kind)
    tensors, unit_count, patient_count = _validate_tensor_payloads(
        tensor_payloads,
        route=route_value,
    )
    bindings = _bindings(
        route=route_value,
        identity_ref=identity_ref,
        event_identity_ref=event_identity_ref,
        montage_derivation_receipt_ref=montage_derivation_receipt_ref,
        route_receipt_roster_ref=route_receipt_roster_ref,
        resource_manifest_ref=resource_manifest_ref,
        prediction_resource_ids=prediction_resource_ids,
    )
    body: dict[str, Any] = {
        "schema_version": V29_REFERENCE_CACHE_SCHEMA_VERSION,
        "cache_id": _PENDING_ID,
        "route": route_value,
        "unit_kind": unit_kind_value,
        "unit_count": unit_count,
        "patient_count": patient_count,
        "fold_count": N_FOLDS,
        "channel_contract": _channel_contract(),
        "bindings": bindings,
        "tensor_receipts": {
            name: _tensor_receipt(tensors[name])
            for name in _tensor_names_for_route(route_value)
        },
        "numerical_contract": _numerical_contract(route_value),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["cache_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_v29_reference_cache(body, tensor_payloads=tensors)


def _validate_tensor_receipt(
    name: str,
    value: object,
    *,
    unit_count: int,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _TENSOR_RECEIPT_KEYS:
        raise ValueError(f"tensor_receipts.{name} fields drifted")
    row = deepcopy(value)
    if row["encoding"] != TENSOR_ENCODING:
        raise ValueError(f"tensor_receipts.{name} encoding drifted")
    if row["dtype"] != _EXPECTED_DTYPES[name]:
        raise ValueError(f"tensor_receipts.{name} dtype drifted")
    if row["shape"] != list(_expected_shape(name, unit_count)):
        raise ValueError(f"tensor_receipts.{name} shape drifted")
    byte_length = row["byte_length"]
    if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length < 1:
        raise TypeError(f"tensor_receipts.{name}.byte_length must be positive")
    digest = row["tensor_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"tensor_receipts.{name}.tensor_sha256 is invalid")
    ref = validate_artifact_ref(row["artifact_ref"])
    if (
        ref["artifact_kind"] != "tensor_cache"
        or ref["media_type"] != TENSOR_MEDIA_TYPE
        or ref["content_hash"]["domain"] != "raw_bytes_v1"
        or ref["payload_schema_version"] is not None
    ):
        raise ValueError(f"tensor_receipts.{name} ArtifactRef type/domain drifted")
    if ref["content_hash"]["sha256"] != digest:
        raise ValueError(f"tensor_receipts.{name} hash receipt disagrees with ArtifactRef")
    if ref["content_hash"]["size_bytes"] != byte_length:
        raise ValueError(f"tensor_receipts.{name} byte length disagrees with ArtifactRef")
    row["artifact_ref"] = ref
    return row


def validate_v29_reference_cache(
    value: object,
    *,
    tensor_payloads: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Validate a cache receipt and optionally replay every tensor byte/value.

    Structural validation closes all references and receipt hashes.  Passing
    ``tensor_payloads`` additionally verifies artifact bytes, exact H/D fusion,
    C18 projection, logits, fold routing, and unit ownership.  Consumers must
    perform that content replay before using a cache, as recorded by the
    numerical contract.
    """

    if type(value) is not dict or set(value) != _TOP_KEYS:
        raise ValueError("v29 reference cache fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != V29_REFERENCE_CACHE_SCHEMA_VERSION:
        raise ValueError("v29 reference cache schema_version drifted")
    route, unit_kind = _validate_route_and_unit_kind(data["route"], data["unit_kind"])
    for field in ("unit_count", "patient_count"):
        number = data[field]
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise TypeError(f"{field} must be an integer >= 1")
    unit_count = data["unit_count"]
    patient_count = data["patient_count"]
    if data["fold_count"] != N_FOLDS:
        raise ValueError("frozen v29 cache must retain exactly five folds")
    if data["channel_contract"] != _channel_contract():
        raise ValueError("v29 C18/Standard19 channel contract drifted")

    bindings = data["bindings"]
    if type(bindings) is not dict or set(bindings) != _BINDING_KEYS:
        raise ValueError("v29 cache binding fields drifted")
    data["bindings"] = _bindings(
        route=route,
        identity_ref=bindings["identity_ref"],
        event_identity_ref=bindings["event_identity_ref"],
        montage_derivation_receipt_ref=bindings[
            "montage_derivation_receipt_ref"
        ],
        route_receipt_roster_ref=bindings["route_receipt_roster_ref"],
        resource_manifest_ref=bindings["resource_manifest_ref"],
        prediction_resource_ids=bindings["prediction_resource_ids"],
    )

    receipts = data["tensor_receipts"]
    tensor_names = _tensor_names_for_route(route)
    if type(receipts) is not dict or set(receipts) != set(tensor_names):
        raise ValueError("v29 cache tensor receipt roster drifted")
    validated_receipts = {
        name: _validate_tensor_receipt(name, receipts[name], unit_count=unit_count)
        for name in tensor_names
    }
    data["tensor_receipts"] = validated_receipts
    if data["numerical_contract"] != _numerical_contract(route):
        raise ValueError("v29 cache numerical contract drifted")

    if tensor_payloads is not None:
        tensors, replay_units, replay_patients = _validate_tensor_payloads(
            tensor_payloads,
            route=route,
            unit_count=unit_count,
        )
        if replay_units != unit_count or replay_patients != patient_count:
            raise ValueError("v29 cache unit/patient counts do not replay from tensors")
        for name, tensor in tensors.items():
            row = validated_receipts[name]
            payload = canonical_tensor_bytes(tensor)
            verify_artifact_content(row["artifact_ref"], payload)
            if tensor_sha256(tensor) != row["tensor_sha256"]:
                raise ValueError(f"tensor_receipts.{name} does not bind supplied tensor")

    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["cache_id"] != expected_id:
        raise ValueError("v29 reference cache_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("v29 reference receipt_sha256 does not bind its content")
    return data


__all__ = [
    "V29_REFERENCE_CACHE_SCHEMA_VERSION",
    "TENSOR_ENCODING",
    "PUBLIC_OOF_ROUTE",
    "EVENT_MEAN_ROUTE",
    "ROUTE_UNIT_KIND",
    "ROUTE_RECEIPT_SCHEMA_VERSION",
    "ROUTE_RECEIPT_ROSTER_SCHEMA_VERSION",
    "ROUTE_RECEIPT_ROSTER_ARTIFACT_KIND",
    "FROZEN_RESOURCE_CONFIG_SCHEMA_VERSION",
    "PREDICTION_RESOURCE_IDS",
    "C18_ORDER",
    "C18_TO_STANDARD19",
    "CORE_TENSOR_NAMES",
    "EVENT_FOLD_TENSOR_NAMES",
    "canonical_tensor_bytes",
    "tensor_sha256",
    "project_c18_to_standard19",
    "hard_bypass_p0",
    "build_v29_reference_cache",
    "validate_v29_reference_cache",
]
