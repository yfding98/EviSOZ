"""Authoritative typed semantics for the finite SOZ evidence carrier.

The numeric cache keeps one compact ``edge[...,14]`` carrier for efficient
serialization, but those 14 columns are not an untyped feature vector.  This
module freezes their names, order, coordinates, clinical ports, and signed
routing policy.  Cache manifests, receipts, content hashes, authorization
objects, and the reasoner all bind the same semantic digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import torch

from .geometry import (
    EVOLUTION_FEATURES,
    MORPHOLOGY_CLASSES,
    N_ICTAL_FEATURES,
    N_MORPHOLOGY_FEATURES,
    N_TCP_EDGES,
)


EVIDENCE_TENSOR_SEMANTICS_SCHEMA = "soz_typed_evidence_semantics_v2"
MORPHOLOGY_FAMILY = "morphology"
ICTAL_FAMILY = "ictal_involvement"
EVOLUTION_FAMILY = "temporal_evolution"

_MORPHOLOGY_SCOPE_BY_CLASS = {
    "SPSW": "candidate",
    "GPED": "context",
    "PLED": "candidate",
    "EYEM": "context",
    "ARTF": "context",
    "BCKG": "context",
}
MORPHOLOGY_FEATURE_NAMES = tuple(
    f"{class_name}_{_MORPHOLOGY_SCOPE_BY_CLASS[class_name]}_{summary}"
    for summary in ("mean", "max")
    for class_name in MORPHOLOGY_CLASSES
)
ICTAL_FEATURE_NAMES = (
    "ictal_involvement_mean",
    "ictal_involvement_max",
)
EVOLUTION_FEATURE_NAMES = tuple(EVOLUTION_FEATURES)
EDGE_FEATURE_NAMES = MORPHOLOGY_FEATURE_NAMES + ICTAL_FEATURE_NAMES

FAMILY_FEATURE_NAMES = {
    MORPHOLOGY_FAMILY: MORPHOLOGY_FEATURE_NAMES,
    ICTAL_FAMILY: ICTAL_FEATURE_NAMES,
    EVOLUTION_FAMILY: EVOLUTION_FEATURE_NAMES,
}

MORPHOLOGY_TYPED_PORT_FEATURE_NAMES = {
    "localizing_positive": (
        "SPSW_candidate_mean",
        "SPSW_candidate_max",
        "PLED_candidate_mean",
        "PLED_candidate_max",
    ),
    "generalized_conflict": ("GPED_context_mean", "GPED_context_max"),
    "quality_abstention": (
        "EYEM_context_mean",
        "EYEM_context_max",
        "ARTF_context_mean",
        "ARTF_context_max",
    ),
    "support_ood": ("BCKG_context_mean", "BCKG_context_max"),
}

_FEATURE_INDEX = {
    feature_name: index
    for index, feature_name in enumerate(MORPHOLOGY_FEATURE_NAMES)
}
MORPHOLOGY_TYPED_PORT_INDICES = {
    port_name: tuple(_FEATURE_INDEX[name] for name in names)
    for port_name, names in MORPHOLOGY_TYPED_PORT_FEATURE_NAMES.items()
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def family_feature_schema_payload(concept_family: str) -> dict[str, object]:
    """Return the exact family schema used by producer authorization."""

    if concept_family not in FAMILY_FEATURE_NAMES:
        raise ValueError(f"Unsupported evidence family: {concept_family!r}")
    payload: dict[str, object] = {
        "concept_family": concept_family,
        "feature_names": list(FAMILY_FEATURE_NAMES[concept_family]),
        "coordinate": (
            "physical_standard19_node"
            if concept_family == EVOLUTION_FAMILY
            else "common20_bipolar_edge"
        ),
        "storage_policy": (
            "four_second_explicit_descriptor"
            if concept_family == EVOLUTION_FAMILY
            else "four_second_port_specific_mean_block_then_max_block"
        ),
    }
    if concept_family == MORPHOLOGY_FAMILY:
        payload["typed_ports"] = {
            name: list(features)
            for name, features in MORPHOLOGY_TYPED_PORT_FEATURE_NAMES.items()
        }
        payload["positive_channel_logit_ports"] = ["localizing_positive"]
        payload["port_mask_policy"] = {
            "localizing_positive": "frozen_candidate_rule_mask",
            "generalized_conflict": "ce6_context_availability_mask",
            "quality_abstention": "ce6_context_availability_mask",
            "support_ood": "ce6_context_availability_mask",
        }
        payload["gped_edge_identity"] = (
            "discarded_by_masked_edge_mean_before_reasoning"
        )
        payload["nonlocalizing_port_policy"] = {
            "generalized_conflict": "non_increasing_specificity_gate_only",
            "quality_abstention": "non_increasing_reliability_gate_only",
            "support_ood": "receipt_and_ood_only_no_logit_path",
        }
    return payload


FAMILY_FEATURE_SCHEMA_SHA256 = {
    family: _canonical_sha256(family_feature_schema_payload(family))
    for family in FAMILY_FEATURE_NAMES
}


def evidence_tensor_semantics_payload() -> dict[str, object]:
    """Return a fresh JSON-ready payload for cache manifests."""

    return {
        "schema_version": EVIDENCE_TENSOR_SEMANTICS_SCHEMA,
        "node": {
            "coordinate": "physical_standard19_node",
            "feature_names": list(EVOLUTION_FEATURE_NAMES),
        },
        "edge": {
            "coordinate": "common20_bipolar_edge",
            "feature_names": list(EDGE_FEATURE_NAMES),
            "morphology_slice": [0, len(MORPHOLOGY_FEATURE_NAMES)],
            "ictal_slice": [
                len(MORPHOLOGY_FEATURE_NAMES),
                len(EDGE_FEATURE_NAMES),
            ],
        },
        "morphology": family_feature_schema_payload(MORPHOLOGY_FAMILY),
        "ictal_involvement": family_feature_schema_payload(ICTAL_FAMILY),
        "temporal_evolution": family_feature_schema_payload(EVOLUTION_FAMILY),
        "mask_policy": {
            "source_target_mask": "forbidden_from_reasoner_cache",
            "edge_mask": "physical_both_endpoints_available",
            "family_masks": "producer_availability_separate_by_family",
            "morphology_localizing_mask": (
                "candidate_rule_pass_subset_of_context"
            ),
            "morphology_context_mask": "ce6_context_availability",
            "ictal_phase_mask": "offset_aware_primary_phase_validity",
        },
    }


EVIDENCE_TENSOR_SEMANTICS_SHA256 = _canonical_sha256(
    evidence_tensor_semantics_payload()
)


@dataclass(frozen=True)
class TypedMorphologyEvidence:
    """Explicit clinical ports reconstructed from the bound flat carrier."""

    localizing_positive: torch.Tensor
    generalized_conflict: torch.Tensor
    generalized_mask: torch.Tensor
    quality_abstention: torch.Tensor
    support_ood: torch.Tensor
    localizing_mask: torch.Tensor
    context_mask: torch.Tensor

    def __post_init__(self) -> None:
        edge_shape = tuple(self.localizing_positive.shape[:-1])
        if len(edge_shape) != 3 or edge_shape[1] != N_TCP_EDGES:
            raise ValueError("Typed morphology edge ports require [B,20,T,F]")
        expected_features = {
            "localizing_positive": 4,
            "quality_abstention": 4,
            "support_ood": 2,
        }
        for name, n_features in expected_features.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != (*edge_shape, n_features):
                raise ValueError(f"{name} typed morphology shape changed")
        if tuple(self.generalized_conflict.shape) != (
            edge_shape[0],
            edge_shape[2],
            2,
        ):
            raise ValueError("GPED generalized port must be [B,T,2]")
        if tuple(self.generalized_mask.shape) != (
            edge_shape[0],
            edge_shape[2],
        ):
            raise ValueError("GPED generalized mask must be [B,T]")
        if tuple(self.localizing_mask.shape) != edge_shape or tuple(
            self.context_mask.shape
        ) != edge_shape:
            raise ValueError("Typed morphology port masks changed shape")
        if (
            self.generalized_mask.dtype != torch.bool
            or self.localizing_mask.dtype != torch.bool
            or self.context_mask.dtype != torch.bool
        ):
            raise TypeError("Typed morphology masks must be bool")
        if (self.localizing_mask & ~self.context_mask).any():
            raise ValueError("Morphology localizing mask must be within context")


def split_typed_morphology(
    flat_morphology: torch.Tensor,
    localizing_mask: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> TypedMorphologyEvidence:
    """Route bound CE6 mean/max storage into explicit clinical ports.

    GPED edge identity is destroyed inside this function.  No caller receives
    a spatially indexed GPED tensor from the typed API.
    """

    if (
        flat_morphology.ndim != 4
        or flat_morphology.shape[-1] != N_MORPHOLOGY_FEATURES
    ):
        raise ValueError("Flat morphology carrier must have shape [B,20,T,12]")
    if flat_morphology.shape[1] != N_TCP_EDGES:
        raise ValueError("Flat morphology carrier must use common20 edges")
    if context_mask is None:
        context_mask = localizing_mask
    for name, mask in (
        ("localizing", localizing_mask),
        ("context", context_mask),
    ):
        if tuple(mask.shape) != tuple(flat_morphology.shape[:-1]):
            raise ValueError(f"Morphology {name} mask must match [B,20,T]")
        if mask.dtype != torch.bool:
            raise TypeError(f"Morphology {name} mask must be bool")
    if (localizing_mask & ~context_mask).any():
        raise ValueError("Morphology localizing mask must be a context subset")
    if not flat_morphology.is_floating_point() or not torch.isfinite(
        flat_morphology
    ).all():
        raise ValueError("Morphology carrier must be finite floating point")

    observed = flat_morphology[context_mask]
    if observed.numel() and torch.any((observed < 0) | (observed > 1)):
        raise ValueError("Observed morphology evidence must lie in [0,1]")
    safe_context = torch.where(
        context_mask.unsqueeze(-1), flat_morphology, 0.0
    )
    safe_local = torch.where(
        localizing_mask.unsqueeze(-1), flat_morphology, 0.0
    )
    localizing = safe_local[
        ..., list(MORPHOLOGY_TYPED_PORT_INDICES["localizing_positive"])
    ]
    quality = safe_context[
        ..., list(MORPHOLOGY_TYPED_PORT_INDICES["quality_abstention"])
    ]
    support = safe_context[
        ..., list(MORPHOLOGY_TYPED_PORT_INDICES["support_ood"])
    ]
    generalized_by_edge = safe_context[
        ..., list(MORPHOLOGY_TYPED_PORT_INDICES["generalized_conflict"])
    ]
    edge_count = context_mask.sum(dim=1)
    generalized = generalized_by_edge.sum(dim=1) / edge_count.clamp_min(1).to(
        dtype=safe_context.dtype
    ).unsqueeze(-1)
    generalized_mask = edge_count > 0
    generalized = torch.where(
        generalized_mask.unsqueeze(-1), generalized, 0.0
    )
    return TypedMorphologyEvidence(
        localizing_positive=localizing,
        generalized_conflict=generalized,
        generalized_mask=generalized_mask,
        quality_abstention=quality,
        support_ood=support,
        localizing_mask=localizing_mask,
        context_mask=context_mask,
    )


def validate_typed_edge_cache(
    edge: torch.Tensor,
    morphology_mask: torch.Tensor,
    morphology_context_mask: torch.Tensor,
    ictal_mask: torch.Tensor,
    *,
    require_zero_masked: bool,
) -> None:
    """Validate the bound probability semantics of the serialized edge carrier."""

    if edge.ndim != 4 or edge.shape[-1] != len(EDGE_FEATURE_NAMES):
        raise ValueError("Typed edge carrier must have shape [B,20,T,14]")
    morphology = edge[..., :N_MORPHOLOGY_FEATURES]
    ictal = edge[
        ..., N_MORPHOLOGY_FEATURES : N_MORPHOLOGY_FEATURES + N_ICTAL_FEATURES
    ]
    split_typed_morphology(
        morphology, morphology_mask, morphology_context_mask
    )
    if (
        tuple(ictal_mask.shape) != tuple(ictal.shape[:-1])
        or ictal_mask.dtype != torch.bool
    ):
        raise ValueError("Ictal mask must be bool [B,20,T]")
    observed_ictal = ictal[ictal_mask]
    if observed_ictal.numel() and torch.any(
        (observed_ictal < 0) | (observed_ictal > 1)
    ):
        raise ValueError("Observed ictal evidence must lie in [0,1]")
    if require_zero_masked:
        local_indices = list(
            MORPHOLOGY_TYPED_PORT_INDICES["localizing_positive"]
        )
        context_indices = list(
            MORPHOLOGY_TYPED_PORT_INDICES["generalized_conflict"]
            + MORPHOLOGY_TYPED_PORT_INDICES["quality_abstention"]
            + MORPHOLOGY_TYPED_PORT_INDICES["support_ood"]
        )
        if torch.any(
            morphology[..., local_indices][~morphology_mask] != 0
        ):
            raise ValueError(
                "Masked localizing morphology values must use zero fill"
            )
        if torch.any(
            morphology[..., context_indices][~morphology_context_mask] != 0
        ):
            raise ValueError(
                "Masked morphology context values must use zero fill"
            )
        if torch.any(ictal[~ictal_mask] != 0):
            raise ValueError("Masked ictal cache values must use zero fill")

    for class_index, class_name in enumerate(MORPHOLOGY_CLASSES):
        mask = (
            morphology_mask
            if class_name in {"SPSW", "PLED"}
            else morphology_context_mask
        )
        if mask.any():
            means = morphology[..., class_index][mask]
            maxima = morphology[..., len(MORPHOLOGY_CLASSES) + class_index][mask]
            if torch.any(maxima + 1e-6 < means):
                raise ValueError(
                    f"{class_name} morphology maxima cannot be below means"
                )
    if ictal_mask.any():
        observed = ictal[ictal_mask]
        if torch.any(observed[:, 1] + 1e-6 < observed[:, 0]):
            raise ValueError("Ictal maxima cannot be below their means")


def require_current_evidence_semantics(sha256_value: object) -> str:
    """Reject a missing, legacy, or relabelled evidence semantic digest."""

    value = str(sha256_value).strip().lower()
    if value != EVIDENCE_TENSOR_SEMANTICS_SHA256:
        raise ValueError("Evidence tensor semantics are missing, legacy, or changed")
    return value


if len(MORPHOLOGY_FEATURE_NAMES) != N_MORPHOLOGY_FEATURES:
    raise RuntimeError("Morphology storage width disagrees with CE6 mean/max schema")
if len(ICTAL_FEATURE_NAMES) != N_ICTAL_FEATURES:
    raise RuntimeError("Ictal storage width disagrees with mean/max schema")
if len(set(EDGE_FEATURE_NAMES)) != len(EDGE_FEATURE_NAMES):
    raise RuntimeError("Evidence feature names must be unique")


__all__ = [
    "EDGE_FEATURE_NAMES",
    "EVIDENCE_TENSOR_SEMANTICS_SCHEMA",
    "EVIDENCE_TENSOR_SEMANTICS_SHA256",
    "EVOLUTION_FAMILY",
    "EVOLUTION_FEATURE_NAMES",
    "FAMILY_FEATURE_NAMES",
    "FAMILY_FEATURE_SCHEMA_SHA256",
    "ICTAL_FEATURE_NAMES",
    "ICTAL_FAMILY",
    "MORPHOLOGY_FEATURE_NAMES",
    "MORPHOLOGY_FAMILY",
    "MORPHOLOGY_TYPED_PORT_FEATURE_NAMES",
    "MORPHOLOGY_TYPED_PORT_INDICES",
    "TypedMorphologyEvidence",
    "evidence_tensor_semantics_payload",
    "family_feature_schema_payload",
    "require_current_evidence_semantics",
    "split_typed_morphology",
    "validate_typed_edge_cache",
]
