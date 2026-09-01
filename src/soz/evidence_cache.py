"""Deterministic conversion from concept probabilities to finite evidence."""

from __future__ import annotations

import torch

from .evidence import EvidenceBatch
from .geometry import N_STANDARD_CHANNELS, N_TCP_EDGES
from .temporal_masks import IctalDeploymentMasks, physical_node_to_edge_mask


def _validate_probability_tensor(name: str, values: torch.Tensor) -> None:
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite floating-point probabilities")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} probabilities must lie in [0,1]")


def build_evidence_batch(
    morphology_probabilities: torch.Tensor,
    ictal_probabilities: torch.Tensor,
    evolution_descriptors: torch.Tensor,
    *,
    morphology_mask: torch.Tensor,
    ictal_deployment_masks: IctalDeploymentMasks,
    ictal_phase_mask: torch.Tensor,
    evolution_mask: torch.Tensor,
    morphology_context_mask: torch.Tensor | None = None,
    seconds_per_tile: int = 4,
) -> EvidenceBatch:
    """Aggregate second-level edge concepts into the frozen 4-second schema.

    A primary evidence tile is available only when all of its constituent
    seconds are available. ``ictal_deployment_masks`` contains producer and
    physical availability,
    never TUSZ ``source_target_mask``.  The latter is intentionally absent
    from this reasoner-cache API, so annotation coverage cannot become a
    learned missingness shortcut.
    """

    _validate_probability_tensor("morphology", morphology_probabilities)
    _validate_probability_tensor("ictal", ictal_probabilities)
    if morphology_probabilities.ndim != 4 or tuple(morphology_probabilities.shape[1::2]) != (N_TCP_EDGES, 6):
        raise ValueError("morphology_probabilities must have shape [B,20,S,6]")
    if ictal_probabilities.ndim != 4 or tuple(ictal_probabilities.shape[1::2]) != (N_TCP_EDGES, 1):
        raise ValueError("ictal_probabilities must have shape [B,20,S,1]")
    if morphology_probabilities.shape[:3] != ictal_probabilities.shape[:3]:
        raise ValueError("Morphology and ictal probabilities must share [B,20,S]")
    batch_size, _, n_seconds, _ = morphology_probabilities.shape
    if seconds_per_tile < 1 or n_seconds % seconds_per_tile != 0:
        raise ValueError("Second-level evidence must form complete fixed-size tiles")
    n_tiles = n_seconds // seconds_per_tile
    if tuple(morphology_mask.shape) != (batch_size, N_TCP_EDGES, n_seconds):
        raise ValueError("morphology_mask must have shape [B,20,S]")
    if morphology_context_mask is None:
        morphology_context_mask = morphology_mask
    if tuple(morphology_context_mask.shape) != (
        batch_size,
        N_TCP_EDGES,
        n_seconds,
    ):
        raise ValueError("morphology_context_mask must have shape [B,20,S]")
    if not isinstance(ictal_deployment_masks, IctalDeploymentMasks):
        raise TypeError(
            "ictal_deployment_masks must be a reasoner-safe "
            "IctalDeploymentMasks view"
        )
    ictal_deployment_mask = (
        ictal_deployment_masks.deployment_prediction_mask
    )
    physical_signal_mask = ictal_deployment_masks.physical_signal_mask
    if tuple(ictal_deployment_mask.shape) != (
        batch_size,
        N_TCP_EDGES,
        n_seconds,
    ):
        raise ValueError("ictal_deployment_mask must have shape [B,20,S]")
    if tuple(physical_signal_mask.shape) != (
        batch_size,
        N_STANDARD_CHANNELS,
        n_seconds,
    ):
        raise ValueError("physical_signal_mask must have shape [B,19,S]")
    if (
        morphology_mask.dtype != torch.bool
        or morphology_context_mask.dtype != torch.bool
        or ictal_deployment_mask.dtype != torch.bool
    ):
        raise TypeError("Edge concept masks must be torch.bool")
    physical_edge_seconds = ictal_deployment_masks.physical_edge_mask
    if (morphology_mask & ~physical_edge_seconds).any():
        raise ValueError(
            "morphology_mask cannot mark an edge with unavailable physical signal"
        )
    if (morphology_context_mask & ~physical_edge_seconds).any():
        raise ValueError(
            "morphology_context_mask cannot mark unavailable physical signal"
        )
    if (morphology_mask & ~morphology_context_mask).any():
        raise ValueError(
            "Localizing morphology mask must be a subset of context mask"
        )
    if tuple(evolution_descriptors.shape) != (
        batch_size,
        N_STANDARD_CHANNELS,
        n_tiles,
        6,
    ):
        raise ValueError("evolution_descriptors must have shape [B,19,S/4,6]")
    if tuple(evolution_mask.shape) != (batch_size, N_STANDARD_CHANNELS, n_tiles):
        raise ValueError("evolution_mask must have shape [B,19,S/4]")
    if evolution_mask.dtype != torch.bool:
        raise TypeError("evolution_mask must be torch.bool")
    if tuple(ictal_phase_mask.shape) != (batch_size, n_tiles):
        raise ValueError("ictal_phase_mask must have shape [B,S/4]")
    if ictal_phase_mask.dtype != torch.bool:
        raise TypeError("ictal_phase_mask must be torch.bool")
    if not evolution_descriptors.is_floating_point() or not torch.isfinite(evolution_descriptors).all():
        raise ValueError("evolution_descriptors must be finite floating-point values")

    morphology = morphology_probabilities.detach().reshape(
        batch_size, N_TCP_EDGES, n_tiles, seconds_per_tile, 6
    )
    ictal = ictal_probabilities.detach().reshape(
        batch_size, N_TCP_EDGES, n_tiles, seconds_per_tile, 1
    )
    morphology_second_mask = morphology_mask.reshape(
        batch_size, N_TCP_EDGES, n_tiles, seconds_per_tile
    )
    morphology_context_seconds = morphology_context_mask.reshape(
        batch_size, N_TCP_EDGES, n_tiles, seconds_per_tile
    )
    morphology_context_tile_mask = morphology_context_seconds.all(dim=-1)
    morphology_tile_mask = (
        morphology_second_mask.any(dim=-1) & morphology_context_tile_mask
    )
    physical_tile_mask, ictal_tile_mask = ictal_deployment_masks.tile_masks(
        seconds_per_tile=seconds_per_tile
    )
    # The physical edge mask is independent of whether either concept
    # producer emitted a prediction at that tile.
    edge_mask = physical_node_to_edge_mask(physical_tile_mask)

    # Context ports retain the CE6 summaries over every fully available
    # second.  SPSW/PLED localizing ports are instead conditional summaries
    # over frozen candidate-rule-positive anchors.  Mixing the two masks is
    # forbidden: otherwise low-confidence local morphology can add support or
    # artifact/GPED context disappears when the candidate rule abstains.
    morphology_mean = morphology.mean(dim=3)
    morphology_max = morphology.amax(dim=3)
    candidate_weight = morphology_second_mask.to(dtype=morphology.dtype)
    candidate_count = candidate_weight.sum(dim=3).clamp_min(1.0)
    candidate_mean = (
        morphology * candidate_weight.unsqueeze(-1)
    ).sum(dim=3) / candidate_count.unsqueeze(-1)
    candidate_max = torch.where(
        morphology_second_mask.unsqueeze(-1),
        morphology,
        torch.full_like(morphology, float("-inf")),
    ).amax(dim=3)
    candidate_max = torch.where(
        morphology_tile_mask.unsqueeze(-1),
        candidate_max,
        torch.zeros_like(candidate_max),
    )
    for class_index in (0, 2):  # SPSW and PLED
        morphology_mean[..., class_index] = candidate_mean[..., class_index]
        morphology_max[..., class_index] = candidate_max[..., class_index]
    ictal_mean = ictal.mean(dim=3)
    ictal_max = ictal.amax(dim=3)
    morphology_features = torch.cat([morphology_mean, morphology_max], dim=-1)
    local_feature_indices = (0, 2, 6, 8)
    context_feature_indices = (1, 3, 4, 5, 7, 9, 10, 11)
    morphology_features[..., list(local_feature_indices)] = torch.where(
        morphology_tile_mask.unsqueeze(-1),
        morphology_features[..., list(local_feature_indices)],
        0.0,
    )
    morphology_features[..., list(context_feature_indices)] = torch.where(
        morphology_context_tile_mask.unsqueeze(-1),
        morphology_features[..., list(context_feature_indices)],
        0.0,
    )
    ictal_features = torch.cat([ictal_mean, ictal_max], dim=-1)
    ictal_features = torch.where(
        ictal_tile_mask.unsqueeze(-1), ictal_features, 0.0
    )
    edge = torch.cat([morphology_features, ictal_features], dim=-1)
    node = torch.where(
        evolution_mask.unsqueeze(-1), evolution_descriptors.detach(), 0.0
    )
    return EvidenceBatch(
        node=node,
        edge=edge,
        node_mask=evolution_mask,
        edge_mask=edge_mask,
        physical_signal_mask=physical_tile_mask,
        ictal_phase_mask=ictal_phase_mask,
        morphology_mask=morphology_tile_mask,
        morphology_context_mask=morphology_context_tile_mask,
        ictal_mask=ictal_tile_mask,
    )
