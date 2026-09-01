"""Fold-local node features for the anchor-constrained endpoint reranker.

This module contains no SOZ target logic.  It converts the frozen LaBraM
block-9 prefix cache plus the existing I/V/AQ evidence carrier into shared
physical-electrode features.  Every learned transform (PCA and feature
standardization) is fitted on an explicitly supplied outer-train patient
set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .development_reasoner import DevelopmentIVEvidenceBatch
from .geometry import (
    CHANNEL_INDEX,
    N_STANDARD_CHANNELS,
    TCP_20_EDGES,
    unsigned_incidence_matrix,
)
from .metrics import DEEPSOZ_STANDARD19_NEIGHBORS


N_H_RAW_FEATURES = 600
N_H_PCA_COMPONENTS = 8
N_H_PATIENT_FEATURES = 16
N_V_PATIENT_FEATURES = 36
N_I_PATIENT_FEATURES = 12
N_Q_PATIENT_FEATURES = 6
ENDPOINT_NODE_FEATURE_DIM = (
    N_H_PATIENT_FEATURES
    + N_V_PATIENT_FEATURES
    + N_I_PATIENT_FEATURES
    + N_Q_PATIENT_FEATURES
)

H_FEATURE_SLICE = slice(0, N_H_PATIENT_FEATURES)
V_FEATURE_SLICE = slice(
    N_H_PATIENT_FEATURES,
    N_H_PATIENT_FEATURES + N_V_PATIENT_FEATURES,
)
I_FEATURE_SLICE = slice(V_FEATURE_SLICE.stop, V_FEATURE_SLICE.stop + N_I_PATIENT_FEATURES)
Q_FEATURE_SLICE = slice(I_FEATURE_SLICE.stop, ENDPOINT_NODE_FEATURE_DIM)


def endpoint_adjacency_edges() -> tuple[tuple[int, int], ...]:
    """Return the frozen undirected TCP plus official one-hop graph."""

    edges: set[tuple[int, int]] = set()
    for left_name, right_name in TCP_20_EDGES:
        left = CHANNEL_INDEX[left_name]
        right = CHANNEL_INDEX[right_name]
        edges.add(tuple(sorted((left, right))))
    for left, neighbours in enumerate(DEEPSOZ_STANDARD19_NEIGHBORS):
        for right in neighbours:
            if left != int(right):
                edges.add(tuple(sorted((left, int(right)))))
    result = tuple(sorted(edges))
    if not result or any(a < 0 or b >= N_STANDARD_CHANNELS or a >= b for a, b in result):
        raise RuntimeError("endpoint adjacency graph is invalid")
    return result


def _masked_mean_and_std(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.ndim != 4 or mask.dtype != torch.bool or values.shape[:-1] != mask.shape:
        raise TypeError("masked moments require values [E,C,T,D] and bool mask [E,C,T]")
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("masked moment values must be finite floating point")
    count = mask.sum(dim=2)
    expanded = mask.unsqueeze(-1)
    safe = torch.where(expanded, values, torch.zeros_like(values))
    denominator = count.clamp_min(1).to(values.dtype).unsqueeze(-1)
    mean = safe.sum(dim=2) / denominator
    second = safe.square().sum(dim=2) / denominator
    std = (second - mean.square()).clamp_min(0).sqrt()
    valid = count > 0
    mean = torch.where(valid.unsqueeze(-1), mean, torch.zeros_like(mean))
    std = torch.where(valid.unsqueeze(-1), std, torch.zeros_like(std))
    return mean, std, valid


def event_temporal_summary(
    values: torch.Tensor,
    availability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full mean, early-minus-full and persistence SD per event/node."""

    if values.ndim != 4 or values.shape[2] != 15:
        raise ValueError("temporal values must have shape [E,C,15,D]")
    if availability.dtype != torch.bool or availability.shape != values.shape[:-1]:
        raise TypeError("temporal availability must be bool [E,C,15]")
    post_selector = torch.zeros(15, dtype=torch.bool, device=values.device)
    post_selector[3:] = True
    early_selector = torch.zeros_like(post_selector)
    early_selector[3:6] = True
    full_mean, full_std, full_valid = _masked_mean_and_std(
        values, availability & post_selector.view(1, 1, 15)
    )
    early_mean, _, early_valid = _masked_mean_and_std(
        values, availability & early_selector.view(1, 1, 15)
    )
    valid = full_valid & early_valid
    summary = torch.cat((full_mean, early_mean - full_mean, full_std), dim=-1)
    summary = torch.where(valid.unsqueeze(-1), summary, torch.zeros_like(summary))
    return summary.contiguous(), valid.contiguous()


def block9_event_node_summary(
    prefix_tokens: torch.Tensor,
    phase_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover physical nodes and summarize frozen LaBraM blocks 0--9."""

    if prefix_tokens.ndim != 4 or tuple(prefix_tokens.shape[1:]) != (15, 77, 200):
        raise ValueError("prefix_tokens must have shape [E,15,77,200]")
    if not prefix_tokens.is_floating_point() or not torch.isfinite(prefix_tokens).all():
        raise ValueError("prefix_tokens must be finite floating point")
    if phase_mask.dtype != torch.bool or tuple(phase_mask.shape) != (
        prefix_tokens.shape[0],
        15,
    ):
        raise TypeError("phase_mask must be bool [E,15]")
    node = (
        prefix_tokens[:, :, 1:, :]
        .reshape(prefix_tokens.shape[0], 15, N_STANDARD_CHANNELS, 4, 200)
        .permute(0, 2, 1, 3, 4)
    )
    tile = node.mean(dim=3)
    available = phase_mask.unsqueeze(1).expand(-1, N_STANDARD_CHANNELS, -1)
    summary, valid = event_temporal_summary(tile, available)
    if summary.shape[-1] != N_H_RAW_FEATURES:
        raise RuntimeError("LaBraM raw summary dimension changed")
    return summary, valid


def _ictal_node_tiles(
    evidence: DevelopmentIVEvidenceBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    incidence = unsigned_incidence_matrix().to(
        device=evidence.ictal.device, dtype=evidence.ictal.dtype
    )
    valid = evidence.ictal_mask.to(evidence.ictal.dtype)
    degree = torch.einsum("ce,bet->bct", incidence, valid)
    support = torch.einsum("ce,betf->bctf", incidence, evidence.ictal)
    support = support / degree.clamp_min(1).unsqueeze(-1)
    mask = degree > 0
    return torch.where(mask.unsqueeze(-1), support, torch.zeros_like(support)), mask


def ivq_event_node_summaries(
    evidence: DevelopmentIVEvidenceBatch,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return event-level V, unsigned-I and reliability summaries."""

    evidence.validate()
    phase = evidence.phase_mask.unsqueeze(1)
    v_summary, v_valid = event_temporal_summary(
        evidence.evolution,
        evidence.evolution_mask & phase,
    )
    ictal_node, ictal_mask = _ictal_node_tiles(evidence)
    i_summary, i_valid = event_temporal_summary(
        ictal_node,
        ictal_mask & phase,
    )

    post_selector = torch.zeros(15, dtype=torch.bool, device=phase.device)
    post_selector[3:] = True
    q_mask = evidence.evolution_mask & phase & post_selector.view(1, 1, 15)
    q_count = q_mask.sum(dim=2)
    q_valid = q_count > 0
    safe_q = torch.where(q_mask, evidence.reliability, torch.zeros_like(evidence.reliability))
    q_mean = safe_q.sum(dim=2) / q_count.clamp_min(1).to(safe_q.dtype)
    q_min = evidence.reliability.masked_fill(~q_mask, torch.inf).amin(dim=2)
    q_min = torch.where(q_valid, q_min, torch.zeros_like(q_min))
    phase_post_count = (
        evidence.phase_mask & post_selector.view(1, 15)
    ).sum(dim=1).clamp_min(1).to(safe_q.dtype)
    q_fraction = q_count.to(safe_q.dtype) / phase_post_count.unsqueeze(1)
    q_summary = torch.stack((q_mean, q_min, q_fraction), dim=-1)
    q_summary = torch.where(
        q_valid.unsqueeze(-1), q_summary, torch.zeros_like(q_summary)
    )
    if v_summary.shape[-1] != 18 or i_summary.shape[-1] != 6:
        raise RuntimeError("I/V temporal summary dimension changed")
    return v_summary, v_valid, i_summary, i_valid, q_summary, q_valid


def aggregate_patient_event_features(
    event_features: torch.Tensor,
    event_valid: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-event patient mean and population SD for every physical node."""

    if event_features.ndim != 3 or event_features.shape[:2] != event_valid.shape:
        raise ValueError("event features/validity must align as [E,19,D]/[E,19]")
    if event_valid.dtype != torch.bool or event_patient_index.dtype != torch.long:
        raise TypeError("event validity must be bool and patient index long")
    if tuple(event_patient_index.shape) != (event_features.shape[0],):
        raise ValueError("event_patient_index must align with events")
    if n_patients < 1 or int(event_patient_index.min()) != 0 or (
        int(event_patient_index.max()) != n_patients - 1
    ):
        raise ValueError("patient carrier is incomplete")
    rows: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for patient in range(n_patients):
        selected = event_patient_index == patient
        values = event_features[selected]
        mask = event_valid[selected]
        count = mask.sum(dim=0)
        safe = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
        denominator = count.clamp_min(1).to(values.dtype).unsqueeze(-1)
        mean = safe.sum(dim=0) / denominator
        second = safe.square().sum(dim=0) / denominator
        std = (second - mean.square()).clamp_min(0).sqrt()
        available = count > 0
        combined = torch.cat((mean, std), dim=-1)
        rows.append(
            torch.where(available.unsqueeze(-1), combined, torch.zeros_like(combined))
        )
        validity.append(available)
    return torch.stack(rows).contiguous(), torch.stack(validity).contiguous()


@dataclass(frozen=True)
class FoldEndpointFeatureState:
    h_center: torch.Tensor
    h_components: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor

    def __post_init__(self) -> None:
        if tuple(self.h_center.shape) != (N_H_RAW_FEATURES,):
            raise ValueError("h_center must have shape [600]")
        if tuple(self.h_components.shape) != (N_H_PCA_COMPONENTS, N_H_RAW_FEATURES):
            raise ValueError("h_components must have shape [8,600]")
        if tuple(self.feature_mean.shape) != (ENDPOINT_NODE_FEATURE_DIM,) or tuple(
            self.feature_scale.shape
        ) != (ENDPOINT_NODE_FEATURE_DIM,):
            raise ValueError("feature standardization must have shape [70]")
        values = (self.h_center, self.h_components, self.feature_mean, self.feature_scale)
        if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in values):
            raise ValueError("feature state must be finite floating point")
        if torch.any(self.feature_scale <= 0):
            raise ValueError("feature_scale must be positive")


def _fit_h_pca(
    h_event: torch.Tensor,
    h_valid: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
    train_patient_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    patient_h, patient_valid = aggregate_patient_event_features(
        h_event, h_valid, event_patient_index, n_patients
    )
    patient_mean = patient_h[..., :N_H_RAW_FEATURES]
    train_index = torch.tensor(tuple(train_patient_indices), dtype=torch.long)
    train_rows = patient_mean.index_select(0, train_index)
    train_valid = patient_valid.index_select(0, train_index)
    matrix = train_rows[train_valid]
    if matrix.shape[0] <= N_H_PCA_COMPONENTS:
        raise ValueError("outer-train H carrier is too small for fixed PCA")
    center = matrix.double().mean(dim=0)
    centered = matrix.double() - center
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:N_H_PCA_COMPONENTS].clone()
    pivot = components.abs().argmax(dim=1)
    signs = torch.sign(components[torch.arange(N_H_PCA_COMPONENTS), pivot])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    components = components * signs.unsqueeze(1)
    return center.float().contiguous(), components.float().contiguous()


def _patient_features_with_h_state(
    h_event: torch.Tensor,
    h_valid: torch.Tensor,
    v_event: torch.Tensor,
    v_valid: torch.Tensor,
    i_event: torch.Tensor,
    i_valid: torch.Tensor,
    q_event: torch.Tensor,
    q_valid: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
    h_center: torch.Tensor,
    h_components: torch.Tensor,
) -> torch.Tensor:
    projected = torch.matmul(
        h_event - h_center.view(1, 1, -1), h_components.transpose(0, 1)
    )
    projected = torch.where(h_valid.unsqueeze(-1), projected, torch.zeros_like(projected))
    h_patient, _ = aggregate_patient_event_features(
        projected, h_valid, event_patient_index, n_patients
    )
    v_patient, _ = aggregate_patient_event_features(
        v_event, v_valid, event_patient_index, n_patients
    )
    i_patient, _ = aggregate_patient_event_features(
        i_event, i_valid, event_patient_index, n_patients
    )
    q_patient, _ = aggregate_patient_event_features(
        q_event, q_valid, event_patient_index, n_patients
    )
    result = torch.cat((h_patient, v_patient, i_patient, q_patient), dim=-1)
    if tuple(result.shape) != (n_patients, N_STANDARD_CHANNELS, ENDPOINT_NODE_FEATURE_DIM):
        raise RuntimeError("endpoint node feature dimension changed")
    return result.contiguous()


def fit_fold_endpoint_features(
    prefix_tokens: torch.Tensor,
    evidence: DevelopmentIVEvidenceBatch,
    event_patient_index: torch.Tensor,
    n_patients: int,
    train_patient_indices: Sequence[int],
) -> tuple[torch.Tensor, FoldEndpointFeatureState]:
    """Fit target-free transforms on outer-train and transform all patients."""

    selected = tuple(int(index) for index in train_patient_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("train_patient_indices must be non-empty and unique")
    if any(index < 0 or index >= n_patients for index in selected):
        raise IndexError("outer-train patient index is out of range")
    h_event, h_valid = block9_event_node_summary(prefix_tokens, evidence.phase_mask)
    v_event, v_valid, i_event, i_valid, q_event, q_valid = ivq_event_node_summaries(
        evidence
    )
    center, components = _fit_h_pca(
        h_event,
        h_valid,
        event_patient_index,
        n_patients,
        selected,
    )
    raw = _patient_features_with_h_state(
        h_event,
        h_valid,
        v_event,
        v_valid,
        i_event,
        i_valid,
        q_event,
        q_valid,
        event_patient_index,
        n_patients,
        center,
        components,
    )
    train_index = torch.tensor(selected, dtype=torch.long)
    train_rows = raw.index_select(0, train_index).reshape(-1, ENDPOINT_NODE_FEATURE_DIM)
    feature_mean = train_rows.mean(dim=0)
    feature_scale = train_rows.std(dim=0, unbiased=False)
    feature_scale = torch.where(
        feature_scale > 1e-6, feature_scale, torch.ones_like(feature_scale)
    )
    standardized = (raw - feature_mean.view(1, 1, -1)) / feature_scale.view(1, 1, -1)
    if not torch.isfinite(standardized).all():
        raise RuntimeError("standardized endpoint features are non-finite")
    state = FoldEndpointFeatureState(
        h_center=center,
        h_components=components,
        feature_mean=feature_mean.contiguous(),
        feature_scale=feature_scale.contiguous(),
    )
    return standardized.float().contiguous(), state


def transform_endpoint_features(
    prefix_tokens: torch.Tensor,
    evidence: DevelopmentIVEvidenceBatch,
    event_patient_index: torch.Tensor,
    n_patients: int,
    state: FoldEndpointFeatureState,
) -> torch.Tensor:
    """Transform an inference cohort with a frozen source-train state.

    Unlike :func:`fit_fold_endpoint_features`, this function never estimates a
    PCA direction, mean, or scale from the supplied cohort.  It is therefore
    the only supported endpoint-feature path for a locked development/eval or
    external cohort.
    """

    if not isinstance(state, FoldEndpointFeatureState):
        raise TypeError("state must be a FoldEndpointFeatureState")
    if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer")
    h_event, h_valid = block9_event_node_summary(prefix_tokens, evidence.phase_mask)
    v_event, v_valid, i_event, i_valid, q_event, q_valid = ivq_event_node_summaries(
        evidence
    )
    raw = _patient_features_with_h_state(
        h_event,
        h_valid,
        v_event,
        v_valid,
        i_event,
        i_valid,
        q_event,
        q_valid,
        event_patient_index,
        n_patients,
        state.h_center,
        state.h_components,
    )
    transformed = (raw - state.feature_mean.view(1, 1, -1)) / state.feature_scale.view(
        1, 1, -1
    )
    if tuple(transformed.shape) != (
        n_patients,
        N_STANDARD_CHANNELS,
        ENDPOINT_NODE_FEATURE_DIM,
    ):
        raise RuntimeError("endpoint inference feature dimension changed")
    if not torch.isfinite(transformed).all():
        raise RuntimeError("transformed endpoint features are non-finite")
    return transformed.float().contiguous()


__all__ = [
    "ENDPOINT_NODE_FEATURE_DIM",
    "H_FEATURE_SLICE",
    "I_FEATURE_SLICE",
    "N_H_PCA_COMPONENTS",
    "N_H_RAW_FEATURES",
    "Q_FEATURE_SLICE",
    "V_FEATURE_SLICE",
    "FoldEndpointFeatureState",
    "aggregate_patient_event_features",
    "block9_event_node_summary",
    "endpoint_adjacency_edges",
    "event_temporal_summary",
    "fit_fold_endpoint_features",
    "ivq_event_node_summaries",
    "transform_endpoint_features",
]
