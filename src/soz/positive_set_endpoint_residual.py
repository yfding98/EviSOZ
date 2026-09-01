"""Minimal C-CAR19 positive-set endpoint residual.

This module implements the frozen, single-reference recovery mechanism that
precedes any runner or promotion protocol.  Its interfaces deliberately have
no raw EEG path, SOZ text, channel identity, quality feature, or spatial I
port:

* bipolar ictal evidence is first reduced over all edges and becomes one
  ``[event, time]`` gate shared by every physical electrode;
* frozen LaBraM block-9 node tokens and the six observable V descriptors are
  summarized as an early, I-gated value minus a pre-anchor baseline;
* an outer-train-only PCA reduces H from 200 to 8 dimensions, giving exactly
  ``H8 + V6 = 14`` patient/node features after equal-event aggregation; and
* one shared bias-free linear utility is optimized with patient positive-set
  mass plus fixed L2 regularization.

The resulting scores are operational DeepSOZ benchmark rankings.  They are
not calibrated SOZ probabilities, propagation estimates, or proof of a
cortical generator.  In particular, an observed zero remains only a benchmark
complement.  Target-mask sensitivity helpers below only censor observed
labels; they never infer or write a missing label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    N_NODE_FEATURES,
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
    N_TIME_TILES,
)


POSITIVE_SET_ENDPOINT_RESIDUAL_SCHEMA = (
    "soz_c_car19_positive_set_endpoint_residual_v10"
)
PRE_TILE_INDICES = (0, 1, 2)
EARLY_TILE_INDICES = (3, 4, 5)
H_TOKEN_DIM = 200
H_PCA_DIM = 8
V_FEATURE_DIM = N_NODE_FEATURES
POSITIVE_SET_ENDPOINT_FEATURE_DIM = H_PCA_DIM + V_FEATURE_DIM
POSITIVE_SET_ENDPOINT_L2_WEIGHT = 0.05

H_FEATURE_SLICE = slice(0, H_PCA_DIM)
V_FEATURE_SLICE = slice(H_PCA_DIM, POSITIVE_SET_ENDPOINT_FEATURE_DIM)


def _require_finite_detached(value: torch.Tensor, *, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached from its upstream producer")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class SharedEarlyIGate:
    """A spatially invariant I gate over the frozen early tiles.

    ``weights`` and ``global_support`` have no channel or edge axis.  This is
    the structural guarantee that I cannot decide between bipolar endpoints.
    """

    weights: torch.Tensor
    global_support: torch.Tensor
    early_tile_mask: torch.Tensor
    event_valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.weights.ndim != 2 or self.weights.shape[1] != N_TIME_TILES:
            raise ValueError("I gate weights must have shape [E,15]")
        expected = tuple(self.weights.shape)
        if tuple(self.global_support.shape) != expected:
            raise ValueError("global I support must have shape [E,15]")
        if tuple(self.early_tile_mask.shape) != expected or (
            self.early_tile_mask.dtype != torch.bool
        ):
            raise TypeError("early I tile mask must be bool [E,15]")
        if tuple(self.event_valid.shape) != (self.weights.shape[0],) or (
            self.event_valid.dtype != torch.bool
        ):
            raise TypeError("I gate event_valid must be bool [E]")
        tensors = (
            self.weights,
            self.global_support,
            self.early_tile_mask,
            self.event_valid,
        )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all I gate tensors must share a device")
        _require_finite_detached(self.weights, name="I gate weights")
        _require_finite_detached(self.global_support, name="global I support")
        if torch.any(self.weights < 0):
            raise ValueError("I gate weights must be non-negative")
        selector = torch.zeros(
            N_TIME_TILES, dtype=torch.bool, device=self.weights.device
        )
        selector[list(EARLY_TILE_INDICES)] = True
        if bool((self.early_tile_mask & ~selector.view(1, -1)).any()):
            raise ValueError("I gate mask contains a non-early tile")
        expected_valid = self.early_tile_mask[:, list(EARLY_TILE_INDICES)].all(
            dim=1
        )
        if not torch.equal(self.event_valid, expected_valid):
            raise ValueError("I gate event validity disagrees with its tile mask")
        if bool((self.weights[~self.early_tile_mask] != 0).any()):
            raise ValueError("I gate assigns weight to an unavailable tile")
        sums = self.weights.sum(dim=1)
        expected_sums = self.event_valid.to(self.weights.dtype)
        if not torch.allclose(sums, expected_sums, atol=1e-6, rtol=1e-6):
            raise ValueError("valid I gates must sum to one and invalid gates to zero")


def build_shared_early_i_gate(
    ictal: torch.Tensor,
    ictal_mask: torch.Tensor,
    phase_mask: torch.Tensor,
) -> SharedEarlyIGate:
    """Collapse I over all TCP edges, then softmax only over tiles 3--5.

    The two frozen I features (four-second mean and maximum) are averaged with
    equal, non-learned weight.  Available edges are then averaged.  No
    incidence matrix, endpoint index, channel index, or learned parameter is
    present in this operation.
    """

    if ictal.ndim != 4 or tuple(ictal.shape[1:]) != (
        N_TCP_EDGES,
        N_TIME_TILES,
        2,
    ):
        raise ValueError("I evidence must have shape [E,20,15,2]")
    events = int(ictal.shape[0])
    if events < 1:
        raise ValueError("I evidence must contain at least one event")
    if tuple(ictal_mask.shape) != (events, N_TCP_EDGES, N_TIME_TILES) or (
        ictal_mask.dtype != torch.bool
    ):
        raise TypeError("I mask must be bool [E,20,15]")
    if tuple(phase_mask.shape) != (events, N_TIME_TILES) or (
        phase_mask.dtype != torch.bool
    ):
        raise TypeError("phase mask must be bool [E,15]")
    if len({ictal.device, ictal_mask.device, phase_mask.device}) != 1:
        raise ValueError("I evidence and masks must share a device")
    _require_finite_detached(ictal, name="I evidence")
    observed = ictal[ictal_mask]
    if observed.numel() and bool(((observed < 0) | (observed > 1)).any()):
        raise ValueError("observed I evidence must lie in [0,1]")

    edge_mask = ictal_mask & phase_mask.unsqueeze(1)
    edge_score = ictal.mean(dim=-1)
    edge_count = edge_mask.sum(dim=1)
    # Compare the same complete TCP-20 carrier at each early tile.  A
    # changing subset of edges would let montage/mask composition masquerade
    # as temporal ictal change even though edge identity is later collapsed.
    tile_available = edge_count == N_TCP_EDGES
    support = torch.where(edge_mask, edge_score, torch.zeros_like(edge_score)).sum(
        dim=1
    )
    support = support / edge_count.clamp_min(1).to(support.dtype)
    support = torch.where(tile_available, support, torch.zeros_like(support))

    selector = torch.zeros(
        N_TIME_TILES, dtype=torch.bool, device=ictal.device
    )
    selector[list(EARLY_TILE_INDICES)] = True
    early_mask = tile_available & selector.view(1, -1)
    # A complete early carrier is mandatory.  Missing one of the three tiles
    # invalidates the event; we never renormalize a shorter temporal window.
    event_valid = early_mask[:, list(EARLY_TILE_INDICES)].all(dim=1)
    masked_energy = support.masked_fill(~early_mask, -torch.inf)
    safe_energy = torch.where(
        event_valid.unsqueeze(1), masked_energy, torch.zeros_like(masked_energy)
    )
    weights = torch.softmax(safe_energy, dim=1)
    weights = torch.where(
        event_valid.unsqueeze(1) & early_mask,
        weights,
        torch.zeros_like(weights),
    )
    return SharedEarlyIGate(
        weights=weights.contiguous(),
        global_support=support.contiguous(),
        early_tile_mask=early_mask.contiguous(),
        event_valid=event_valid.contiguous(),
    )


@dataclass(frozen=True)
class EarlyPreEndpointContrasts:
    """Detached event/node H and V early-minus-pre contrasts."""

    h: torch.Tensor
    v: torch.Tensor
    node_mask: torch.Tensor
    temporal_gate: SharedEarlyIGate

    def __post_init__(self) -> None:
        if self.h.ndim != 3 or tuple(self.h.shape[1:]) != (
            N_STANDARD_CHANNELS,
            H_TOKEN_DIM,
        ):
            raise ValueError("H contrast must have shape [E,19,200]")
        events = int(self.h.shape[0])
        if tuple(self.v.shape) != (
            events,
            N_STANDARD_CHANNELS,
            V_FEATURE_DIM,
        ):
            raise ValueError("V contrast must have shape [E,19,6]")
        if tuple(self.node_mask.shape) != (events, N_STANDARD_CHANNELS) or (
            self.node_mask.dtype != torch.bool
        ):
            raise TypeError("contrast node mask must be bool [E,19]")
        if not isinstance(self.temporal_gate, SharedEarlyIGate) or (
            self.temporal_gate.weights.shape[0] != events
        ):
            raise ValueError("contrast temporal gate does not align with events")
        tensors = (self.h, self.v, self.node_mask, self.temporal_gate.weights)
        if len({value.device for value in tensors}) != 1:
            raise ValueError("contrast tensors must share a device")
        _require_finite_detached(self.h, name="H contrast")
        _require_finite_detached(self.v, name="V contrast")
        if bool((self.h[~self.node_mask] != 0).any()) or bool(
            (self.v[~self.node_mask] != 0).any()
        ):
            raise ValueError("unavailable event/node contrasts must be exactly zero")


def _block9_node_tiles(prefix_tokens: torch.Tensor) -> torch.Tensor:
    if prefix_tokens.ndim != 4 or tuple(prefix_tokens.shape[1:]) != (
        N_TIME_TILES,
        77,
        H_TOKEN_DIM,
    ):
        raise ValueError("LaBraM prefix must have shape [E,15,77,200]")
    if prefix_tokens.shape[0] < 1:
        raise ValueError("LaBraM prefix must contain at least one event")
    _require_finite_detached(prefix_tokens, name="LaBraM prefix")
    events = int(prefix_tokens.shape[0])
    return (
        prefix_tokens[:, :, 1:, :]
        .reshape(events, N_TIME_TILES, N_STANDARD_CHANNELS, 4, H_TOKEN_DIM)
        .mean(dim=3)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def build_early_pre_endpoint_contrasts(
    prefix_tokens: torch.Tensor,
    evolution: torch.Tensor,
    evolution_mask: torch.Tensor,
    phase_mask: torch.Tensor,
    temporal_gate: SharedEarlyIGate,
) -> EarlyPreEndpointContrasts:
    """Create H/V early-gated minus pre-anchor event/node contrasts.

    The same ``[E,15]`` weights are used for every H/V node.  A node is
    available only when all three pre tiles and every positively weighted
    early tile are available.  Missing tiles are not renormalized per channel,
    which would otherwise create a hidden spatial I route.
    """

    h_tiles = _block9_node_tiles(prefix_tokens)
    events = int(h_tiles.shape[0])
    if evolution.ndim != 4 or tuple(evolution.shape) != (
        events,
        N_STANDARD_CHANNELS,
        N_TIME_TILES,
        V_FEATURE_DIM,
    ):
        raise ValueError("V evidence must have shape [E,19,15,6]")
    if tuple(evolution_mask.shape) != (
        events,
        N_STANDARD_CHANNELS,
        N_TIME_TILES,
    ) or evolution_mask.dtype != torch.bool:
        raise TypeError("V mask must be bool [E,19,15]")
    if tuple(phase_mask.shape) != (events, N_TIME_TILES) or (
        phase_mask.dtype != torch.bool
    ):
        raise TypeError("phase mask must be bool [E,15]")
    if not isinstance(temporal_gate, SharedEarlyIGate) or (
        temporal_gate.weights.shape[0] != events
    ):
        raise ValueError("temporal gate must align with the event carrier")
    tensors = (
        prefix_tokens,
        evolution,
        evolution_mask,
        phase_mask,
        temporal_gate.weights,
    )
    if len({value.device for value in tensors}) != 1:
        raise ValueError("H/V inputs and temporal gate must share a device")
    _require_finite_detached(evolution, name="V evidence")
    if bool((temporal_gate.early_tile_mask & ~phase_mask).any()):
        raise ValueError("I gate uses a tile excluded by the supplied phase mask")

    node_tile_mask = evolution_mask & phase_mask.unsqueeze(1)
    pre_selector = torch.tensor(
        PRE_TILE_INDICES, dtype=torch.long, device=evolution.device
    )
    pre_valid = node_tile_mask.index_select(2, pre_selector).all(dim=2)
    early_missing = temporal_gate.early_tile_mask.unsqueeze(1) & ~node_tile_mask
    early_valid = temporal_gate.event_valid.unsqueeze(1) & ~early_missing.any(dim=2)
    node_valid = pre_valid & early_valid
    partially_valid = node_valid.any(dim=1) & ~node_valid.all(dim=1)
    if bool(partially_valid.any()):
        raise ValueError(
            "H/V residual events must be common-carrier all-19 or unavailable"
        )

    weights = temporal_gate.weights.to(h_tiles.dtype)
    early_h = torch.einsum("et,ectd->ecd", weights, h_tiles)
    pre_h = h_tiles.index_select(2, pre_selector).mean(dim=2)
    early_v = torch.einsum(
        "et,ectd->ecd", temporal_gate.weights.to(evolution.dtype), evolution
    )
    pre_v = evolution.index_select(2, pre_selector).mean(dim=2)
    h = torch.where(
        node_valid.unsqueeze(-1), early_h - pre_h, torch.zeros_like(early_h)
    )
    v = torch.where(
        node_valid.unsqueeze(-1), early_v - pre_v, torch.zeros_like(early_v)
    )
    return EarlyPreEndpointContrasts(
        h=h.contiguous(),
        v=v.contiguous(),
        node_mask=node_valid.contiguous(),
        temporal_gate=temporal_gate,
    )


def aggregate_patient_event_mean(
    event_values: torch.Tensor,
    event_mask: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equally average available seizure events inside every patient/node."""

    if event_values.ndim != 3 or event_values.shape[:2] != event_mask.shape:
        raise ValueError("event values/mask must have shape [E,19,D]/[E,19]")
    events = int(event_values.shape[0])
    if events < 1 or event_values.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("event values must contain standard-19 events")
    if event_mask.dtype != torch.bool:
        raise TypeError("event mask must be bool [E,19]")
    if tuple(event_patient_index.shape) != (events,) or (
        event_patient_index.dtype != torch.long
    ):
        raise TypeError("event_patient_index must be long [E]")
    if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer")
    tensors = (event_values, event_mask, event_patient_index)
    if len({value.device for value in tensors}) != 1:
        raise ValueError("event features, mask, and patient index must share a device")
    _require_finite_detached(event_values, name="event endpoint features")
    if int(event_patient_index.min()) != 0 or int(event_patient_index.max()) != (
        n_patients - 1
    ) or int(torch.unique(event_patient_index).numel()) != n_patients:
        raise ValueError("event_patient_index must cover every patient contiguously")

    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    for patient in range(n_patients):
        selected = event_patient_index == patient
        values = event_values[selected]
        available = event_mask[selected]
        count = available.sum(dim=0)
        safe = torch.where(available.unsqueeze(-1), values, torch.zeros_like(values))
        mean = safe.sum(dim=0) / count.clamp_min(1).to(values.dtype).unsqueeze(-1)
        valid = count > 0
        rows.append(torch.where(valid.unsqueeze(-1), mean, torch.zeros_like(mean)))
        masks.append(valid)
        counts.append(count)
    return (
        torch.stack(rows).contiguous(),
        torch.stack(masks).contiguous(),
        torch.stack(counts).contiguous(),
    )


@dataclass(frozen=True)
class PositiveSetEndpointFeatureState:
    """Outer-train-only PCA and feature standardization state."""

    h_center: torch.Tensor
    h_components: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor

    def __post_init__(self) -> None:
        if tuple(self.h_center.shape) != (H_TOKEN_DIM,):
            raise ValueError("H center must have shape [200]")
        if tuple(self.h_components.shape) != (H_PCA_DIM, H_TOKEN_DIM):
            raise ValueError("H PCA components must have shape [8,200]")
        if tuple(self.feature_mean.shape) != (POSITIVE_SET_ENDPOINT_FEATURE_DIM,) or (
            tuple(self.feature_scale.shape) != (POSITIVE_SET_ENDPOINT_FEATURE_DIM,)
        ):
            raise ValueError("feature state must have shape [14]")
        tensors = (
            self.h_center,
            self.h_components,
            self.feature_mean,
            self.feature_scale,
        )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("feature-state tensors must share a device")
        for name, value in zip(
            ("H center", "H components", "feature mean", "feature scale"),
            tensors,
        ):
            _require_finite_detached(value, name=name)
        if bool((self.feature_scale <= 0).any()):
            raise ValueError("feature scale must be positive")


@dataclass(frozen=True)
class PositiveSetEndpointFeatureBatch:
    """Patient/node H8+V6 features with no I, Q, or channel-ID column."""

    values: torch.Tensor
    node_mask: torch.Tensor
    event_counts: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or tuple(self.values.shape[1:]) != (
            N_STANDARD_CHANNELS,
            POSITIVE_SET_ENDPOINT_FEATURE_DIM,
        ):
            raise ValueError("endpoint features must have shape [P,19,14]")
        patients = int(self.values.shape[0])
        expected = (patients, N_STANDARD_CHANNELS)
        if tuple(self.node_mask.shape) != expected or self.node_mask.dtype != torch.bool:
            raise TypeError("endpoint feature mask must be bool [P,19]")
        if tuple(self.event_counts.shape) != expected or self.event_counts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("endpoint event counts must be integer [P,19]")
        tensors = (self.values, self.node_mask, self.event_counts)
        if len({value.device for value in tensors}) != 1:
            raise ValueError("endpoint feature tensors must share a device")
        _require_finite_detached(self.values, name="endpoint features")
        if not torch.equal(self.node_mask, self.event_counts > 0):
            raise ValueError("feature mask disagrees with event counts")
        if bool((self.values[~self.node_mask] != 0).any()):
            raise ValueError("unavailable patient/node features must be exactly zero")


def _validate_train_indices(
    train_patient_indices: Sequence[int], n_patients: int
) -> tuple[int, ...]:
    selected = tuple(int(value) for value in train_patient_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("train_patient_indices must be non-empty and unique")
    if any(value < 0 or value >= n_patients for value in selected):
        raise IndexError("a train patient index is outside the patient carrier")
    return selected


def _patient_raw_contrasts(
    contrasts: EarlyPreEndpointContrasts,
    event_patient_index: torch.Tensor,
    n_patients: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(contrasts, EarlyPreEndpointContrasts):
        raise TypeError("contrasts must be EarlyPreEndpointContrasts")
    h, h_mask, counts = aggregate_patient_event_mean(
        contrasts.h,
        contrasts.node_mask,
        event_patient_index,
        n_patients,
    )
    v, v_mask, v_counts = aggregate_patient_event_mean(
        contrasts.v,
        contrasts.node_mask,
        event_patient_index,
        n_patients,
    )
    if not torch.equal(h_mask, v_mask) or not torch.equal(counts, v_counts):
        raise RuntimeError("H/V patient aggregation masks diverged")
    return h, v, h_mask, counts


def _fit_h_pca(
    patient_h: torch.Tensor,
    patient_mask: torch.Tensor,
    train_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = patient_h.index_select(0, train_index)
    mask = patient_mask.index_select(0, train_index)
    matrix = rows[mask]
    if matrix.shape[0] <= H_PCA_DIM:
        raise ValueError("outer-train carrier has too few valid nodes for PCA8")
    center = matrix.double().mean(dim=0)
    centered = matrix.double() - center
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:H_PCA_DIM].clone()
    pivot = components.abs().argmax(dim=1)
    signs = torch.sign(components[torch.arange(H_PCA_DIM), pivot])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    components = components * signs.unsqueeze(1)
    return center.float().contiguous(), components.float().contiguous()


def _raw_h8_v6_features(
    patient_h: torch.Tensor,
    patient_v: torch.Tensor,
    patient_mask: torch.Tensor,
    h_center: torch.Tensor,
    h_components: torch.Tensor,
) -> torch.Tensor:
    h8 = torch.matmul(
        patient_h - h_center.view(1, 1, -1), h_components.transpose(0, 1)
    )
    raw = torch.cat((h8, patient_v), dim=-1)
    if tuple(raw.shape[1:]) != (
        N_STANDARD_CHANNELS,
        POSITIVE_SET_ENDPOINT_FEATURE_DIM,
    ):
        raise RuntimeError("H8+V6 endpoint feature shape drifted")
    return torch.where(patient_mask.unsqueeze(-1), raw, torch.zeros_like(raw))


def _center_valid_nodes(
    values: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Remove each patient's feature-wise mean over available electrodes."""

    if values.ndim != 3 or tuple(values.shape[:2]) != tuple(node_mask.shape):
        raise ValueError("node centering requires [P,19,D] values and [P,19] mask")
    if node_mask.dtype != torch.bool:
        raise TypeError("node centering mask must be bool")
    count = node_mask.sum(dim=1, keepdim=True)
    safe = torch.where(node_mask.unsqueeze(-1), values, torch.zeros_like(values))
    center = safe.sum(dim=1, keepdim=True) / count.clamp_min(1).to(
        values.dtype
    ).unsqueeze(-1)
    centered = values - center
    return torch.where(
        node_mask.unsqueeze(-1), centered, torch.zeros_like(centered)
    ).contiguous()


def fit_fold_positive_set_endpoint_features(
    contrasts: EarlyPreEndpointContrasts,
    event_patient_index: torch.Tensor,
    n_patients: int,
    train_patient_indices: Sequence[int],
) -> tuple[PositiveSetEndpointFeatureBatch, PositiveSetEndpointFeatureState]:
    """Fit PCA8/scaling on outer-train patients and transform all patients."""

    selected = _validate_train_indices(train_patient_indices, n_patients)
    patient_h, patient_v, patient_mask, counts = _patient_raw_contrasts(
        contrasts, event_patient_index, n_patients
    )
    train_index = torch.tensor(
        selected, dtype=torch.long, device=patient_h.device
    )
    h_center, h_components = _fit_h_pca(patient_h, patient_mask, train_index)
    raw = _raw_h8_v6_features(
        patient_h, patient_v, patient_mask, h_center, h_components
    )
    train_values = raw.index_select(0, train_index)
    train_mask = patient_mask.index_select(0, train_index)
    matrix = train_values[train_mask]
    if matrix.shape[0] < 2:
        raise ValueError("outer-train carrier has too few rows for standardization")
    feature_mean = matrix.mean(dim=0)
    feature_scale = matrix.std(dim=0, unbiased=False)
    feature_scale = torch.where(
        feature_scale > 1e-6, feature_scale, torch.ones_like(feature_scale)
    )
    values = (raw - feature_mean.view(1, 1, -1)) / feature_scale.view(1, 1, -1)
    values = _center_valid_nodes(values, patient_mask)
    state = PositiveSetEndpointFeatureState(
        h_center=h_center.detach().contiguous(),
        h_components=h_components.detach().contiguous(),
        feature_mean=feature_mean.detach().contiguous(),
        feature_scale=feature_scale.detach().contiguous(),
    )
    batch = PositiveSetEndpointFeatureBatch(
        values=values.detach().float().contiguous(),
        node_mask=patient_mask.contiguous(),
        event_counts=counts.contiguous(),
    )
    return batch, state


def transform_positive_set_endpoint_features(
    contrasts: EarlyPreEndpointContrasts,
    event_patient_index: torch.Tensor,
    n_patients: int,
    state: PositiveSetEndpointFeatureState,
) -> PositiveSetEndpointFeatureBatch:
    """Transform a held cohort without estimating any held-cohort statistic."""

    if not isinstance(state, PositiveSetEndpointFeatureState):
        raise TypeError("state must be PositiveSetEndpointFeatureState")
    patient_h, patient_v, patient_mask, counts = _patient_raw_contrasts(
        contrasts, event_patient_index, n_patients
    )
    if state.h_center.device != patient_h.device:
        raise ValueError("feature state and held contrasts must share a device")
    raw = _raw_h8_v6_features(
        patient_h,
        patient_v,
        patient_mask,
        state.h_center,
        state.h_components,
    )
    values = (raw - state.feature_mean.view(1, 1, -1)) / state.feature_scale.view(
        1, 1, -1
    )
    values = _center_valid_nodes(values, patient_mask)
    return PositiveSetEndpointFeatureBatch(
        values=values.detach().float().contiguous(),
        node_mask=patient_mask.contiguous(),
        event_counts=counts.contiguous(),
    )


class PositiveSetEndpointResidual(nn.Module):
    """One shared 14-parameter bias-free physical-endpoint utility."""

    def __init__(self, input_dim: int = POSITIVE_SET_ENDPOINT_FEATURE_DIM) -> None:
        super().__init__()
        if input_dim != POSITIVE_SET_ENDPOINT_FEATURE_DIM:
            raise ValueError("positive-set endpoint input_dim is frozen at 14")
        self.input_dim = POSITIVE_SET_ENDPOINT_FEATURE_DIM
        self.endpoint_utility = nn.Linear(self.input_dim, 1, bias=False)
        with torch.no_grad():
            self.endpoint_utility.weight.zero_()

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def score_nodes(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.input_dim:
            raise ValueError("node features must end in the frozen dimension 14")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("node features must be finite floating point")
        return self.endpoint_utility(features).squeeze(-1)

    def forward(self, endpoint_features: torch.Tensor) -> torch.Tensor:
        if endpoint_features.ndim != 3 or tuple(endpoint_features.shape[1:]) != (
            2,
            self.input_dim,
        ):
            raise ValueError("endpoint pairs must have shape [Q,2,14]")
        if not endpoint_features.is_floating_point() or not torch.isfinite(
            endpoint_features
        ).all():
            raise ValueError("endpoint pairs must be finite floating point")
        difference = endpoint_features[:, 1] - endpoint_features[:, 0]
        return F.linear(
            difference, self.endpoint_utility.weight, bias=None
        ).squeeze(-1)


def _validate_targets(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    patients: int | None = None,
) -> int:
    if targets.ndim != 2 or targets.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("targets must have shape [P,19]")
    inferred = int(targets.shape[0])
    if patients is not None and inferred != patients:
        raise ValueError("targets do not align with the patient carrier")
    if tuple(target_mask.shape) != tuple(targets.shape) or target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be bool [P,19]")
    if not targets.is_floating_point():
        raise TypeError("targets must be floating point")
    if targets.device != target_mask.device:
        raise ValueError("targets and target_mask must share a device")
    observed = targets[target_mask]
    if not torch.isfinite(observed).all() or (
        observed.numel() and not torch.all((observed == 0) | (observed == 1))
    ):
        raise ValueError("observed targets must be finite binary values")
    return inferred


def positive_set_mass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Patient-mean exact positive-set mass, with no neighbour dilation."""

    if logits.ndim != 2 or logits.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("logits must have shape [P,19]")
    patients = _validate_targets(targets, target_mask, patients=logits.shape[0])
    if logits.device != targets.device:
        raise ValueError("logits and targets must share a device")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating point")
    rows: list[torch.Tensor] = []
    for patient in range(patients):
        observed = target_mask[patient]
        positive = observed & (targets[patient] == 1)
        if not bool(observed.any()) or not bool(positive.any()):
            raise ValueError("every patient requires an observed exact positive")
        rows.append(
            torch.logsumexp(logits[patient, observed], dim=0)
            - torch.logsumexp(logits[patient, positive], dim=0)
        )
    return torch.stack(rows).mean()


@dataclass(frozen=True)
class PositiveSetEndpointObjectiveOutput:
    total: torch.Tensor
    exact_set_mass: torch.Tensor
    l2_penalty: torch.Tensor


def positive_set_endpoint_objective(
    model: PositiveSetEndpointResidual,
    features: PositiveSetEndpointFeatureBatch,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> PositiveSetEndpointObjectiveOutput:
    """Exact positive-set mass plus the frozen ``0.05 * ||w||^2``."""

    if type(model) is not PositiveSetEndpointResidual:
        raise TypeError("model must be PositiveSetEndpointResidual")
    if not isinstance(features, PositiveSetEndpointFeatureBatch):
        raise TypeError("features must be PositiveSetEndpointFeatureBatch")
    patients = int(features.values.shape[0])
    _validate_targets(targets, target_mask, patients=patients)
    parameter = model.endpoint_utility.weight
    devices = {features.values.device, targets.device, parameter.device}
    if len(devices) != 1:
        raise ValueError("model, features, and targets must share a device")
    known_positive = target_mask & (targets == 1)
    unavailable_positive = known_positive & ~features.node_mask
    if bool(unavailable_positive.any()):
        raise ValueError("an observed exact positive lacks endpoint features")
    effective_mask = target_mask & features.node_mask
    logits = model.score_nodes(features.values)
    exact = positive_set_mass_loss(logits, targets, effective_mask)
    l2 = parameter.square().sum()
    return PositiveSetEndpointObjectiveOutput(
        total=exact + POSITIVE_SET_ENDPOINT_L2_WEIGHT * l2,
        exact_set_mass=exact,
        l2_penalty=l2,
    )


def positive_set_mask_sensitivity_mass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    kind: TargetMaskSensitivityKind,
) -> torch.Tensor:
    """Patient-equal leave-one-observation-to-unknown sensitivity loss.

    Jackknife rows are averaged *inside their owning patient* before patients
    are averaged.  A patient with many known labels therefore has exactly the
    same objective weight as a singleton-positive patient.  For known-positive
    censoring, a singleton-positive patient uses its unchanged full row.
    """

    if logits.ndim != 2 or logits.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("logits must have shape [P,19]")
    patients = _validate_targets(targets, target_mask, patients=logits.shape[0])
    if logits.device != targets.device:
        raise ValueError("logits and targets must share a device")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating point")
    if kind not in {
        "known_positive_to_unknown",
        "observed_zero_to_unknown",
    }:
        raise ValueError("unknown target-mask sensitivity kind")

    patient_rows: list[torch.Tensor] = []
    for patient in range(patients):
        observed = target_mask[patient]
        positive = observed & (targets[patient] == 1)
        if not bool(positive.any()):
            raise ValueError("every patient requires an observed exact positive")
        if kind == "known_positive_to_unknown":
            candidates = torch.nonzero(positive, as_tuple=False).flatten()
            if candidates.numel() == 1:
                variants = (observed,)
            else:
                rows: list[torch.Tensor] = []
                for channel in candidates.tolist():
                    variant = observed.clone()
                    variant[channel] = False
                    rows.append(variant)
                variants = tuple(rows)
        else:
            candidates = torch.nonzero(
                observed & (targets[patient] == 0), as_tuple=False
            ).flatten()
            if candidates.numel() == 0:
                variants = (observed,)
            else:
                rows = []
                for channel in candidates.tolist():
                    variant = observed.clone()
                    variant[channel] = False
                    rows.append(variant)
                variants = tuple(rows)

        variant_rows: list[torch.Tensor] = []
        for variant in variants:
            variant_positive = variant & (targets[patient] == 1)
            if not bool(variant_positive.any()):
                raise RuntimeError("sensitivity censoring removed the last positive")
            variant_rows.append(
                torch.logsumexp(logits[patient, variant], dim=0)
                - torch.logsumexp(logits[patient, variant_positive], dim=0)
            )
        patient_rows.append(torch.stack(variant_rows).mean())
    return torch.stack(patient_rows).mean()


def positive_set_endpoint_sensitivity_objective(
    model: PositiveSetEndpointResidual,
    features: PositiveSetEndpointFeatureBatch,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    kind: TargetMaskSensitivityKind,
) -> PositiveSetEndpointObjectiveOutput:
    """Patient-equal target-censoring mass plus frozen L2 regularization."""

    if type(model) is not PositiveSetEndpointResidual:
        raise TypeError("model must be PositiveSetEndpointResidual")
    if not isinstance(features, PositiveSetEndpointFeatureBatch):
        raise TypeError("features must be PositiveSetEndpointFeatureBatch")
    patients = int(features.values.shape[0])
    _validate_targets(targets, target_mask, patients=patients)
    parameter = model.endpoint_utility.weight
    devices = {features.values.device, targets.device, parameter.device}
    if len(devices) != 1:
        raise ValueError("model, features, and targets must share a device")
    known_positive = target_mask & (targets == 1)
    if bool((known_positive & ~features.node_mask).any()):
        raise ValueError("an observed exact positive lacks endpoint features")
    effective_mask = target_mask & features.node_mask
    logits = model.score_nodes(features.values)
    exact = positive_set_mask_sensitivity_mass_loss(
        logits,
        targets,
        effective_mask,
        kind=kind,
    )
    l2 = parameter.square().sum()
    return PositiveSetEndpointObjectiveOutput(
        total=exact + POSITIVE_SET_ENDPOINT_L2_WEIGHT * l2,
        exact_set_mass=exact,
        l2_penalty=l2,
    )


TargetMaskSensitivityKind = Literal[
    "known_positive_to_unknown",
    "observed_zero_to_unknown",
]


@dataclass(frozen=True)
class TargetMaskSensitivityBatch:
    """One-label-at-a-time censoring variants; no label is imputed."""

    masks: torch.Tensor
    patient_index: torch.Tensor
    channel_index: torch.Tensor
    kind: TargetMaskSensitivityKind

    def __post_init__(self) -> None:
        if self.masks.ndim != 3 or tuple(self.masks.shape[1:]) != (
            self.patient_count,
            N_STANDARD_CHANNELS,
        ) or self.masks.dtype != torch.bool:
            raise TypeError("sensitivity masks must be bool [J,P,19]")
        variants = int(self.masks.shape[0])
        if tuple(self.patient_index.shape) != (variants,) or (
            self.patient_index.dtype != torch.long
        ):
            raise TypeError("sensitivity patient_index must be long [J]")
        if tuple(self.channel_index.shape) != (variants,) or (
            self.channel_index.dtype != torch.long
        ):
            raise TypeError("sensitivity channel_index must be long [J]")
        if self.kind not in {
            "known_positive_to_unknown",
            "observed_zero_to_unknown",
        }:
            raise ValueError("unknown target-mask sensitivity kind")
        tensors = (self.masks, self.patient_index, self.channel_index)
        if len({value.device for value in tensors}) != 1:
            raise ValueError("sensitivity tensors must share a device")
        if variants and (
            bool((self.patient_index < 0).any())
            or bool((self.patient_index >= self.patient_count).any())
            or bool((self.channel_index < 0).any())
            or bool((self.channel_index >= N_STANDARD_CHANNELS).any())
        ):
            raise ValueError("sensitivity edit index is outside its carrier")

    @property
    def patient_count(self) -> int:
        return int(self.masks.shape[1]) if self.masks.ndim == 3 else 0

    @property
    def variant_count(self) -> int:
        return int(self.masks.shape[0])


def build_leave_one_target_unknown_masks(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    kind: TargetMaskSensitivityKind,
) -> TargetMaskSensitivityBatch:
    """Return deterministic one-observation-to-unknown sensitivity masks.

    A known positive is censored only when the same patient retains at least
    one other known positive.  This keeps every variant compatible with the
    positive-set objective.  Observed-zero variants make no claim that the
    censored complement is in fact positive.
    """

    patients = _validate_targets(targets, target_mask)
    if kind not in {
        "known_positive_to_unknown",
        "observed_zero_to_unknown",
    }:
        raise ValueError("unknown target-mask sensitivity kind")
    positive = target_mask & (targets == 1)
    if kind == "known_positive_to_unknown":
        eligible = positive & (positive.sum(dim=1, keepdim=True) > 1)
    else:
        eligible = target_mask & (targets == 0)
    coordinates = torch.nonzero(eligible, as_tuple=False)
    masks: list[torch.Tensor] = []
    for patient, channel in coordinates.detach().cpu().tolist():
        variant = target_mask.clone()
        variant[patient, channel] = False
        masks.append(variant)
    if masks:
        stacked = torch.stack(masks).contiguous()
    else:
        stacked = torch.empty(
            (0, patients, N_STANDARD_CHANNELS),
            dtype=torch.bool,
            device=target_mask.device,
        )
    return TargetMaskSensitivityBatch(
        masks=stacked,
        patient_index=coordinates[:, 0].to(dtype=torch.long).contiguous(),
        channel_index=coordinates[:, 1].to(dtype=torch.long).contiguous(),
        kind=kind,
    )


def mask_all_observed_zeros_unknown(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Extreme sensitivity mask retaining observed positives only.

    Under this mask, positive-set mass is identically zero.  The helper exists
    to expose that non-identifiability, not to create a trainable candidate.
    """

    _validate_targets(targets, target_mask)
    return (target_mask & (targets == 1)).contiguous()


__all__ = [
    "EARLY_TILE_INDICES",
    "H_FEATURE_SLICE",
    "H_PCA_DIM",
    "H_TOKEN_DIM",
    "POSITIVE_SET_ENDPOINT_FEATURE_DIM",
    "POSITIVE_SET_ENDPOINT_L2_WEIGHT",
    "POSITIVE_SET_ENDPOINT_RESIDUAL_SCHEMA",
    "PRE_TILE_INDICES",
    "V_FEATURE_DIM",
    "V_FEATURE_SLICE",
    "EarlyPreEndpointContrasts",
    "PositiveSetEndpointFeatureBatch",
    "PositiveSetEndpointFeatureState",
    "PositiveSetEndpointObjectiveOutput",
    "PositiveSetEndpointResidual",
    "SharedEarlyIGate",
    "TargetMaskSensitivityBatch",
    "TargetMaskSensitivityKind",
    "aggregate_patient_event_mean",
    "build_early_pre_endpoint_contrasts",
    "build_leave_one_target_unknown_masks",
    "build_shared_early_i_gate",
    "fit_fold_positive_set_endpoint_features",
    "mask_all_observed_zeros_unknown",
    "positive_set_endpoint_objective",
    "positive_set_endpoint_sensitivity_objective",
    "positive_set_mask_sensitivity_mass_loss",
    "positive_set_mass_loss",
    "transform_positive_set_endpoint_features",
]
