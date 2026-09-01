"""Physical-time, permission-locked sparse encoder for BA-IEG events.

The module is intentionally small enough to audit.  It consumes a
``BAIEGCollatedEventBatch`` and obtains its inputs exclusively through the
positive-onset branch.  Consequently, offline-context values and phase
posteriors are not merely discouraged: they have no computational path to an
onset logit.

Sparse message edges are constructed from each token's *actual physical
support* in recording-relative seconds and the signed analysis-unit reference
matrix.  Token row order is only a storage detail; padding, another event in
the batch, or a source whose support ends in the future cannot affect an
earlier target.  The learned edges are deliberately called associations, not
physiological propagation.

This is an implementation primitive, not a clinically qualified SOZ model.
Its outputs are token/event representations and research logits that require
patient-disjoint training, calibration, and term/spatial qualification before
they can enter a report.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Mapping

import torch
from torch import nn

from src.soz.geometry import CHANNEL_INDEX, STANDARD_19, TCP_20_EDGES

from .ba_ieg_training_contract import (
    BA_IEG_EVIDENCE_FAMILIES,
    BA_IEG_TOKEN_SCALES,
    BAIEGCollatedEventBatch,
)


BA_IEG_PHYSICAL_TIME_ENCODER_ID: Final[str] = (
    "ba_ieg_permission_locked_sparse_physical_graph_interval_encoder_v2"
)
BA_IEG_SPARSE_EDGE_RELATIONS: Final[tuple[str, ...]] = (
    "self",
    "same_unit_temporal_neighbor",
    "overlapping_physical_neighbor",
    "early_to_later_spatiotemporal_association",
)


@dataclass(frozen=True)
class BAIEGSparsePhysicalTimeGraph:
    """Padded sparse edge list over token rows.

    ``source_token_index -> target_token_index`` is always event-local and
    causal in actual support-end time.  The graph retains analysis-unit
    semantics: a bipolar row remains one derivation and is never expanded into
    either endpoint as an electrode target.
    """

    source_token_index: torch.Tensor
    target_token_index: torch.Tensor
    relation_index: torch.Tensor
    edge_mask: torch.Tensor

    @property
    def edge_count(self) -> torch.Tensor:
        return self.edge_mask.sum(dim=1)


@dataclass(frozen=True)
class BAIEGPhysicalTimeEncoderOutput:
    """Masked research outputs from the onset-causal branch."""

    source_input_batch_sha256: str
    token_embeddings: torch.Tensor
    token_onset_logits: torch.Tensor
    token_mask: torch.Tensor
    event_embedding: torch.Tensor
    event_onset_logit: torch.Tensor
    event_evaluable_mask: torch.Tensor
    analysis_unit_embeddings: torch.Tensor
    analysis_unit_onset_logits: torch.Tensor
    analysis_unit_onset_intervals_seconds: torch.Tensor
    analysis_unit_onset_association_rank: torch.Tensor
    analysis_unit_mask: torch.Tensor
    sparse_graph: BAIEGSparsePhysicalTimeGraph


class _PhysicalTimeFeatures(nn.Module):
    """Encode real support coordinates without a learned row-position ID."""

    def __init__(
        self,
        periods_seconds: tuple[float, ...] = (
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
            64.0,
            128.0,
            256.0,
        ),
    ) -> None:
        super().__init__()
        if not periods_seconds or any(
            not math.isfinite(value) or value <= 0.0
            for value in periods_seconds
        ):
            raise ValueError("physical-time periods must be finite and positive")
        self.register_buffer(
            "periods_seconds",
            torch.tensor(periods_seconds, dtype=torch.float32),
            persistent=True,
        )

    @property
    def output_dim(self) -> int:
        # Four non-periodic physical scalars plus midpoint and duration
        # sine/cosine pairs at every registered physical period.
        return 4 + 4 * int(self.periods_seconds.numel())

    def forward(
        self,
        bounds_seconds: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if bounds_seconds.ndim != 3 or bounds_seconds.shape[-1] != 2:
            raise ValueError("token physical bounds must have shape [B,T,2]")
        if active_mask.dtype != torch.bool or tuple(active_mask.shape) != tuple(
            bounds_seconds.shape[:2]
        ):
            raise ValueError("physical-time mask must align with token rows")

        start = bounds_seconds[..., 0]
        stop = bounds_seconds[..., 1]
        duration = (stop - start).clamp_min(0.0)
        midpoint = 0.5 * (start + stop)
        # asinh preserves sign and physical ordering while avoiding a large
        # raw-second scale for recordings with long recording-relative times.
        scalars = torch.stack(
            (
                torch.asinh(start / 60.0),
                torch.asinh(stop / 60.0),
                torch.asinh(midpoint / 60.0),
                torch.log1p(duration),
            ),
            dim=-1,
        )
        periods = self.periods_seconds.to(
            device=bounds_seconds.device,
            dtype=bounds_seconds.dtype,
        )
        midpoint_angle = 2.0 * math.pi * midpoint.unsqueeze(-1) / periods
        duration_angle = 2.0 * math.pi * duration.unsqueeze(-1) / periods
        encoded = torch.cat(
            (
                scalars,
                torch.sin(midpoint_angle),
                torch.cos(midpoint_angle),
                torch.sin(duration_angle),
                torch.cos(duration_angle),
            ),
            dim=-1,
        )
        return torch.where(
            active_mask.unsqueeze(-1), encoded, torch.zeros_like(encoded)
        )


_PHYSICAL_ELECTRODE_NEIGHBORS: Final[tuple[frozenset[int], ...]] = tuple(
    frozenset(
        {electrode_index}
        | {
            CHANNEL_INDEX[right] if CHANNEL_INDEX[left] == electrode_index else CHANNEL_INDEX[left]
            for left, right in TCP_20_EDGES
            if electrode_index in (CHANNEL_INDEX[left], CHANNEL_INDEX[right])
        }
    )
    for electrode_index in range(len(STANDARD_19))
)


def _dominant_reference_footprint(
    reference_row: torch.Tensor,
    physical_evidence_mask: torch.Tensor,
    *,
    relative_coefficient_threshold: float,
    maximum_local_footprint: int,
) -> frozenset[int]:
    """Return a local physical footprint without assigning an edge endpoint.

    A CAR electrode row normally has one dominant coefficient and many small
    common-reference coefficients; a bipolar row normally retains both equal
    endpoints.  A genuinely broad virtual row is deliberately excluded from
    local spatial edges rather than being allowed to connect the entire graph.
    """

    magnitude = reference_row.detach().abs().cpu()
    evidence = physical_evidence_mask.detach().cpu()
    maximum = float(magnitude.max()) if magnitude.numel() else 0.0
    if maximum <= 0.0:
        return frozenset()
    selected = frozenset(
        int(index)
        for index in torch.nonzero(
            (magnitude >= maximum * relative_coefficient_threshold) & evidence,
            as_tuple=False,
        ).flatten()
    )
    if not selected or len(selected) > maximum_local_footprint:
        return frozenset()
    return selected


def _footprints_are_physical_neighbors(
    first: frozenset[int], second: frozenset[int]
) -> bool:
    return bool(first and second) and any(
        right in _PHYSICAL_ELECTRODE_NEIGHBORS[left]
        for left in first
        for right in second
    )


def _nearest_time_group_members(
    candidates: list[tuple[int, float]],
    *,
    maximum_groups: int,
    tolerance_seconds: float,
) -> tuple[int, ...]:
    """Select nearest distinct physical-time groups with tie closure."""

    if not candidates:
        return ()
    distinct: list[float] = []
    for _, distance in sorted(candidates, key=lambda item: item[1]):
        if not distinct or abs(distance - distinct[-1]) > tolerance_seconds:
            distinct.append(distance)
    cutoff = distinct[min(maximum_groups, len(distinct)) - 1]
    return tuple(
        index
        for index, distance in candidates
        if distance <= cutoff + tolerance_seconds
    )


def _build_sparse_physical_time_graph_from_inputs(
    inputs: Mapping[str, torch.Tensor],
    *,
    temporal_neighbor_groups: int,
    overlapping_neighbor_groups: int,
    association_neighbor_groups: int,
    maximum_association_seconds: float,
    time_tolerance_seconds: float,
    relative_reference_threshold: float,
    maximum_local_footprint: int,
) -> BAIEGSparsePhysicalTimeGraph:
    """Build an event-local edge list without a token-by-token dense matrix."""

    active_mask = inputs["token_row_mask"] & inputs["token_signal_mask"]
    bounds = inputs["token_time_bounds_seconds"]
    token_units = inputs["token_unit_index"]
    unit_rows = inputs["unit_row_mask"]
    references = inputs["unit_reference_matrix"]
    physical_evidence = inputs["physical_evidence_mask"]
    if active_mask.ndim != 2 or bounds.shape[:2] != active_mask.shape:
        raise ValueError("sparse graph token coordinates are not aligned")

    batch_edges: list[list[tuple[int, int, int]]] = []
    for batch_index in range(int(active_mask.shape[0])):
        active_indices = tuple(
            int(index)
            for index in torch.nonzero(
                active_mask[batch_index], as_tuple=False
            ).flatten().detach().cpu()
        )
        local_bounds = bounds[batch_index].detach().cpu()
        local_units = token_units[batch_index].detach().cpu()
        tokens_by_unit: dict[int, list[int]] = {}
        for token_index in active_indices:
            unit_index = int(local_units[token_index])
            tokens_by_unit.setdefault(unit_index, []).append(token_index)

        footprints: dict[int, frozenset[int]] = {}
        for unit_index in tokens_by_unit:
            if unit_index < 0 or not bool(unit_rows[batch_index, unit_index]):
                continue
            footprints[unit_index] = _dominant_reference_footprint(
                references[batch_index, unit_index],
                physical_evidence[batch_index],
                relative_coefficient_threshold=relative_reference_threshold,
                maximum_local_footprint=maximum_local_footprint,
            )

        physical_neighbor_units: dict[int, tuple[int, ...]] = {}
        for target_unit, target_footprint in footprints.items():
            physical_neighbor_units[target_unit] = tuple(
                source_unit
                for source_unit, source_footprint in footprints.items()
                if source_unit != target_unit
                and _footprints_are_physical_neighbors(
                    source_footprint, target_footprint
                )
            )

        edges: set[tuple[int, int, int]] = set()
        self_relation = BA_IEG_SPARSE_EDGE_RELATIONS.index("self")
        temporal_relation = BA_IEG_SPARSE_EDGE_RELATIONS.index(
            "same_unit_temporal_neighbor"
        )
        overlap_relation = BA_IEG_SPARSE_EDGE_RELATIONS.index(
            "overlapping_physical_neighbor"
        )
        association_relation = BA_IEG_SPARSE_EDGE_RELATIONS.index(
            "early_to_later_spatiotemporal_association"
        )
        for token_index in active_indices:
            edges.add((token_index, token_index, self_relation))

        for unit_tokens in tokens_by_unit.values():
            for target in unit_tokens:
                target_end = float(local_bounds[target, 1])
                candidates = [
                    (source, max(0.0, target_end - float(local_bounds[source, 1])))
                    for source in unit_tokens
                    if source != target
                    and float(local_bounds[source, 1])
                    <= target_end + time_tolerance_seconds
                ]
                for source in _nearest_time_group_members(
                    candidates,
                    maximum_groups=temporal_neighbor_groups,
                    tolerance_seconds=time_tolerance_seconds,
                ):
                    edges.add((source, target, temporal_relation))

        for target_unit, target_tokens in tokens_by_unit.items():
            for source_unit in physical_neighbor_units.get(target_unit, ()):
                source_tokens = tokens_by_unit[source_unit]
                for target in target_tokens:
                    target_start = float(local_bounds[target, 0])
                    target_stop = float(local_bounds[target, 1])
                    overlapping: list[tuple[int, float]] = []
                    earlier: list[tuple[int, float]] = []
                    for source in source_tokens:
                        source_start = float(local_bounds[source, 0])
                        source_stop = float(local_bounds[source, 1])
                        if source_stop > target_stop + time_tolerance_seconds:
                            continue
                        overlap = min(source_stop, target_stop) - max(
                            source_start, target_start
                        )
                        if overlap > time_tolerance_seconds:
                            overlapping.append(
                                (source, max(0.0, target_stop - source_stop))
                            )
                            continue
                        gap = target_start - source_stop
                        if (
                            gap >= -time_tolerance_seconds
                            and gap <= maximum_association_seconds + time_tolerance_seconds
                        ):
                            earlier.append((source, max(0.0, gap)))
                    for source in _nearest_time_group_members(
                        overlapping,
                        maximum_groups=overlapping_neighbor_groups,
                        tolerance_seconds=time_tolerance_seconds,
                    ):
                        edges.add((source, target, overlap_relation))
                    for source in _nearest_time_group_members(
                        earlier,
                        maximum_groups=association_neighbor_groups,
                        tolerance_seconds=time_tolerance_seconds,
                    ):
                        edges.add((source, target, association_relation))

        # Sorting is solely for deterministic serialization.  Message
        # aggregation is a commutative index-add and never uses this order as
        # a learned position.
        batch_edges.append(
            sorted(edges, key=lambda edge: (edge[1], edge[0], edge[2]))
        )

    maximum_edges = max(1, *(len(edges) for edges in batch_edges))
    device = bounds.device
    source = torch.full(
        (len(batch_edges), maximum_edges), -1, dtype=torch.long, device=device
    )
    target = torch.full_like(source, -1)
    relation = torch.full_like(source, -1)
    edge_mask = torch.zeros_like(source, dtype=torch.bool)
    for batch_index, edges in enumerate(batch_edges):
        if not edges:
            continue
        edge_tensor = torch.tensor(edges, dtype=torch.long, device=device)
        count = int(edge_tensor.shape[0])
        source[batch_index, :count] = edge_tensor[:, 0]
        target[batch_index, :count] = edge_tensor[:, 1]
        relation[batch_index, :count] = edge_tensor[:, 2]
        edge_mask[batch_index, :count] = True
    return BAIEGSparsePhysicalTimeGraph(
        source_token_index=source,
        target_token_index=target,
        relation_index=relation,
        edge_mask=edge_mask,
    )


def build_ba_ieg_sparse_physical_time_graph(
    batch: BAIEGCollatedEventBatch,
    *,
    temporal_neighbor_groups: int = 2,
    overlapping_neighbor_groups: int = 2,
    association_neighbor_groups: int = 2,
    maximum_association_seconds: float = 12.0,
    time_tolerance_seconds: float = 1e-6,
    relative_reference_threshold: float = 0.2,
    maximum_local_footprint: int = 6,
) -> BAIEGSparsePhysicalTimeGraph:
    """Build the registered onset graph, excluding every offline token."""

    if not isinstance(batch, BAIEGCollatedEventBatch):
        raise TypeError("sparse graph construction requires a registered batch")
    if min(
        temporal_neighbor_groups,
        overlapping_neighbor_groups,
        association_neighbor_groups,
        maximum_local_footprint,
    ) <= 0:
        raise ValueError("sparse graph neighbor/footprint limits must be positive")
    if (
        not math.isfinite(maximum_association_seconds)
        or maximum_association_seconds < 0.0
        or not math.isfinite(time_tolerance_seconds)
        or time_tolerance_seconds < 0.0
        or not math.isfinite(relative_reference_threshold)
        or relative_reference_threshold <= 0.0
        or relative_reference_threshold > 1.0
    ):
        raise ValueError("sparse graph physical thresholds are invalid")
    return _build_sparse_physical_time_graph_from_inputs(
        batch.positive_onset_inputs(),
        temporal_neighbor_groups=temporal_neighbor_groups,
        overlapping_neighbor_groups=overlapping_neighbor_groups,
        association_neighbor_groups=association_neighbor_groups,
        maximum_association_seconds=maximum_association_seconds,
        time_tolerance_seconds=time_tolerance_seconds,
        relative_reference_threshold=relative_reference_threshold,
        maximum_local_footprint=maximum_local_footprint,
    )


class _SparsePhysicalAssociationBlock(nn.Module):
    """Sparse causal message passing over predeclared physical associations."""

    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.hidden_dim = int(hidden_dim)
        self.pre_message_norm = nn.LayerNorm(hidden_dim)
        self.source_projection = nn.Linear(hidden_dim, hidden_dim)
        self.relation_embedding = nn.Embedding(
            len(BA_IEG_SPARSE_EDGE_RELATIONS), hidden_dim
        )
        self.edge_feature_projection = nn.Linear(4, hidden_dim, bias=False)
        self.message_gate = nn.Linear(2 * hidden_dim, 1)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        *,
        graph: BAIEGSparsePhysicalTimeGraph,
        bounds_seconds: torch.Tensor,
        token_unit_index: torch.Tensor,
        unit_reference_matrix: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, hidden_dim = token_embeddings.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError("sparse block hidden dimension drifted")
        normalized = self.pre_message_norm(token_embeddings)
        aggregated = torch.zeros_like(token_embeddings)

        for batch_index in range(batch_size):
            valid_edges = graph.edge_mask[batch_index]
            if not bool(valid_edges.any()):
                continue
            source = graph.source_token_index[batch_index, valid_edges]
            target = graph.target_token_index[batch_index, valid_edges]
            relation = graph.relation_index[batch_index, valid_edges]
            if (
                torch.any(source < 0)
                or torch.any(source >= token_count)
                or torch.any(target < 0)
                or torch.any(target >= token_count)
                or torch.any(~active_mask[batch_index, source])
                or torch.any(~active_mask[batch_index, target])
            ):
                raise RuntimeError("sparse graph references a forbidden token row")

            source_bounds = bounds_seconds[batch_index, source]
            target_bounds = bounds_seconds[batch_index, target]
            delta_end = (target_bounds[:, 1] - source_bounds[:, 1]).clamp_min(0.0)
            gap = (target_bounds[:, 0] - source_bounds[:, 1]).clamp_min(0.0)
            overlap = (
                torch.minimum(target_bounds[:, 1], source_bounds[:, 1])
                - torch.maximum(target_bounds[:, 0], source_bounds[:, 0])
            ).clamp_min(0.0)
            source_units = token_unit_index[batch_index, source]
            target_units = token_unit_index[batch_index, target]
            source_reference = unit_reference_matrix[
                batch_index, source_units
            ].abs()
            target_reference = unit_reference_matrix[
                batch_index, target_units
            ].abs()
            reference_cosine = (
                (source_reference * target_reference).sum(dim=-1)
                / (
                    source_reference.norm(dim=-1)
                    * target_reference.norm(dim=-1)
                ).clamp_min(1e-12)
            )
            edge_features = torch.stack(
                (
                    torch.asinh(delta_end),
                    torch.log1p(gap),
                    torch.log1p(overlap),
                    reference_cosine,
                ),
                dim=-1,
            ).to(dtype=token_embeddings.dtype)
            source_message = (
                self.source_projection(normalized[batch_index, source])
                + self.relation_embedding(relation)
                + self.edge_feature_projection(edge_features)
            )
            gates = torch.sigmoid(
                self.message_gate(
                    torch.cat(
                        (normalized[batch_index, target], source_message), dim=-1
                    )
                )
            )
            weighted_message = self.dropout(source_message) * gates
            local_sum = torch.zeros_like(token_embeddings[batch_index])
            local_weight = torch.zeros(
                (token_count, 1),
                dtype=token_embeddings.dtype,
                device=token_embeddings.device,
            )
            local_sum.index_add_(0, target, weighted_message)
            local_weight.index_add_(0, target, gates)
            aggregated[batch_index] = local_sum / local_weight.clamp_min(1e-12)

        active = active_mask.unsqueeze(-1)
        token_embeddings = torch.where(
            active,
            token_embeddings + self.output_projection(aggregated),
            torch.zeros_like(token_embeddings),
        )
        token_embeddings = torch.where(
            active,
            token_embeddings + self.mlp(self.pre_mlp_norm(token_embeddings)),
            torch.zeros_like(token_embeddings),
        )
        return token_embeddings


class BAIEGPhysicalTimeOnsetEncoder(nn.Module):
    """Trainable local encoder with a fail-closed onset evidence route.

    The model never accepts an arbitrary shared tensor dictionary.  Calling
    the permission-aware branch on the registered batch here prevents a
    trainer from accidentally adding ``phase_posterior`` or offline values to
    the positive onset path.
    """

    implementation_id: Final[str] = BA_IEG_PHYSICAL_TIME_ENCODER_ID

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.0,
        time_tolerance_seconds: float = 1e-6,
        temporal_neighbor_groups: int = 2,
        overlapping_neighbor_groups: int = 2,
        association_neighbor_groups: int = 2,
        maximum_association_seconds: float = 12.0,
        relative_reference_threshold: float = 0.2,
        maximum_local_footprint: int = 6,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or depth <= 0:
            raise ValueError("feature_dim, hidden_dim and depth must be positive")
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if min(
            temporal_neighbor_groups,
            overlapping_neighbor_groups,
            association_neighbor_groups,
            maximum_local_footprint,
        ) <= 0:
            raise ValueError("sparse graph neighbor/footprint limits must be positive")
        if (
            not math.isfinite(maximum_association_seconds)
            or maximum_association_seconds < 0.0
            or not math.isfinite(time_tolerance_seconds)
            or time_tolerance_seconds < 0.0
            or not math.isfinite(relative_reference_threshold)
            or relative_reference_threshold <= 0.0
            or relative_reference_threshold > 1.0
        ):
            raise ValueError("sparse graph physical thresholds are invalid")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        # ``num_heads`` remains an explicit capacity-compatibility parameter
        # for experiment manifests; v2 uses sparse relation messages rather
        # than constructing a dense multi-head token attention matrix.
        self.num_heads = int(num_heads)
        self.time_tolerance_seconds = float(time_tolerance_seconds)
        self.temporal_neighbor_groups = int(temporal_neighbor_groups)
        self.overlapping_neighbor_groups = int(overlapping_neighbor_groups)
        self.association_neighbor_groups = int(association_neighbor_groups)
        self.maximum_association_seconds = float(maximum_association_seconds)
        self.relative_reference_threshold = float(relative_reference_threshold)
        self.maximum_local_footprint = int(maximum_local_footprint)
        self.time_features = _PhysicalTimeFeatures()
        self.value_projection = nn.Linear(2 * feature_dim, hidden_dim)
        self.time_projection = nn.Linear(
            self.time_features.output_dim, hidden_dim, bias=False
        )
        self.scale_embedding = nn.Embedding(len(BA_IEG_TOKEN_SCALES), hidden_dim)
        self.reference_projection = nn.Linear(
            len(STANDARD_19), hidden_dim, bias=False
        )
        self.family_projection = nn.Linear(
            len(BA_IEG_EVIDENCE_FAMILIES), hidden_dim, bias=False
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            _SparsePhysicalAssociationBlock(hidden_dim, dropout=dropout)
            for _ in range(depth)
        )
        self.token_head = nn.Linear(hidden_dim, 1)
        self.analysis_unit_gate = nn.Linear(hidden_dim, 1)
        self.analysis_unit_head = nn.Linear(hidden_dim, 1)
        self.event_gate = nn.Linear(hidden_dim, 1)
        self.event_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, batch: BAIEGCollatedEventBatch
    ) -> BAIEGPhysicalTimeEncoderOutput:
        if not isinstance(batch, BAIEGCollatedEventBatch):
            raise TypeError("BA-IEG onset encoder requires a registered collated batch")
        inputs = batch.positive_onset_inputs()
        if "phase_posterior" in inputs:  # pragma: no cover - contract guard
            raise RuntimeError("offline phase posterior reached the onset encoder")

        values = inputs["token_values"]
        feature_mask = inputs["token_feature_mask"]
        active_mask = inputs["token_row_mask"] & inputs["token_signal_mask"]
        if values.shape[-1] != self.feature_dim:
            raise ValueError("input token feature dimension does not match the encoder")
        if torch.any(inputs["token_future_sample_access"] & active_mask):
            raise RuntimeError("future-dependent token reached the onset encoder")
        if torch.any(~inputs["token_positive_onset_mask"] & active_mask):
            raise RuntimeError("onset encoder received a non-positive-eligible token")

        active = active_mask.unsqueeze(-1)
        feature_mask = feature_mask & active
        masked_values = torch.where(
            feature_mask, values, torch.zeros_like(values)
        )
        value_input = torch.cat(
            (masked_values, feature_mask.to(dtype=values.dtype)), dim=-1
        )

        bounds = inputs["token_time_bounds_seconds"]
        time_encoding = self.time_features(bounds, active_mask)
        scale_index = inputs["token_scale_index"].clamp(
            min=0, max=len(BA_IEG_TOKEN_SCALES) - 1
        )

        unit_index = inputs["token_unit_index"]
        safe_unit_index = unit_index.clamp_min(0)
        references = torch.gather(
            inputs["unit_reference_matrix"],
            dim=1,
            index=safe_unit_index.unsqueeze(-1).expand(
                -1, -1, len(STANDARD_19)
            ),
        )
        references = torch.where(active, references, torch.zeros_like(references))
        families = torch.where(
            active,
            inputs["token_family_mask"].to(dtype=values.dtype),
            torch.zeros_like(inputs["token_family_mask"], dtype=values.dtype),
        )

        embeddings = (
            self.value_projection(value_input)
            + self.time_projection(time_encoding)
            + self.scale_embedding(scale_index)
            # The onset-ranking branch uses the reference *support* and is
            # invariant to an arbitrary global sign reversal of one bipolar
            # derivation.  The original signed matrix remains available for a
            # separate polarity/field head; no endpoint attribution occurs.
            + self.reference_projection(references.abs())
            + self.family_projection(families)
        )
        embeddings = torch.where(
            active, self.input_norm(embeddings), torch.zeros_like(embeddings)
        )
        sparse_graph = _build_sparse_physical_time_graph_from_inputs(
            inputs,
            temporal_neighbor_groups=self.temporal_neighbor_groups,
            overlapping_neighbor_groups=self.overlapping_neighbor_groups,
            association_neighbor_groups=self.association_neighbor_groups,
            maximum_association_seconds=self.maximum_association_seconds,
            time_tolerance_seconds=self.time_tolerance_seconds,
            relative_reference_threshold=self.relative_reference_threshold,
            maximum_local_footprint=self.maximum_local_footprint,
        )
        for block in self.blocks:
            embeddings = block(
                embeddings,
                graph=sparse_graph,
                bounds_seconds=bounds,
                token_unit_index=unit_index,
                unit_reference_matrix=inputs["unit_reference_matrix"],
                active_mask=active_mask,
            )

        token_logits = self.token_head(embeddings).squeeze(-1)
        token_logits = torch.where(
            active_mask, token_logits, torch.zeros_like(token_logits)
        )

        batch_size = int(embeddings.shape[0])
        maximum_units = int(inputs["unit_row_mask"].shape[1])
        analysis_unit_embeddings = torch.zeros(
            (batch_size, maximum_units, self.hidden_dim),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        analysis_unit_intervals = torch.zeros(
            (batch_size, maximum_units, 2),
            dtype=bounds.dtype,
            device=bounds.device,
        )
        analysis_unit_mask = torch.zeros(
            (batch_size, maximum_units),
            dtype=torch.bool,
            device=embeddings.device,
        )
        for batch_index in range(batch_size):
            for analysis_unit_index in range(maximum_units):
                selected = (
                    active_mask[batch_index]
                    & (unit_index[batch_index] == analysis_unit_index)
                    & inputs["unit_row_mask"][batch_index, analysis_unit_index]
                )
                if not bool(selected.any()):
                    continue
                selected_embeddings = embeddings[batch_index, selected]
                unit_gate_logits = self.analysis_unit_gate(
                    selected_embeddings
                ).squeeze(-1)
                unit_gate_weights = torch.softmax(unit_gate_logits, dim=0)
                analysis_unit_embeddings[
                    batch_index, analysis_unit_index
                ] = torch.sum(
                    selected_embeddings * unit_gate_weights.unsqueeze(-1), dim=0
                )
                # A soft selection over actual token supports is an auditable
                # research onset interval head.  It is not a calibrated
                # clinical confidence interval until patient-disjoint
                # coverage-width qualification is supplied downstream.
                interval_weights = torch.softmax(
                    token_logits[batch_index, selected], dim=0
                )
                analysis_unit_intervals[
                    batch_index, analysis_unit_index
                ] = torch.sum(
                    bounds[batch_index, selected]
                    * interval_weights.unsqueeze(-1),
                    dim=0,
                )
                analysis_unit_mask[batch_index, analysis_unit_index] = True

        analysis_unit_logits = self.analysis_unit_head(
            analysis_unit_embeddings
        ).squeeze(-1)
        analysis_unit_logits = torch.where(
            analysis_unit_mask,
            analysis_unit_logits,
            torch.zeros_like(analysis_unit_logits),
        )
        analysis_unit_rank = torch.zeros(
            (batch_size, maximum_units),
            dtype=torch.long,
            device=embeddings.device,
        )
        for batch_index in range(batch_size):
            evaluable_units = torch.nonzero(
                analysis_unit_mask[batch_index], as_tuple=False
            ).flatten()
            if not bool(evaluable_units.numel()):
                continue
            ordered_units = evaluable_units[
                torch.argsort(
                    analysis_unit_logits[batch_index, evaluable_units],
                    descending=True,
                    stable=True,
                )
            ]
            analysis_unit_rank[batch_index, ordered_units] = torch.arange(
                1,
                int(ordered_units.numel()) + 1,
                dtype=torch.long,
                device=embeddings.device,
            )

        gate_logits = self.event_gate(analysis_unit_embeddings).squeeze(-1)
        gate_logits = gate_logits.masked_fill(
            ~analysis_unit_mask, torch.finfo(gate_logits.dtype).min
        )
        gate_weights = torch.softmax(gate_logits, dim=-1)
        gate_weights = torch.where(
            analysis_unit_mask, gate_weights, torch.zeros_like(gate_weights)
        )
        gate_weights = gate_weights / gate_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        event_embedding = torch.sum(
            analysis_unit_embeddings * gate_weights.unsqueeze(-1), dim=1
        )
        event_evaluable = analysis_unit_mask.any(dim=1)
        event_embedding = torch.where(
            event_evaluable.unsqueeze(-1),
            event_embedding,
            torch.zeros_like(event_embedding),
        )
        event_logit = self.event_head(event_embedding).squeeze(-1)
        event_logit = torch.where(
            event_evaluable, event_logit, torch.zeros_like(event_logit)
        )
        return BAIEGPhysicalTimeEncoderOutput(
            source_input_batch_sha256=batch.input_batch_sha256,
            token_embeddings=embeddings,
            token_onset_logits=token_logits,
            token_mask=active_mask,
            event_embedding=event_embedding,
            event_onset_logit=event_logit,
            event_evaluable_mask=event_evaluable,
            analysis_unit_embeddings=analysis_unit_embeddings,
            analysis_unit_onset_logits=analysis_unit_logits,
            analysis_unit_onset_intervals_seconds=analysis_unit_intervals,
            analysis_unit_onset_association_rank=analysis_unit_rank,
            analysis_unit_mask=analysis_unit_mask,
            sparse_graph=sparse_graph,
        )


__all__ = [
    "BA_IEG_PHYSICAL_TIME_ENCODER_ID",
    "BA_IEG_SPARSE_EDGE_RELATIONS",
    "BAIEGSparsePhysicalTimeGraph",
    "BAIEGPhysicalTimeEncoderOutput",
    "BAIEGPhysicalTimeOnsetEncoder",
    "build_ba_ieg_sparse_physical_time_graph",
]
