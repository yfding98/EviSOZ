"""Low-capacity V-directed routing of bipolar ictal evidence.

The module keeps the frozen LaBraM I/V evidence interface used by the
temporal-MIL recovery head.  It changes only the edge-to-physical-electrode
assignment: a shared physical-node V scorer directs each TCP edge between its
two endpoints.  The learned routing probability is discriminative evidence,
not an onset, propagation, or causal-source probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .development_reasoner import DevelopmentIVEvidenceBatch
from .geometry import N_STANDARD_CHANNELS, edge_endpoint_indices
from .temporal_mil_recovery import (
    TemporalMILEvidenceReasoner,
    _inverse_softplus,
    _masked_softmax,
    _positive_only_reliability_gate,
)


V_DIRECTED_ENDPOINT_RECOVERY_SCHEMA = (
    "soz_labram_v_directed_endpoint_temporal_mil_recovery_v5"
)


def masked_binary_endpoint_softmax(
    logits: torch.Tensor,
    endpoint_valid: torch.Tensor,
) -> torch.Tensor:
    """Return endpoint probabilities with a symmetric no-V fallback.

    Both valid endpoints use an ordinary binary softmax.  One valid endpoint
    receives probability one.  If neither endpoint has V evidence, the rule
    falls back to ``(0.5, 0.5)`` rather than manufacturing a direction.
    """

    if logits.ndim != 4 or logits.shape[-1] != 2:
        raise ValueError("endpoint logits must have shape [B,E,T,2]")
    if tuple(endpoint_valid.shape) != tuple(logits.shape) or (
        endpoint_valid.dtype != torch.bool
    ):
        raise TypeError("endpoint_valid must be bool with shape [B,E,T,2]")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("endpoint logits must be finite floating point")

    available = endpoint_valid.any(dim=-1, keepdim=True)
    masked = logits.masked_fill(~endpoint_valid, -torch.inf)
    safe = torch.where(available, masked, torch.zeros_like(masked))
    probability = torch.softmax(safe, dim=-1)
    probability = torch.where(
        available,
        probability,
        torch.full_like(probability, 0.5),
    )
    if not torch.allclose(
        probability.sum(dim=-1),
        torch.ones_like(probability[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("endpoint probabilities do not sum to one")
    return probability


def route_edge_support_to_nodes(
    edge_support: torch.Tensor,
    edge_valid: torch.Tensor,
    endpoint_probability: torch.Tensor,
    endpoint_indices: torch.Tensor,
    *,
    n_nodes: int = N_STANDARD_CHANNELS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route edge evidence with ``2*pi`` and unweighted valid-edge degree.

    Keeping the denominator unweighted is essential: weighting numerator and
    denominator by the same endpoint probability would cancel the direction
    on a node with one incident edge.  At ``pi=(0.5,0.5)`` this function
    exactly recovers unsigned-incidence averaging.
    """

    if edge_support.ndim != 3:
        raise ValueError("edge_support must have shape [B,E,T]")
    if tuple(edge_valid.shape) != tuple(edge_support.shape) or (
        edge_valid.dtype != torch.bool
    ):
        raise TypeError("edge_valid must be bool with shape [B,E,T]")
    expected_probability = (*edge_support.shape, 2)
    if tuple(endpoint_probability.shape) != expected_probability:
        raise ValueError("endpoint_probability must have shape [B,E,T,2]")
    if tuple(endpoint_indices.shape) != (edge_support.shape[1], 2) or (
        endpoint_indices.dtype != torch.long
    ):
        raise TypeError("endpoint_indices must be long with shape [E,2]")
    if edge_support.device != edge_valid.device or (
        edge_support.device != endpoint_probability.device
        or edge_support.device != endpoint_indices.device
    ):
        raise ValueError("routing tensors must share a device")
    if not edge_support.is_floating_point() or not torch.isfinite(edge_support).all():
        raise ValueError("edge_support must be finite floating point")
    if not endpoint_probability.is_floating_point() or not torch.isfinite(
        endpoint_probability
    ).all():
        raise ValueError("endpoint_probability must be finite floating point")
    if torch.any(endpoint_indices < 0) or torch.any(endpoint_indices >= n_nodes):
        raise ValueError("endpoint index is outside the node carrier")
    if torch.any(endpoint_probability < 0) or not torch.allclose(
        endpoint_probability.sum(dim=-1),
        torch.ones_like(endpoint_probability[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("endpoint probabilities must be non-negative and sum to one")

    batch, edges, tiles = edge_support.shape
    numerator = edge_support.new_zeros((batch, n_nodes, tiles))
    degree = edge_support.new_zeros((batch, n_nodes, tiles))
    masked_support = torch.where(edge_valid, edge_support, 0.0)
    valid_float = edge_valid.to(edge_support.dtype)
    routing_weight = 2.0 * endpoint_probability.to(edge_support.dtype)
    for endpoint_position in range(2):
        node_index = endpoint_indices[:, endpoint_position].view(1, edges, 1)
        node_index = node_index.expand(batch, edges, tiles)
        numerator = numerator.scatter_add(
            1,
            node_index,
            masked_support * routing_weight[..., endpoint_position],
        )
        degree = degree.scatter_add(1, node_index, valid_float)
    node_mask = degree > 0
    support = numerator / degree.clamp_min(1.0)
    return torch.where(node_mask, support, 0.0), node_mask


@dataclass(frozen=True)
class VDirectedEndpointEvidenceOutput:
    event_logits: torch.Tensor
    channel_prior: torch.Tensor
    ictal_contribution: torch.Tensor
    evolution_contribution: torch.Tensor
    temporal_weights: torch.Tensor
    ictal_node_support: torch.Tensor
    ictal_node_mask: torch.Tensor
    evolution_tile_score: torch.Tensor
    routing_node_score: torch.Tensor
    endpoint_probability: torch.Tensor
    endpoint_valid: torch.Tensor
    endpoint_scale: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return (
            self.channel_prior
            + self.ictal_contribution
            + self.evolution_contribution
        )


class VDirectedEndpointTemporalMILReasoner(TemporalMILEvidenceReasoner):
    """Temporal-MIL head with one-scalar V-directed TCP endpoint routing."""

    def __init__(self, prior_logits: torch.Tensor, *, hidden_dim: int = 8) -> None:
        super().__init__(prior_logits, hidden_dim=hidden_dim)
        self.raw_endpoint_scale = torch.nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.register_buffer(
            "endpoint_indices", edge_endpoint_indices(), persistent=True
        )
        if self.n_trainable_parameters >= 500:
            raise ValueError("V-directed temporal-MIL head exceeds its capacity gate")

    def _route_ictal_directed(
        self,
        evidence: DevelopmentIVEvidenceBatch,
        ictal_mask: torch.Tensor,
        evolution_mask: torch.Tensor,
        routing_node_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_weights = self.ictal_feature_logits.softmax(dim=0).to(
            evidence.ictal.dtype
        )
        edge_support = (evidence.ictal * feature_weights).sum(dim=-1)
        edge_support = torch.where(ictal_mask, edge_support, 0.0)

        endpoints = self.endpoint_indices.to(device=edge_support.device)
        endpoint_score = routing_node_score[:, endpoints, :].permute(0, 1, 3, 2)
        endpoint_valid = evolution_mask[:, endpoints, :].permute(0, 1, 3, 2)
        endpoint_scale = F.softplus(self.raw_endpoint_scale).to(edge_support.dtype)
        endpoint_probability = masked_binary_endpoint_softmax(
            endpoint_scale * endpoint_score,
            endpoint_valid,
        )
        node_support, node_mask = route_edge_support_to_nodes(
            edge_support,
            ictal_mask,
            endpoint_probability,
            endpoints,
        )
        return node_support, node_mask, endpoint_probability, endpoint_valid

    def forward(
        self,
        evidence: DevelopmentIVEvidenceBatch,
    ) -> VDirectedEndpointEvidenceOutput:
        if not isinstance(evidence, DevelopmentIVEvidenceBatch):
            raise TypeError("V-directed reasoner accepts evidence batches only")
        evidence.validate()
        phase = evidence.phase_mask.unsqueeze(1)
        evolution_mask = evidence.evolution_mask & phase
        ictal_mask = evidence.ictal_mask & phase

        current = torch.where(
            evolution_mask.unsqueeze(-1), evidence.evolution, 0.0
        )
        previous = torch.roll(current, shifts=1, dims=2)
        previous_mask = torch.roll(evolution_mask, shifts=1, dims=2)
        previous_mask[:, :, 0] = False
        delta_mask = evolution_mask & previous_mask
        delta = torch.where(
            delta_mask.unsqueeze(-1), current - previous, torch.zeros_like(current)
        )
        evolution_features = torch.cat((current, delta), dim=-1)
        evolution_tile = self.evolution_scorer(evolution_features).squeeze(-1)
        evolution_tile = torch.where(evolution_mask, evolution_tile, 0.0)

        reliability = torch.where(
            evolution_mask,
            evidence.reliability,
            torch.zeros_like(evidence.reliability),
        )
        routing_node_score = _positive_only_reliability_gate(
            evolution_tile,
            reliability,
        )
        (
            ictal_node,
            ictal_node_mask,
            endpoint_probability,
            endpoint_valid,
        ) = self._route_ictal_directed(
            evidence,
            ictal_mask,
            evolution_mask,
            routing_node_score,
        )

        attention_scale = F.softplus(self.raw_attention_scale).to(ictal_node.dtype)
        centered_bias = self.temporal_bias - self.temporal_bias.mean()
        energy = attention_scale * ictal_node + centered_bias.view(1, 1, -1)
        temporal_weights = _masked_softmax(energy, evolution_mask)

        gated_ictal = (
            ictal_node * reliability * ictal_node_mask.to(ictal_node.dtype)
        )
        evolution_gain = F.softplus(self.raw_evolution_gain).to(evolution_tile.dtype)
        ictal_gain = F.softplus(self.raw_ictal_gain).to(ictal_node.dtype)
        evolution_contribution = evolution_gain * (
            temporal_weights * routing_node_score
        ).sum(dim=-1)
        ictal_contribution = ictal_gain * (
            temporal_weights * gated_ictal
        ).sum(dim=-1)
        prior = self.channel_prior_logits.to(evolution_tile.dtype).unsqueeze(0)
        prior = prior.expand(evidence.batch_size, -1)
        logits = prior + evolution_contribution + ictal_contribution
        output = VDirectedEndpointEvidenceOutput(
            event_logits=logits,
            channel_prior=prior,
            ictal_contribution=ictal_contribution,
            evolution_contribution=evolution_contribution,
            temporal_weights=temporal_weights,
            ictal_node_support=ictal_node,
            ictal_node_mask=ictal_node_mask,
            evolution_tile_score=evolution_tile,
            routing_node_score=routing_node_score,
            endpoint_probability=endpoint_probability,
            endpoint_valid=endpoint_valid,
            endpoint_scale=F.softplus(self.raw_endpoint_scale),
        )
        if not torch.allclose(
            output.reconstructed_logits(), logits, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("V-directed contribution decomposition drifted")
        return output
