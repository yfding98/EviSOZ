"""Small additive SOZ reasoner that accepts evidence concepts only.

Morphology is deliberately routed through clinically typed ports.  Only the
local SPSW/PLED port can add localization support.  GPED loses edge identity
before it forms a non-increasing specificity gate, EYEM/ARTF form a
non-increasing reliability gate, and BCKG is receipt-only support/OOD
metadata.  This module therefore does not expose the legacy unconstrained
``CE6 mean/max -> MLP -> channel logit`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..evidence import EvidenceBatch
from ..evidence_schema import split_typed_morphology
from ..geometry import (
    N_ICTAL_FEATURES,
    N_MORPHOLOGY_FEATURES,
    N_NODE_FEATURES,
    N_STANDARD_CHANNELS,
    unsigned_incidence_matrix,
)


N_REASONER_TILES = 15
PHASE_COMPONENT_NAMES = (
    "pre",
    "early",
    "late",
    "early_minus_pre",
    "late_minus_early",
)
_PHASE_BOUNDS = ((0, 3), (3, 6), (6, 15))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class _FamilyScorer(nn.Module):
    """Shared low-capacity scorer for one named evidence family."""

    def __init__(self, n_features: int, hidden_dim: int) -> None:
        super().__init__()
        if n_features < 1 or hidden_dim < 1:
            raise ValueError("n_features and hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence).squeeze(-1)


class _PositiveLocalizingScorer(nn.Module):
    """Convex SPSW/PLED scorer with a zero-preserving positive path only."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        self.feature_logits = nn.Parameter(torch.zeros(n_features))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.shape[-1] != self.feature_logits.numel():
            raise ValueError("Localizing evidence feature dimension drifted")
        weights = self.feature_logits.softmax(dim=0).to(dtype=evidence.dtype)
        return (evidence * weights).sum(dim=-1)


class _NonIncreasingGate(nn.Module):
    """Learned gate that cannot increase when any burden feature increases."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        self.feature_logits = nn.Parameter(torch.zeros(n_features))
        self.raw_strength = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )

    def forward(self, burden: torch.Tensor) -> torch.Tensor:
        if burden.shape[-1] != self.feature_logits.numel():
            raise ValueError("Gate burden feature dimension drifted")
        weights = self.feature_logits.softmax(dim=0).to(dtype=burden.dtype)
        strength = F.softplus(self.raw_strength).to(dtype=burden.dtype)
        scalar_burden = (burden * weights).sum(dim=-1)
        return torch.exp(-strength * scalar_burden)


def _masked_time_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (values * mask.to(dtype=values.dtype)).sum(dim=-1)
    denominator = mask.sum(dim=-1).clamp_min(1).to(dtype=values.dtype)
    return numerator / denominator


def _masked_feature_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-aware mean over the penultimate (time) axis."""

    if values.ndim < 3 or tuple(values.shape[:-1]) != tuple(mask.shape):
        raise ValueError("Feature values/mask must share all non-feature axes")
    valid = mask.any(dim=-1)
    numerator = (
        values * mask.unsqueeze(-1).to(dtype=values.dtype)
    ).sum(dim=-2)
    denominator = mask.sum(dim=-1).clamp_min(1).to(dtype=values.dtype)
    mean = numerator / denominator.unsqueeze(-1)
    return torch.where(valid.unsqueeze(-1), mean, 0.0), valid


def _validate_observed_unit_interval(
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if tuple(values.shape[:-1]) != tuple(mask.shape):
        raise ValueError(f"{name} values and mask disagree")
    observed = values[mask]
    if observed.numel() and torch.any((observed < 0) | (observed > 1)):
        raise ValueError(f"Observed {name} evidence must lie in [0,1]")


def _phase_components(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return three phase means and two valid deterministic contrasts.

    The final axis is fixed to the 15 four-second bins spanning
    ``[-12,48)``. Component order follows :data:`PHASE_COMPONENT_NAMES`.
    """

    if values.shape != mask.shape or values.ndim != 3:
        raise ValueError("Phase values and masks must share shape [B,item,15]")
    if values.shape[-1] != N_REASONER_TILES:
        raise ValueError("SOZ reasoner requires exactly 15 four-second evidence tiles")
    phase_values: list[torch.Tensor] = []
    phase_masks: list[torch.Tensor] = []
    for start, stop in _PHASE_BOUNDS:
        phase_mask = mask[..., start:stop]
        phase_valid = phase_mask.any(dim=-1)
        phase_mean = _masked_time_mean(values[..., start:stop], phase_mask)
        phase_values.append(torch.where(phase_valid, phase_mean, 0.0))
        phase_masks.append(phase_valid)

    pre, early, late = phase_values
    pre_valid, early_valid, late_valid = phase_masks
    early_pre_valid = early_valid & pre_valid
    late_early_valid = late_valid & early_valid
    early_minus_pre = torch.where(early_pre_valid, early - pre, 0.0)
    late_minus_early = torch.where(late_early_valid, late - early, 0.0)
    component_values = torch.stack(
        (pre, early, late, early_minus_pre, late_minus_early), dim=-1
    )
    component_mask = torch.stack(
        (pre_valid, early_valid, late_valid, early_pre_valid, late_early_valid),
        dim=-1,
    )
    return component_values, component_mask


class _PhaseCombiner(nn.Module):
    """Five learned scalar weights over named, deterministic time components."""

    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.full((len(PHASE_COMPONENT_NAMES),), 0.2))

    def forward(
        self,
        components: torch.Tensor,
        component_mask: torch.Tensor,
    ) -> torch.Tensor:
        if components.shape != component_mask.shape or components.shape[-1] != len(
            PHASE_COMPONENT_NAMES
        ):
            raise ValueError("Phase components and masks must share [...,5] shape")
        return (
            components
            * component_mask.to(dtype=components.dtype)
            * self.weights.to(dtype=components.dtype)
        )


class _PositivePhaseCombiner(nn.Module):
    """Combine one family as non-negative localization support.

    A positive phase value or temporal contrast can add support; a negative
    value is retained in the raw receipt but cannot become a sign-reversing
    localization path.  This is what makes subsequent non-increasing
    reliability gates signed by construction.
    """

    def __init__(self) -> None:
        super().__init__()
        initial = _inverse_softplus(0.2)
        self.weights = nn.Parameter(
            torch.full((len(PHASE_COMPONENT_NAMES),), initial)
        )

    def forward(
        self,
        components: torch.Tensor,
        component_mask: torch.Tensor,
    ) -> torch.Tensor:
        if components.shape != component_mask.shape or components.shape[-1] != len(
            PHASE_COMPONENT_NAMES
        ):
            raise ValueError("Phase components and masks must share [...,5] shape")
        weights = F.softplus(self.weights).to(dtype=components.dtype)
        return (
            components.clamp_min(0.0)
            * component_mask.to(dtype=components.dtype)
            * weights
        )


def _degree_normalized_edge_components(
    edge_contributions: torch.Tensor,
    component_mask: torch.Tensor,
    incidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route edge components by valid incident-edge mean, never raw sum."""

    if edge_contributions.shape != component_mask.shape or edge_contributions.ndim != 3:
        raise ValueError("Edge contributions/mask must share [B,20,5] shape")
    if tuple(incidence.shape) != (N_STANDARD_CHANNELS, edge_contributions.shape[1]):
        raise ValueError("Incidence matrix shape drifted")
    incidence = incidence.to(
        device=edge_contributions.device, dtype=edge_contributions.dtype
    )
    valid = component_mask.to(dtype=edge_contributions.dtype)
    valid_degree = torch.einsum("ce,beq->bcq", incidence, valid)
    routed = (
        edge_contributions.unsqueeze(1)
        * incidence.unsqueeze(0).unsqueeze(-1)
        / valid_degree.clamp_min(1.0).unsqueeze(2)
    )
    routed = torch.where(
        (valid_degree > 0).unsqueeze(2), routed, torch.zeros_like(routed)
    )
    return routed, valid_degree


def _conservative_node_artifact_reliability(
    quality_abstention: torch.Tensor,
    context_mask: torch.Tensor,
    incidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project EYEM/ARTF burden to a deterministic node reliability.

    The bipolar morphology head cannot identify which endpoint generated an
    artifact, so every observed edge burden is conservatively assigned to both
    endpoints.  The maximum over the four EYEM/ARTF summaries and over time is
    used rather than a target-fitted pooling rule.  FZ/PZ have no common-20
    incident edge and therefore receive the event-global burden whenever any
    morphology context is available.  If the morphology family is entirely
    unavailable, reliability is neutral (one); that case is exposed explicitly
    and is only suitable for the prespecified V-only baseline.
    """

    if quality_abstention.ndim != 4 or quality_abstention.shape[-1] != 4:
        raise ValueError("Morphology quality evidence must have shape [B,20,T,4]")
    if tuple(context_mask.shape) != tuple(quality_abstention.shape[:-1]):
        raise ValueError("Morphology quality evidence/context mask disagree")
    if context_mask.dtype != torch.bool:
        raise TypeError("Morphology context mask must be bool")
    if tuple(incidence.shape) != (
        N_STANDARD_CHANNELS,
        quality_abstention.shape[1],
    ):
        raise ValueError("Incidence matrix shape drifted for artifact projection")

    edge_burden = quality_abstention.amax(dim=-1)
    observed_edge_burden = torch.where(context_mask, edge_burden, 0.0)
    incidence_mask = incidence.to(device=context_mask.device, dtype=torch.bool)
    incident_context = (
        context_mask.unsqueeze(1) & incidence_mask.unsqueeze(0).unsqueeze(-1)
    )
    incident_burden = torch.where(
        incident_context,
        observed_edge_burden.unsqueeze(1),
        torch.zeros_like(observed_edge_burden.unsqueeze(1)),
    ).amax(dim=(2, 3))
    incident_available = incident_context.any(dim=(2, 3))

    event_context_available = context_mask.any(dim=(1, 2))
    event_burden = observed_edge_burden.amax(dim=(1, 2))
    fallback_burden = event_burden.unsqueeze(1).expand(
        -1, N_STANDARD_CHANNELS
    )
    node_burden = torch.where(
        incident_available,
        incident_burden,
        torch.where(
            event_context_available.unsqueeze(1),
            fallback_burden,
            torch.zeros_like(fallback_burden),
        ),
    ).clamp(0.0, 1.0)
    reliability = 1.0 - node_burden
    return reliability, node_burden, incident_available


@dataclass(frozen=True)
class ReasonerOutput:
    """Raw prediction plus a numerical receipt that exactly reconstructs it.

    ``event_logits`` are uncalibrated event-level logits.  They are averaged
    within patient before the reasoner objective.  A global calibrator may be
    fitted only to patient logits after reasoner weights are frozen; calibrated
    probabilities intentionally have no field or computation path here.
    """

    event_logits: torch.Tensor
    channel_prior: torch.Tensor
    evolution_tile: torch.Tensor
    evolution_node: torch.Tensor
    morphology_edge_tile: torch.Tensor
    morphology_node_edge: torch.Tensor
    ictal_edge_tile: torch.Tensor
    ictal_node_edge: torch.Tensor
    evolution_phase_component: torch.Tensor
    evolution_phase_component_mask: torch.Tensor
    evolution_phase_contribution: torch.Tensor
    evolution_artifact_reliability: torch.Tensor
    evolution_artifact_burden: torch.Tensor
    evolution_artifact_incident_context_available: torch.Tensor
    morphology_edge_phase_component: torch.Tensor
    morphology_edge_phase_component_mask: torch.Tensor
    morphology_edge_phase_raw_contribution: torch.Tensor
    morphology_edge_phase_contribution: torch.Tensor
    morphology_node_edge_phase_contribution: torch.Tensor
    morphology_node_valid_degree: torch.Tensor
    morphology_quality_tile: torch.Tensor
    morphology_generalized_tile: torch.Tensor
    morphology_generalized_tile_mask: torch.Tensor
    morphology_support_tile: torch.Tensor
    morphology_quality_gate: torch.Tensor
    morphology_specificity_gate: torch.Tensor
    ictal_edge_phase_component: torch.Tensor
    ictal_edge_phase_component_mask: torch.Tensor
    ictal_edge_phase_contribution: torch.Tensor
    ictal_node_edge_phase_contribution: torch.Tensor
    ictal_node_valid_degree: torch.Tensor
    ictal_phase_mask: torch.Tensor
    evolution_effective_mask: torch.Tensor
    morphology_effective_mask: torch.Tensor
    morphology_context_effective_mask: torch.Tensor
    ictal_effective_mask: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return sum(self.component_contributions().values())

    def component_contributions(self) -> dict[str, torch.Tensor]:
        """Expose every weighted phase/contrast term in channel coordinates."""

        contributions: dict[str, torch.Tensor] = {
            "channel_prior": self.channel_prior,
        }
        for component_index, component_name in enumerate(PHASE_COMPONENT_NAMES):
            contributions[f"evolution/{component_name}"] = (
                self.evolution_phase_contribution[..., component_index]
            )
            contributions[f"morphology/{component_name}"] = (
                self.morphology_node_edge_phase_contribution[
                    ..., component_index
                ].sum(dim=-1)
            )
            contributions[f"ictal_involvement/{component_name}"] = (
                self.ictal_node_edge_phase_contribution[
                    ..., component_index
                ].sum(dim=-1)
            )
        return contributions

    def family_contributions(self) -> dict[str, torch.Tensor]:
        return {
            "channel_prior": self.channel_prior,
            "evolution": self.evolution_phase_contribution.sum(dim=-1),
            "morphology": self.morphology_node_edge_phase_contribution.sum(
                dim=(-1, -2)
            ),
            "ictal_involvement": self.ictal_node_edge_phase_contribution.sum(
                dim=(-1, -2)
            ),
        }


class AdditiveEvidenceReasoner(nn.Module):
    """Map finite node/edge evidence to event-level standard-19 logits.

    The public API accepts :class:`EvidenceBatch`, so raw EEG, foundation
    tokens, hidden adapter states, patient identity, and clinical text have no
    input path.  Bipolar membership is projected through an unsigned incidence
    matrix, so polarity cannot choose an endpoint.  Each node then receives
    the mean over its valid incident edges; the node-specific denominator
    removes degree bias without claiming an edge endpoint as SOZ truth.
    """

    def __init__(self, *, hidden_dim: int = 16) -> None:
        super().__init__()
        self.evolution_scorer = _FamilyScorer(N_NODE_FEATURES, hidden_dim)
        self.morphology_scorer = _PositiveLocalizingScorer(4)
        self.morphology_quality_gate = _NonIncreasingGate(4)
        self.morphology_specificity_gate = _NonIncreasingGate(2)
        self.ictal_scorer = _FamilyScorer(N_ICTAL_FEATURES, hidden_dim)
        # Evolution is positive support, not an unconstrained negative-evidence
        # path.  This makes the deterministic artifact reliability monotone:
        # increasing EYEM/ARTF burden cannot increase V localization support.
        self.evolution_phase_combiner = _PositivePhaseCombiner()
        self.morphology_phase_combiner = _PositivePhaseCombiner()
        self.ictal_phase_combiner = _PhaseCombiner()
        self.channel_bias = nn.Parameter(torch.zeros(N_STANDARD_CHANNELS))
        self.register_buffer(
            "incidence",
            unsigned_incidence_matrix(),
            persistent=True,
        )
        if self.n_trainable_parameters >= 50_000:
            raise ValueError("Reasoner exceeds the frozen 50k-parameter capacity gate")

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, evidence: EvidenceBatch) -> ReasonerOutput:
        if not isinstance(evidence, EvidenceBatch):
            raise TypeError("AdditiveEvidenceReasoner accepts EvidenceBatch only")
        if evidence.node.requires_grad or evidence.edge.requires_grad:
            raise ValueError(
                "Evidence bottleneck tensors must be detached before SOZ reasoning; "
                "gradients into concept extractors are forbidden"
            )
        if evidence.n_tiles != N_REASONER_TILES:
            raise ValueError(
                "SOZ reasoner requires exactly 15 four-second evidence tiles "
                "covering [-12,48) seconds"
            )

        ictal_phase_mask = evidence.ictal_phase_mask
        phase_node_mask = ictal_phase_mask.unsqueeze(1)
        node_mask = (
            evidence.node_mask
            & evidence.physical_signal_mask
            & phase_node_mask
        )
        morphology_mask = evidence.morphology_mask & phase_node_mask
        morphology_context_mask = (
            evidence.morphology_context_mask & phase_node_mask
        )
        ictal_mask = evidence.ictal_mask & phase_node_mask
        safe_node = torch.where(node_mask.unsqueeze(-1), evidence.node, 0.0)

        evolution_tile = self.evolution_scorer(safe_node)
        evolution_tile = evolution_tile * node_mask.to(evolution_tile.dtype)
        evolution_phase_component, evolution_phase_component_mask = (
            _phase_components(evolution_tile, node_mask)
        )
        flat_morphology = evidence.edge[..., :N_MORPHOLOGY_FEATURES]
        ictal = torch.where(
            ictal_mask.unsqueeze(-1),
            evidence.edge[
            ...,
            N_MORPHOLOGY_FEATURES : N_MORPHOLOGY_FEATURES + N_ICTAL_FEATURES,
            ],
            0.0,
        )
        _validate_observed_unit_interval(
            "morphology", flat_morphology, morphology_context_mask
        )
        _validate_observed_unit_interval("ictal", ictal, ictal_mask)
        typed_morphology = split_typed_morphology(
            flat_morphology,
            morphology_mask,
            morphology_context_mask,
        )
        (
            evolution_artifact_reliability,
            evolution_artifact_burden,
            evolution_artifact_incident_context_available,
        ) = _conservative_node_artifact_reliability(
            typed_morphology.quality_abstention,
            morphology_context_mask,
            self.incidence,
        )
        evolution_phase_contribution = self.evolution_phase_combiner(
            evolution_phase_component, evolution_phase_component_mask
        )
        evolution_phase_contribution = (
            evolution_phase_contribution
            * evolution_artifact_reliability.unsqueeze(-1)
        )
        evolution_node = evolution_phase_contribution.sum(dim=-1)
        morphology_edge_tile = self.morphology_scorer(
            typed_morphology.localizing_positive
        )
        ictal_edge_tile = self.ictal_scorer(ictal)
        morphology_edge_tile = morphology_edge_tile * morphology_mask.to(
            morphology_edge_tile.dtype
        )
        ictal_edge_tile = ictal_edge_tile * ictal_mask.to(ictal_edge_tile.dtype)

        morphology_edge_phase_component, morphology_edge_phase_component_mask = (
            _phase_components(morphology_edge_tile, morphology_mask)
        )
        ictal_edge_phase_component, ictal_edge_phase_component_mask = (
            _phase_components(ictal_edge_tile, ictal_mask)
        )
        morphology_edge_phase_raw_contribution = self.morphology_phase_combiner(
            morphology_edge_phase_component, morphology_edge_phase_component_mask
        )
        ictal_edge_phase_contribution = self.ictal_phase_combiner(
            ictal_edge_phase_component, ictal_edge_phase_component_mask
        )

        quality_summary, quality_valid = _masked_feature_mean(
            typed_morphology.quality_abstention,
            morphology_context_mask,
        )
        generalized_summary, generalized_valid = _masked_feature_mean(
            typed_morphology.generalized_conflict,
            typed_morphology.generalized_mask,
        )
        morphology_quality_gate = self.morphology_quality_gate(quality_summary)
        morphology_quality_gate = torch.where(
            quality_valid, morphology_quality_gate, 0.0
        )
        morphology_specificity_gate = self.morphology_specificity_gate(
            generalized_summary
        )
        morphology_specificity_gate = torch.where(
            generalized_valid, morphology_specificity_gate, 0.0
        )
        morphology_edge_phase_contribution = (
            morphology_edge_phase_raw_contribution
            * morphology_quality_gate.unsqueeze(-1)
            * morphology_specificity_gate.unsqueeze(1).unsqueeze(-1)
        )

        morphology_node_edge_phase_contribution, morphology_node_valid_degree = (
            _degree_normalized_edge_components(
                morphology_edge_phase_contribution,
                morphology_edge_phase_component_mask,
                self.incidence,
            )
        )
        ictal_node_edge_phase_contribution, ictal_node_valid_degree = (
            _degree_normalized_edge_components(
                ictal_edge_phase_contribution,
                ictal_edge_phase_component_mask,
                self.incidence,
            )
        )
        morphology_node_edge = morphology_node_edge_phase_contribution.sum(dim=-1)
        ictal_node_edge = ictal_node_edge_phase_contribution.sum(dim=-1)

        channel_prior = self.channel_bias.to(dtype=evidence.node.dtype).unsqueeze(0)
        channel_prior = channel_prior.expand(evidence.batch_size, -1)
        event_logits = (
            channel_prior
            + evolution_node
            + morphology_node_edge.sum(dim=-1)
            + ictal_node_edge.sum(dim=-1)
        )
        return ReasonerOutput(
            event_logits=event_logits,
            channel_prior=channel_prior,
            evolution_tile=evolution_tile,
            evolution_node=evolution_node,
            morphology_edge_tile=morphology_edge_tile,
            morphology_node_edge=morphology_node_edge,
            ictal_edge_tile=ictal_edge_tile,
            ictal_node_edge=ictal_node_edge,
            evolution_phase_component=evolution_phase_component,
            evolution_phase_component_mask=evolution_phase_component_mask,
            evolution_phase_contribution=evolution_phase_contribution,
            evolution_artifact_reliability=evolution_artifact_reliability,
            evolution_artifact_burden=evolution_artifact_burden,
            evolution_artifact_incident_context_available=(
                evolution_artifact_incident_context_available
            ),
            morphology_edge_phase_component=morphology_edge_phase_component,
            morphology_edge_phase_component_mask=morphology_edge_phase_component_mask,
            morphology_edge_phase_raw_contribution=(
                morphology_edge_phase_raw_contribution
            ),
            morphology_edge_phase_contribution=morphology_edge_phase_contribution,
            morphology_node_edge_phase_contribution=(
                morphology_node_edge_phase_contribution
            ),
            morphology_node_valid_degree=morphology_node_valid_degree,
            morphology_quality_tile=typed_morphology.quality_abstention,
            morphology_generalized_tile=typed_morphology.generalized_conflict,
            morphology_generalized_tile_mask=(
                typed_morphology.generalized_mask
            ),
            morphology_support_tile=typed_morphology.support_ood,
            morphology_quality_gate=morphology_quality_gate,
            morphology_specificity_gate=morphology_specificity_gate,
            ictal_edge_phase_component=ictal_edge_phase_component,
            ictal_edge_phase_component_mask=ictal_edge_phase_component_mask,
            ictal_edge_phase_contribution=ictal_edge_phase_contribution,
            ictal_node_edge_phase_contribution=ictal_node_edge_phase_contribution,
            ictal_node_valid_degree=ictal_node_valid_degree,
            ictal_phase_mask=ictal_phase_mask,
            evolution_effective_mask=node_mask,
            morphology_effective_mask=morphology_mask,
            morphology_context_effective_mask=morphology_context_mask,
            ictal_effective_mask=ictal_mask,
        )
