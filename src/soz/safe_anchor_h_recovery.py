"""Top-1-safe fusion and strictly gated TCP-endpoint reranking.

This module contains no data loader and no model-selection loop.  The safe
fusion is target free: it may reorder electrodes below a frozen anchor's top
set, but it is constructed so that it cannot change that top set.  The
endpoint gate is separate because changing the top electrode can improve (or
harm) strict SOZ localization and therefore requires outer-train cross-fitted
predictions and labels.

The LaBraM score accepted here is a ranking signal, not a named morphology,
onset, propagation, or causal SOZ concept.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .geometry import (
    CHANNEL_INDEX,
    N_STANDARD_CHANNELS,
    TCP_20_EDGES,
)


SAFE_ANCHOR_H_RECOVERY_SCHEMA = "soz_labram_safe_anchor_h_recovery_v4"
SAFE_RESIDUAL_BUDGET_FRACTION = 0.5
ENDPOINT_GATE_ALPHA = 0.05


def _validate_score_triplet(
    anchor_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
) -> None:
    expected = (anchor_scores.shape[0], N_STANDARD_CHANNELS)
    if anchor_scores.ndim != 2 or tuple(anchor_scores.shape) != expected:
        raise ValueError("anchor_scores must have shape [P,19]")
    if tuple(residual_scores.shape) != expected or tuple(evaluable_mask.shape) != expected:
        raise ValueError("residual_scores and evaluable_mask must have shape [P,19]")
    if not anchor_scores.is_floating_point() or not residual_scores.is_floating_point():
        raise TypeError("anchor and residual scores must be floating point")
    if evaluable_mask.dtype != torch.bool:
        raise TypeError("evaluable_mask must be bool")
    if anchor_scores.device != residual_scores.device or (
        anchor_scores.device != evaluable_mask.device
    ):
        raise ValueError("safe-fusion inputs must share a device")
    if anchor_scores.shape[0] < 1 or not evaluable_mask.any(dim=1).all():
        raise ValueError("every patient must have at least one evaluable electrode")
    if not torch.isfinite(anchor_scores).all() or not torch.isfinite(
        residual_scores
    ).all():
        raise ValueError("safe-fusion scores must be finite")


def _masked_top_set(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~mask, -torch.inf)
    top = masked.max(dim=1, keepdim=True).values
    return mask & (scores == top)


def prior_cancelled_log_probability_ratio(
    node_probabilities: torch.Tensor,
    prior_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return ``log(P_node / P_prior)`` as a target-free ranking residual.

    Both inputs must be normalized patient-level probability maps.  The ratio
    removes the explicit fold-local prevalence map multiplicatively.  It does
    not prove that remaining variation is physiological: static LaBraM
    channel/position signatures may remain.
    """

    if node_probabilities.ndim != 2 or node_probabilities.shape[1] != (
        N_STANDARD_CHANNELS
    ):
        raise ValueError("node_probabilities must have shape [P,19]")
    if tuple(prior_probabilities.shape) != tuple(node_probabilities.shape):
        raise ValueError("node and prior probability shapes differ")
    if not node_probabilities.is_floating_point() or not (
        prior_probabilities.is_floating_point()
    ):
        raise TypeError("probability maps must be floating point")
    if node_probabilities.device != prior_probabilities.device:
        raise ValueError("probability maps must share a device")
    for name, value in (
        ("node", node_probabilities),
        ("prior", prior_probabilities),
    ):
        if not torch.isfinite(value).all() or torch.any(value <= 0):
            raise ValueError(f"{name} probabilities must be finite and strictly positive")
        if not torch.allclose(
            value.sum(dim=1),
            torch.ones(value.shape[0], dtype=value.dtype, device=value.device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(f"{name} probability rows must sum to one")
    return (node_probabilities.log() - prior_probabilities.log()).contiguous()


@dataclass(frozen=True)
class Top1SafeResidualOutput:
    scores: torch.Tensor
    delta: torch.Tensor
    anchor_top_set: torch.Tensor
    margin_budget: torch.Tensor

    @property
    def changed_patient_count(self) -> int:
        return int((self.delta != 0).any(dim=1).sum().item())


def top1_safe_bounded_residual(
    anchor_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
    *,
    budget_fraction: float = SAFE_RESIDUAL_BUDGET_FRACTION,
) -> Top1SafeResidualOutput:
    """Rerank only below the anchor top set with a runner-up-gap budget.

    For patient ``p``, let ``g_p`` be the gap between the frozen top value and
    the best strictly lower score.  The centered residual is normalized to
    ``[-1, 1]`` and receives at most ``0.5 * g_p``.  Every member of the
    original top tie set is left bitwise unchanged.  A ``nextafter`` ceiling
    protects the strict inequality after finite-precision rounding.

    Consequently the complete evaluable top tie set is identical before and
    after fusion.  Strict and relaxed Top-1 metrics are therefore identical;
    AP, MRR, and Hit@K for K>1 are not mathematically guaranteed.
    """

    _validate_score_triplet(anchor_scores, residual_scores, evaluable_mask)
    if isinstance(budget_fraction, bool) or not math.isfinite(budget_fraction) or (
        budget_fraction != SAFE_RESIDUAL_BUDGET_FRACTION
    ):
        raise ValueError("budget_fraction is frozen at 0.5; it is not tunable")

    top_set = _masked_top_set(anchor_scores, evaluable_mask)
    scores = anchor_scores.clone()
    delta = torch.zeros_like(anchor_scores)
    budgets = anchor_scores.new_zeros((anchor_scores.shape[0],))
    negative_infinity = torch.full((), -torch.inf, device=anchor_scores.device)

    for patient in range(anchor_scores.shape[0]):
        lower = evaluable_mask[patient] & ~top_set[patient]
        if not bool(lower.any()):
            continue
        top_value = anchor_scores[patient][top_set[patient]][0]
        runner_up = anchor_scores[patient][lower].max()
        gap = top_value - runner_up
        if not bool(gap > 0):
            raise RuntimeError("strictly lower anchor set has a non-positive gap")
        budget = gap * SAFE_RESIDUAL_BUDGET_FRACTION
        centered = residual_scores[patient][lower]
        centered = centered - centered.mean()
        denominator = centered.abs().max()
        if bool(denominator > 0):
            proposed_delta = budget * centered / denominator
            proposed = anchor_scores[patient][lower] + proposed_delta
            ceiling = torch.nextafter(top_value, negative_infinity)
            proposed = torch.minimum(proposed, ceiling)
            scores[patient][lower] = proposed
            delta[patient][lower] = proposed - anchor_scores[patient][lower]
        budgets[patient] = budget

    if not torch.equal(_masked_top_set(scores, evaluable_mask), top_set):
        raise RuntimeError("bounded residual changed the frozen anchor top set")
    if not torch.equal(scores[top_set], anchor_scores[top_set]):
        raise RuntimeError("bounded residual changed a frozen top-set value")
    if bool((delta[~evaluable_mask] != 0).any()):
        raise RuntimeError("bounded residual changed a non-evaluable electrode")
    return Top1SafeResidualOutput(
        scores=scores,
        delta=delta,
        anchor_top_set=top_set,
        margin_budget=budgets,
    )


def _tcp_neighbours() -> tuple[tuple[int, ...], ...]:
    rows: list[set[int]] = [set() for _ in range(N_STANDARD_CHANNELS)]
    for left, right in TCP_20_EDGES:
        left_index = CHANNEL_INDEX[left]
        right_index = CHANNEL_INDEX[right]
        rows[left_index].add(right_index)
        rows[right_index].add(left_index)
    return tuple(tuple(sorted(row)) for row in rows)


TCP_ENDPOINT_NEIGHBOURS = _tcp_neighbours()


@dataclass(frozen=True)
class TCPEndpointFlipProposal:
    anchor_index: torch.Tensor
    candidate_index: torch.Tensor
    proposed: torch.Tensor
    residual_margin: torch.Tensor

    @property
    def proposal_count(self) -> int:
        return int(self.proposed.sum().item())


def propose_tcp_endpoint_flips(
    anchor_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
) -> TCPEndpointFlipProposal:
    """Propose a label-free flip only across one observed TCP-20 edge.

    A proposal requires a unique anchor Top-1, a unique best evaluable TCP
    neighbour under the residual, and a strictly positive residual margin over
    the anchor electrode.  DeepSOZ's broader one-hop evaluation neighbourhood
    is deliberately not used here.
    """

    _validate_score_triplet(anchor_scores, residual_scores, evaluable_mask)
    patients = anchor_scores.shape[0]
    anchor_index = torch.full(
        (patients,), -1, dtype=torch.long, device=anchor_scores.device
    )
    candidate_index = torch.full_like(anchor_index, -1)
    proposed = torch.zeros(patients, dtype=torch.bool, device=anchor_scores.device)
    residual_margin = anchor_scores.new_zeros((patients,))
    top_set = _masked_top_set(anchor_scores, evaluable_mask)

    for patient in range(patients):
        top_indices = torch.nonzero(top_set[patient], as_tuple=False).flatten()
        if top_indices.numel() != 1:
            continue
        anchor = int(top_indices.item())
        neighbours = tuple(
            index
            for index in TCP_ENDPOINT_NEIGHBOURS[anchor]
            if bool(evaluable_mask[patient, index])
        )
        if not neighbours:
            continue
        neighbour_index = torch.tensor(
            neighbours, dtype=torch.long, device=anchor_scores.device
        )
        neighbour_scores = residual_scores[patient].index_select(0, neighbour_index)
        maximum = neighbour_scores.max()
        tied = neighbour_index[neighbour_scores == maximum]
        if tied.numel() != 1:
            continue
        candidate = int(tied.item())
        margin = residual_scores[patient, candidate] - residual_scores[patient, anchor]
        if not bool(margin > 0):
            continue
        anchor_index[patient] = anchor
        candidate_index[patient] = candidate
        proposed[patient] = True
        residual_margin[patient] = margin

    return TCPEndpointFlipProposal(
        anchor_index=anchor_index,
        candidate_index=candidate_index,
        proposed=proposed,
        residual_margin=residual_margin,
    )


def _one_sided_exact_sign_p_value(beneficial: int, harmful: int) -> float:
    decisive = beneficial + harmful
    if decisive == 0:
        return 1.0
    numerator = sum(
        math.comb(decisive, successes)
        for successes in range(beneficial, decisive + 1)
    )
    return float(numerator / (2**decisive))


@dataclass(frozen=True)
class EndpointFlipGate:
    enabled: bool
    proposal_count: int
    beneficial_count: int
    harmful_count: int
    neutral_count: int
    decisive_count: int
    one_sided_exact_p_value: float
    alpha: float
    scope: str
    within_tcp_anchor_direction_accuracy: float
    within_tcp_residual_direction_accuracy: float
    within_tcp_direction_prerequisite_pass: bool

    def __post_init__(self) -> None:
        counts = (
            self.proposal_count,
            self.beneficial_count,
            self.harmful_count,
            self.neutral_count,
            self.decisive_count,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("endpoint gate counts must be non-negative integers")
        if self.decisive_count != self.beneficial_count + self.harmful_count or (
            self.proposal_count
            != self.decisive_count + self.neutral_count
        ):
            raise ValueError("endpoint gate counts are internally inconsistent")
        if self.alpha != ENDPOINT_GATE_ALPHA or not (
            0 <= self.one_sided_exact_p_value <= 1
        ):
            raise ValueError("endpoint gate alpha or p-value is invalid")
        if self.scope != "outer_train_inner_oof_only":
            raise ValueError("endpoint gate has a non-nested scope")
        if not math.isfinite(self.within_tcp_anchor_direction_accuracy) or not (
            math.isfinite(self.within_tcp_residual_direction_accuracy)
        ):
            raise ValueError("endpoint direction accuracies must be finite")
        if self.enabled and not (
            self.within_tcp_direction_prerequisite_pass
            and self.beneficial_count > self.harmful_count
            and self.one_sided_exact_p_value <= self.alpha
        ):
            raise ValueError("enabled endpoint gate does not pass all frozen checks")

    @property
    def net_strict_gain_count(self) -> int:
        return self.beneficial_count - self.harmful_count


@dataclass(frozen=True)
class WithinTCPEdgeDirectionMetrics:
    """Observed-positive versus observed-negative endpoint direction only."""

    patient_macro_accuracy: float
    eligible_patient_count: int
    informative_pair_count: int
    patient_accuracy: torch.Tensor
    informative_pair_count_by_patient: torch.Tensor


def within_tcp_edge_direction_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> WithinTCPEdgeDirectionMetrics:
    """Audit direction on TCP edges with one observed positive endpoint.

    An informative pair has both endpoints observed and exactly one positive.
    The positive endpoint is compared with its observed negative neighbour;
    ties receive 0.5.  Results are first averaged within patient and then
    across patients, preventing patients with many positives from dominating.
    This is a read-only diagnostic and creates no endpoint pseudo-label.
    """

    expected = (scores.shape[0], N_STANDARD_CHANNELS)
    if scores.ndim != 2 or tuple(scores.shape) != expected:
        raise ValueError("scores must have shape [P,19]")
    if tuple(targets.shape) != expected or tuple(target_mask.shape) != expected:
        raise ValueError("targets and target_mask must have shape [P,19]")
    if not scores.is_floating_point() or not targets.is_floating_point():
        raise TypeError("scores and targets must be floating point")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be bool")
    if scores.device != targets.device or scores.device != target_mask.device:
        raise ValueError("direction diagnostic inputs must share a device")
    observed = targets[target_mask]
    if not torch.isfinite(scores).all() or not torch.isfinite(observed).all() or (
        observed.numel() and not torch.all((observed == 0) | (observed == 1))
    ):
        raise ValueError("direction diagnostic requires finite binary observations")

    patient_values = scores.new_full((scores.shape[0],), torch.nan)
    pair_counts = torch.zeros(
        scores.shape[0], dtype=torch.long, device=scores.device
    )
    for patient in range(scores.shape[0]):
        rows: list[torch.Tensor] = []
        for left, right in TCP_20_EDGES:
            left_index = CHANNEL_INDEX[left]
            right_index = CHANNEL_INDEX[right]
            if not (
                bool(target_mask[patient, left_index])
                and bool(target_mask[patient, right_index])
            ):
                continue
            left_positive = bool(targets[patient, left_index] == 1)
            right_positive = bool(targets[patient, right_index] == 1)
            if left_positive == right_positive:
                continue
            positive = left_index if left_positive else right_index
            negative = right_index if left_positive else left_index
            difference = scores[patient, positive] - scores[patient, negative]
            rows.append(
                torch.where(
                    difference > 0,
                    difference.new_tensor(1.0),
                    torch.where(
                        difference < 0,
                        difference.new_tensor(0.0),
                        difference.new_tensor(0.5),
                    ),
                )
            )
        if rows:
            pair_counts[patient] = len(rows)
            patient_values[patient] = torch.stack(rows).mean()
    eligible = pair_counts > 0
    if not bool(eligible.any()):
        raise ValueError("no informative observed TCP endpoint pair is available")
    return WithinTCPEdgeDirectionMetrics(
        patient_macro_accuracy=float(patient_values[eligible].mean().detach().cpu()),
        eligible_patient_count=int(eligible.sum().item()),
        informative_pair_count=int(pair_counts.sum().item()),
        patient_accuracy=patient_values,
        informative_pair_count_by_patient=pair_counts,
    )


def fit_outer_train_crossfit_endpoint_gate(
    proposal: TCPEndpointFlipProposal,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    anchor_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    scope: str = "outer_train_inner_oof_only",
    alpha: float = ENDPOINT_GATE_ALPHA,
) -> EndpointFlipGate:
    """Fit one fixed global flip switch from outer-train inner-OOF rows.

    There is no threshold grid.  Conditional on decisive proposed flips, a
    one-sided exact sign test asks whether beneficial flips outnumber harmful
    flips.  A second hard prerequisite requires the residual to outperform the
    anchor on patient-macro observed-positive versus observed-negative TCP
    endpoint direction.  The switch is enabled only when both checks pass.
    This is valid for held application only when every score/proposal row was
    generated by models that excluded that row and the eventual outer-held
    patients.
    """

    if type(proposal) is not TCPEndpointFlipProposal:
        raise TypeError("proposal must come from propose_tcp_endpoint_flips")
    patients = int(proposal.proposed.shape[0])
    if tuple(targets.shape) != (patients, N_STANDARD_CHANNELS) or tuple(
        target_mask.shape
    ) != (patients, N_STANDARD_CHANNELS):
        raise ValueError("targets and target_mask must have shape [P,19]")
    if not targets.is_floating_point() or target_mask.dtype != torch.bool:
        raise TypeError("targets must be floating point and target_mask bool")
    if targets.device != proposal.proposed.device or target_mask.device != (
        proposal.proposed.device
    ):
        raise ValueError("gate inputs must share a device")
    observed = targets[target_mask]
    if not torch.isfinite(observed).all() or (
        observed.numel() and not torch.all((observed == 0) | (observed == 1))
    ):
        raise ValueError("observed gate targets must be finite binary values")
    if alpha != ENDPOINT_GATE_ALPHA:
        raise ValueError("endpoint alpha is frozen at 0.05; it is not tunable")
    if scope != "outer_train_inner_oof_only":
        raise ValueError("endpoint gate accepts outer-train inner-OOF rows only")
    _validate_score_triplet(anchor_scores, residual_scores, target_mask)
    expected_proposal = propose_tcp_endpoint_flips(
        anchor_scores, residual_scores, target_mask
    )
    for name in (
        "anchor_index",
        "candidate_index",
        "proposed",
        "residual_margin",
    ):
        if not torch.equal(getattr(proposal, name), getattr(expected_proposal, name)):
            raise ValueError("endpoint proposal does not replay from cross-fit scores")
    anchor_direction = within_tcp_edge_direction_metrics(
        anchor_scores, targets, target_mask
    )
    residual_direction = within_tcp_edge_direction_metrics(
        residual_scores, targets, target_mask
    )
    direction_pass = (
        residual_direction.patient_macro_accuracy
        > anchor_direction.patient_macro_accuracy
    )

    beneficial = 0
    harmful = 0
    neutral = 0
    for patient in torch.nonzero(proposal.proposed, as_tuple=False).flatten().tolist():
        anchor = int(proposal.anchor_index[patient].item())
        candidate = int(proposal.candidate_index[patient].item())
        if anchor < 0 or candidate < 0 or not (
            bool(target_mask[patient, anchor]) and bool(target_mask[patient, candidate])
        ):
            raise ValueError("a proposed endpoint lacks an observed cross-fit target")
        anchor_positive = bool(targets[patient, anchor] == 1)
        candidate_positive = bool(targets[patient, candidate] == 1)
        if candidate_positive and not anchor_positive:
            beneficial += 1
        elif anchor_positive and not candidate_positive:
            harmful += 1
        else:
            neutral += 1
    p_value = _one_sided_exact_sign_p_value(beneficial, harmful)
    enabled = (
        direction_pass
        and (beneficial > harmful)
        and (p_value <= ENDPOINT_GATE_ALPHA)
    )
    return EndpointFlipGate(
        enabled=enabled,
        proposal_count=int(proposal.proposed.sum().item()),
        beneficial_count=beneficial,
        harmful_count=harmful,
        neutral_count=neutral,
        decisive_count=beneficial + harmful,
        one_sided_exact_p_value=p_value,
        alpha=ENDPOINT_GATE_ALPHA,
        scope=scope,
        within_tcp_anchor_direction_accuracy=(
            anchor_direction.patient_macro_accuracy
        ),
        within_tcp_residual_direction_accuracy=(
            residual_direction.patient_macro_accuracy
        ),
        within_tcp_direction_prerequisite_pass=direction_pass,
    )


@dataclass(frozen=True)
class EndpointFlipOutput:
    scores: torch.Tensor
    applied: torch.Tensor

    @property
    def applied_count(self) -> int:
        return int(self.applied.sum().item())


def apply_crossfit_gated_endpoint_flips(
    anchor_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
    proposal: TCPEndpointFlipProposal,
    gate: EndpointFlipGate,
) -> EndpointFlipOutput:
    """Apply a previously fitted outer-train gate without reading held labels."""

    if anchor_scores.ndim != 2 or tuple(anchor_scores.shape[1:]) != (
        N_STANDARD_CHANNELS,
    ):
        raise ValueError("anchor_scores must have shape [P,19]")
    if tuple(evaluable_mask.shape) != tuple(anchor_scores.shape) or (
        evaluable_mask.dtype != torch.bool
    ):
        raise TypeError("evaluable_mask must be bool [P,19]")
    if type(proposal) is not TCPEndpointFlipProposal or type(gate) is not EndpointFlipGate:
        raise TypeError("endpoint application requires verified proposal and gate objects")
    if tuple(proposal.proposed.shape) != (anchor_scores.shape[0],):
        raise ValueError("proposal patient count differs from held scores")
    if anchor_scores.device != evaluable_mask.device or anchor_scores.device != (
        proposal.proposed.device
    ):
        raise ValueError("endpoint application tensors must share a device")
    if not torch.isfinite(anchor_scores).all():
        raise ValueError("anchor scores must be finite")

    scores = anchor_scores.clone()
    applied = torch.zeros_like(proposal.proposed)
    if not gate.enabled:
        return EndpointFlipOutput(scores=scores, applied=applied)

    original_top_set = _masked_top_set(anchor_scores, evaluable_mask)
    for patient in torch.nonzero(proposal.proposed, as_tuple=False).flatten().tolist():
        anchor = int(proposal.anchor_index[patient].item())
        candidate = int(proposal.candidate_index[patient].item())
        if int(original_top_set[patient].sum().item()) != 1 or not bool(
            original_top_set[patient, anchor]
        ):
            raise ValueError("proposal anchor is not the unique held anchor Top-1")
        if candidate not in TCP_ENDPOINT_NEIGHBOURS[anchor] or not bool(
            evaluable_mask[patient, candidate]
        ):
            raise ValueError("proposal candidate is not an evaluable TCP endpoint")
        anchor_value = scores[patient, anchor].clone()
        candidate_value = scores[patient, candidate].clone()
        scores[patient, anchor] = candidate_value
        scores[patient, candidate] = anchor_value
        applied[patient] = True

    new_top_set = _masked_top_set(scores, evaluable_mask)
    for patient in torch.nonzero(applied, as_tuple=False).flatten().tolist():
        candidate = int(proposal.candidate_index[patient].item())
        if int(new_top_set[patient].sum().item()) != 1 or not bool(
            new_top_set[patient, candidate]
        ):
            raise RuntimeError("endpoint swap failed to assign the proposed unique Top-1")
    return EndpointFlipOutput(scores=scores, applied=applied)


__all__ = [
    "ENDPOINT_GATE_ALPHA",
    "SAFE_ANCHOR_H_RECOVERY_SCHEMA",
    "SAFE_RESIDUAL_BUDGET_FRACTION",
    "TCP_ENDPOINT_NEIGHBOURS",
    "EndpointFlipGate",
    "EndpointFlipOutput",
    "TCPEndpointFlipProposal",
    "Top1SafeResidualOutput",
    "WithinTCPEdgeDirectionMetrics",
    "apply_crossfit_gated_endpoint_flips",
    "fit_outer_train_crossfit_endpoint_gate",
    "prior_cancelled_log_probability_ratio",
    "propose_tcp_endpoint_flips",
    "top1_safe_bounded_residual",
    "within_tcp_edge_direction_metrics",
]
