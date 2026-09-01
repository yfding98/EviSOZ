"""Hemisphere-balanced hierarchical set reasoner for the block-9 carrier.

HB-HSR is the single recovery candidate frozen by the v11.1 failure audit.  It
does not alter LaBraM or the fold-fitted feature transform.  For patient ``p``
and candidate channel ``c``, the shared low-capacity scorer is

``r[p,c] = JeffreysPrior[c] + H16[p,c] @ w_H + Fine20[p,c] @ w_F``.

The fixed 18 candidates (canonical PZ is unavailable) are partitioned into
left, right, and midline/central side sets.  The deployable probability is
factorized without a hard side gate:

``p(c) = p(side(c)) * p(c | side(c))``.

``p(side)`` is a softmax over side-wise ``logsumexp(r)``.  Within each side,
outer-training positive-patient support defines

``q[c|side] proportional to (support[c] + 0.5) ** -0.5``.

Crucially, ``q`` is normalized separately inside every side and conditional
probabilities use ``softmax(r + log(q))``.  Thus both the positive numerator
and full-side denominator contain the same prior; no weight can create more
than one unit of probability mass.  The exact same balanced hierarchical
channel log-probability is returned for deployment ranking.

The loss is an unscaled 1:1 sum.  Its side term is negative log probability
mass on all gold sides.  Its conditional term averages, equally across the
patient's gold sides, negative log conditional mass on positive channels in
that side.  This handles bilateral/midline positive sets without choosing one
side and never treats the side prediction as a hard routing decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import torch
import torch.nn as nn

from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS, STANDARD_19
from .v11_reasoner import (
    TransformedPatientFeatures,
    V11_CANDIDATE_MASK,
    V11_FINE_DIM,
    V11_H_PCA_DIM,
    jeffreys_reference_prior_logits,
)


HB_HSR_SIDE_NAMES: Final[tuple[str, ...]] = ("L", "R", "M")
HB_HSR_SIDE_CHANNELS: Final[tuple[tuple[str, ...], ...]] = (
    ("FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"),
    ("FP2", "F4", "F8", "C4", "T8", "P4", "P8", "O2"),
    ("FZ", "CZ"),
)
HB_HSR_SIDE_INDICES: Final[tuple[tuple[int, ...], ...]] = tuple(
    tuple(CHANNEL_INDEX[channel] for channel in channels)
    for channels in HB_HSR_SIDE_CHANNELS
)
HB_HSR_PZ_INDEX: Final[int] = CHANNEL_INDEX["PZ"]


def _build_channel_to_side() -> torch.Tensor:
    result = torch.full((N_STANDARD_CHANNELS,), -1, dtype=torch.long)
    for side, indices in enumerate(HB_HSR_SIDE_INDICES):
        for index in indices:
            if result[index].item() != -1:
                raise RuntimeError("HB-HSR side sets overlap")
            result[index] = side
    if result[HB_HSR_PZ_INDEX].item() != -1:
        raise RuntimeError("PZ cannot enter an HB-HSR side set")
    candidate_mask = V11_CANDIDATE_MASK
    if not torch.equal(result >= 0, candidate_mask):
        missing = tuple(
            STANDARD_19[index]
            for index in range(N_STANDARD_CHANNELS)
            if bool(candidate_mask[index]) != bool(result[index] >= 0)
        )
        raise RuntimeError(f"HB-HSR side sets do not partition fixed candidates: {missing}")
    return result


HB_HSR_CHANNEL_TO_SIDE: Final[torch.Tensor] = _build_channel_to_side()


def _validate_target_shapes(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise TypeError("targets must be a floating-point tensor")
    if not isinstance(target_mask, torch.Tensor) or target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be a torch.bool tensor")
    if targets.ndim != 2 or targets.shape[1] != N_STANDARD_CHANNELS or (
        tuple(target_mask.shape) != tuple(targets.shape)
    ):
        raise ValueError("targets and target_mask must have aligned shape [P,19]")
    fixed = V11_CANDIDATE_MASK.to(target_mask.device).expand_as(target_mask)
    if not torch.equal(target_mask, fixed):
        raise ValueError("HB-HSR requires the fixed 18-candidate mask with PZ excluded")


def _validate_observed_binary(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    observed = targets[target_mask]
    if not torch.isfinite(observed).all() or not (
        ((observed == 0) | (observed == 1)).all()
    ):
        raise ValueError("observed targets must be finite binary values")


@dataclass(frozen=True)
class FoldLocalHBHSRPriors:
    """Outer-training-only priors used by one HB-HSR fold.

    ``conditional_channel_prior`` is a proper categorical prior within every
    side, not an unnormalized loss multiplier.  PZ has exactly zero mass.
    ``outer_train_patient_indices`` records which rows were permitted to
    contribute support and the Jeffreys reference prior.
    """

    reference_prior_logits: torch.Tensor
    positive_patient_support: torch.Tensor
    conditional_channel_prior: torch.Tensor
    outer_train_patient_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if tuple(self.reference_prior_logits.shape) != (N_STANDARD_CHANNELS,) or (
            not self.reference_prior_logits.is_floating_point()
        ) or not torch.isfinite(self.reference_prior_logits).all():
            raise ValueError("reference_prior_logits must be finite floating [19]")
        if tuple(self.positive_patient_support.shape) != (N_STANDARD_CHANNELS,) or (
            self.positive_patient_support.dtype != torch.long
        ) or torch.any(self.positive_patient_support < 0):
            raise ValueError("positive_patient_support must be nonnegative long [19]")
        prior = self.conditional_channel_prior
        if tuple(prior.shape) != (N_STANDARD_CHANNELS,) or (
            not prior.is_floating_point()
        ) or not torch.isfinite(prior).all() or torch.any(prior < 0):
            raise ValueError("conditional_channel_prior must be finite nonnegative [19]")
        if self.positive_patient_support[HB_HSR_PZ_INDEX].item() != 0 or (
            prior[HB_HSR_PZ_INDEX].item() != 0.0
        ):
            raise ValueError("PZ support and conditional mass must remain zero")
        for indices in HB_HSR_SIDE_INDICES:
            index = torch.tensor(indices, dtype=torch.long, device=prior.device)
            mass = prior.index_select(0, index).sum()
            if not torch.allclose(
                mass,
                torch.ones((), dtype=prior.dtype, device=prior.device),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("conditional channel prior must sum to one per side")
        selected = self.outer_train_patient_indices
        if not selected or len(set(selected)) != len(selected) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in selected
        ):
            raise ValueError("outer_train_patient_indices must be unique nonnegative ints")
        for value in (
            self.reference_prior_logits,
            self.positive_patient_support,
            self.conditional_channel_prior,
        ):
            if value.requires_grad:
                raise ValueError("fold-local priors must be detached")


def fit_fold_local_hb_hsr_priors(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    outer_train_patient_indices: Sequence[int],
) -> FoldLocalHBHSRPriors:
    """Fit Jeffreys and inverse-sqrt priors from outer-training patients only.

    Held-out target values are never inspected.  The full tensors are accepted
    only so a runner can provide stable patient indices; selection occurs
    before finite/binary target validation or any statistic is calculated.
    """

    _validate_target_shapes(targets, target_mask)
    selected = tuple(int(value) for value in outer_train_patient_indices)
    patients = int(targets.shape[0])
    if not selected or len(set(selected)) != len(selected) or any(
        isinstance(value, bool) or value < 0 or value >= patients
        for value in selected
    ):
        raise ValueError("outer_train_patient_indices must be unique valid rows")
    index = torch.tensor(selected, dtype=torch.long, device=targets.device)
    train_targets = targets.index_select(0, index)
    train_mask = target_mask.index_select(0, index)
    _validate_observed_binary(train_targets, train_mask)

    support = ((train_targets == 1) & train_mask).sum(dim=0).to(torch.long)
    reference = jeffreys_reference_prior_logits(train_targets, train_mask)
    inverse_sqrt = (support.to(torch.float32) + 0.5).rsqrt()
    conditional_prior = torch.zeros(
        N_STANDARD_CHANNELS,
        dtype=torch.float32,
        device=targets.device,
    )
    for indices in HB_HSR_SIDE_INDICES:
        side_index = torch.tensor(indices, dtype=torch.long, device=targets.device)
        side_weight = inverse_sqrt.index_select(0, side_index)
        conditional_prior.index_copy_(
            0,
            side_index,
            side_weight / side_weight.sum(),
        )
    return FoldLocalHBHSRPriors(
        reference_prior_logits=reference.detach().cpu().contiguous(),
        positive_patient_support=support.detach().cpu().contiguous(),
        conditional_channel_prior=conditional_prior.detach().cpu().contiguous(),
        outer_train_patient_indices=selected,
    )


@dataclass(frozen=True)
class HBHSRReasonerOutput:
    """Scores and normalized masses produced by :class:`HBHSRReasoner`.

    ``channel_log_probability`` is the only deployment ranking score.  It is
    the same balanced hierarchical probability used by the training loss,
    sums to one over the fixed 18 candidates, and is ``-inf`` at PZ.
    """

    base_logits: torch.Tensor
    prior_contribution: torch.Tensor
    h_contribution: torch.Tensor
    fine_contribution: torch.Tensor
    side_logits: torch.Tensor
    side_log_probability: torch.Tensor
    conditional_logits: torch.Tensor
    conditional_log_probability: torch.Tensor
    channel_log_probability: torch.Tensor

    @property
    def deployment_scores(self) -> torch.Tensor:
        return self.channel_log_probability

    def reconstructed_base_logits(self) -> torch.Tensor:
        return self.prior_contribution + self.h_contribution + self.fine_contribution


class HBHSRReasoner(nn.Module):
    """Shared H16/fine20 scorer with a soft, balanced L/R/M factorization."""

    def __init__(
        self,
        fold_priors: FoldLocalHBHSRPriors,
        *,
        use_h: bool = True,
        use_fine: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(fold_priors, FoldLocalHBHSRPriors):
            raise TypeError("fold_priors must be FoldLocalHBHSRPriors")
        if not use_h and not use_fine:
            raise ValueError("HB-HSR must use at least one evidence family")
        self.use_h = bool(use_h)
        self.use_fine = bool(use_fine)
        if self.use_h:
            self.h_weight = nn.Parameter(torch.zeros(V11_H_PCA_DIM))
        else:
            self.register_parameter("h_weight", None)
        if self.use_fine:
            self.fine_weight = nn.Parameter(torch.zeros(V11_FINE_DIM))
        else:
            self.register_parameter("fine_weight", None)
        self.register_buffer(
            "reference_prior_logits",
            fold_priors.reference_prior_logits.detach().float().contiguous(),
        )
        self.register_buffer(
            "conditional_channel_prior",
            fold_priors.conditional_channel_prior.detach().float().contiguous(),
        )
        self.register_buffer("candidate_mask", V11_CANDIDATE_MASK.clone())
        self.register_buffer("channel_to_side", HB_HSR_CHANNEL_TO_SIDE.clone())

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(self, evidence: TransformedPatientFeatures) -> HBHSRReasonerOutput:
        if not isinstance(evidence, TransformedPatientFeatures):
            raise TypeError("HB-HSR accepts TransformedPatientFeatures only")
        if evidence.h.device != evidence.fine.device:
            raise ValueError("H and fine features must share a device")
        patients = int(evidence.h.shape[0])
        dtype = evidence.h.dtype
        device = evidence.h.device
        prior = self.reference_prior_logits.to(device=device, dtype=dtype).expand(
            patients, -1
        )
        h = torch.zeros_like(prior)
        fine = torch.zeros_like(prior)
        if self.h_weight is not None:
            h = torch.einsum("pcd,d->pc", evidence.h, self.h_weight.to(dtype=dtype))
        if self.fine_weight is not None:
            fine = torch.einsum(
                "pcd,d->pc",
                evidence.fine,
                self.fine_weight.to(dtype=evidence.fine.dtype),
            ).to(dtype=dtype)
        base = prior + h + fine
        side_logits = torch.stack(
            tuple(
                torch.logsumexp(
                    base.index_select(
                        1,
                        torch.tensor(indices, dtype=torch.long, device=device),
                    ),
                    dim=1,
                )
                for indices in HB_HSR_SIDE_INDICES
            ),
            dim=1,
        )
        side_log_probability = torch.log_softmax(side_logits, dim=1)

        q = self.conditional_channel_prior.to(device=device, dtype=dtype)
        log_q = torch.full_like(q, -torch.inf)
        log_q[self.candidate_mask] = q[self.candidate_mask].log()
        conditional_logits = base + log_q.unsqueeze(0)
        conditional_normalizer = torch.stack(
            tuple(
                torch.logsumexp(
                    conditional_logits.index_select(
                        1,
                        torch.tensor(indices, dtype=torch.long, device=device),
                    ),
                    dim=1,
                )
                for indices in HB_HSR_SIDE_INDICES
            ),
            dim=1,
        )
        safe_side = self.channel_to_side.clamp_min(0)
        conditional_log_probability = conditional_logits - (
            conditional_normalizer.index_select(1, safe_side)
        )
        conditional_log_probability = conditional_log_probability.masked_fill(
            ~self.candidate_mask.unsqueeze(0), -torch.inf
        )
        channel_log_probability = (
            side_log_probability.index_select(1, safe_side)
            + conditional_log_probability
        ).masked_fill(~self.candidate_mask.unsqueeze(0), -torch.inf)

        output = HBHSRReasonerOutput(
            base_logits=base,
            prior_contribution=prior,
            h_contribution=h,
            fine_contribution=fine,
            side_logits=side_logits,
            side_log_probability=side_log_probability,
            conditional_logits=conditional_logits,
            conditional_log_probability=conditional_log_probability,
            channel_log_probability=channel_log_probability,
        )
        if not torch.allclose(
            output.base_logits,
            output.reconstructed_base_logits(),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("HB-HSR base-score decomposition drifted")
        if not torch.allclose(
            torch.logsumexp(output.channel_log_probability, dim=1),
            torch.zeros(patients, dtype=dtype, device=device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("HB-HSR deployment probability is not normalized")
        return output


@dataclass(frozen=True)
class HBHSRLossOutput:
    """Unscaled ``1 * side_set + 1 * channel_given_side_set`` loss."""

    total: torch.Tensor
    side_set: torch.Tensor
    channel_given_side_set: torch.Tensor
    per_patient_side_set: torch.Tensor
    per_patient_channel_given_side_set: torch.Tensor


def hb_hsr_set_mass_loss(
    output: HBHSRReasonerOutput,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> HBHSRLossOutput:
    """Compute soft gold-side and side-conditional positive-set mass losses.

    For a multi-side positive patient, ``side_set`` uses the total probability
    mass of all gold sides.  The conditional loss is computed separately in
    every gold side and then averaged with equal side weight.  Its positive
    numerator and full-side denominator both use ``r + log(q)`` through the
    normalized ``conditional_log_probability`` supplied by the reasoner.
    """

    if not isinstance(output, HBHSRReasonerOutput):
        raise TypeError("output must be HBHSRReasonerOutput")
    _validate_target_shapes(targets, target_mask)
    _validate_observed_binary(targets, target_mask)
    patients = int(targets.shape[0])
    expected_channel = (patients, N_STANDARD_CHANNELS)
    if tuple(output.base_logits.shape) != expected_channel or (
        tuple(output.side_log_probability.shape) != (patients, len(HB_HSR_SIDE_NAMES))
    ) or tuple(output.conditional_log_probability.shape) != expected_channel or (
        tuple(output.channel_log_probability.shape) != expected_channel
    ):
        raise ValueError("HB-HSR output and targets do not share a patient roster")
    device = output.base_logits.device
    if targets.device != device or target_mask.device != device:
        raise ValueError("output, targets, and target_mask must share a device")
    candidate_mask = V11_CANDIDATE_MASK.to(device=device)
    for value, name in (
        (output.base_logits, "base logits"),
        (output.side_log_probability, "side log probability"),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if not torch.isfinite(output.conditional_log_probability[:, candidate_mask]).all():
        raise ValueError("candidate conditional log probabilities must be finite")
    if not torch.isfinite(output.channel_log_probability[:, candidate_mask]).all():
        raise ValueError("candidate channel log probabilities must be finite")
    if not torch.isneginf(output.conditional_log_probability[:, HB_HSR_PZ_INDEX]).all() or (
        not torch.isneginf(output.channel_log_probability[:, HB_HSR_PZ_INDEX]).all()
    ):
        raise ValueError("PZ must be excluded with -inf log probability")
    if not torch.allclose(
        torch.logsumexp(output.side_log_probability, dim=1),
        torch.zeros(patients, device=device, dtype=output.base_logits.dtype),
        atol=1e-5,
        rtol=1e-5,
    ) or not torch.allclose(
        torch.logsumexp(output.channel_log_probability, dim=1),
        torch.zeros(patients, device=device, dtype=output.base_logits.dtype),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("HB-HSR probabilities must be normalized")

    positive = target_mask & (targets == 1)
    side_rows = []
    conditional_rows = []
    for patient in range(patients):
        gold_sides: list[int] = []
        side_conditional_losses = []
        for side, indices in enumerate(HB_HSR_SIDE_INDICES):
            side_index = torch.tensor(indices, dtype=torch.long, device=device)
            positive_in_side = positive[patient].index_select(0, side_index)
            if bool(positive_in_side.any()):
                gold_sides.append(side)
                positive_index = side_index[positive_in_side]
                side_conditional_losses.append(
                    -torch.logsumexp(
                        output.conditional_log_probability[
                            patient
                        ].index_select(0, positive_index),
                        dim=0,
                    )
                )
        if not gold_sides:
            raise ValueError("HB-HSR loss requires at least one positive per patient")
        gold_side_index = torch.tensor(gold_sides, dtype=torch.long, device=device)
        side_rows.append(
            -torch.logsumexp(
                output.side_log_probability[patient].index_select(0, gold_side_index),
                dim=0,
            )
        )
        conditional_rows.append(torch.stack(side_conditional_losses).mean())

    per_patient_side = torch.stack(side_rows)
    per_patient_conditional = torch.stack(conditional_rows)
    side_loss = per_patient_side.mean()
    conditional_loss = per_patient_conditional.mean()
    return HBHSRLossOutput(
        total=side_loss + conditional_loss,
        side_set=side_loss,
        channel_given_side_set=conditional_loss,
        per_patient_side_set=per_patient_side,
        per_patient_channel_given_side_set=per_patient_conditional,
    )


__all__ = [
    "FoldLocalHBHSRPriors",
    "HBHSRLossOutput",
    "HBHSRReasoner",
    "HBHSRReasonerOutput",
    "HB_HSR_CHANNEL_TO_SIDE",
    "HB_HSR_PZ_INDEX",
    "HB_HSR_SIDE_CHANNELS",
    "HB_HSR_SIDE_INDICES",
    "HB_HSR_SIDE_NAMES",
    "fit_fold_local_hb_hsr_priors",
    "hb_hsr_set_mass_loss",
]
