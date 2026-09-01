"""Mask-aware patient-level channel-localization metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS, STANDARD_19


# Exact one-hop physical-electrode neighbourhood used by the published
# DeepSOZ evaluation notebook (MICCAI 2023), expressed in this project's
# modern T7/T8/P7/P8 aliases.  It is intentionally an evaluation sensitivity
# analysis, not a training target and not a statement that adjacent electrodes
# share the same biological SOZ.
DEEPSOZ_STANDARD19_NEIGHBORS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4),
    (0, 4, 5, 6),
    (0, 3, 4, 7, 8),
    (0, 2, 4, 8, 9),
    (0, 1, 3, 5, 9),
    (1, 4, 6, 9, 10),
    (1, 4, 5, 10, 11),
    (2, 8, 12, 13, 17),
    (2, 3, 4, 7, 9, 12, 13, 14),
    (3, 4, 5, 8, 10, 13, 14, 15),
    (4, 5, 6, 9, 11, 14, 15, 16),
    (6, 10, 15, 16, 18),
    (7, 8, 13, 17),
    (7, 8, 9, 12, 14, 17),
    (8, 9, 10, 13, 15, 17, 18),
    (9, 10, 11, 14, 16, 18),
    (10, 11, 15, 18),
    (7, 12, 13, 14, 18),
    (11, 14, 15, 16, 17),
)


@dataclass(frozen=True)
class Top1LocalizationMetrics:
    """Strict and DeepSOZ-style one-hop top-1 localization endpoints.

    ``relaxed_accuracy`` follows the published DeepSOZ convention that a
    one-hop neighbour is accepted only when a reference contains at most four
    positive electrodes.  When explicit spread labels are available, known
    spread electrodes are removed from the relaxed acceptable set and are
    reported separately; propagation is never relabelled as SOZ.
    """

    n_samples: int
    strict_accuracy: float
    relaxed_accuracy: float
    neighbor_only_accuracy_gain: float
    spread_top1_rate: float
    n_neighbor_eligible_samples: int


def deepsoz_style_top1_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    spread_targets: torch.Tensor | None = None,
    spread_mask: torch.Tensor | None = None,
    max_positive_for_neighbor: int = 4,
) -> Top1LocalizationMetrics:
    """Evaluate strict top-1 and the published DeepSOZ one-hop sensitivity.

    Exact score ties are evaluated as the expectation under uniform random
    tie breaking.  Predictions are restricted to target-evaluable electrodes.
    ``spread_targets`` is optional for public DeepSOZ, but should be supplied
    for the private cohort so a known propagated electrode cannot become a
    neighbour-relaxed success.
    """

    _validate_metric_inputs(logits, targets, target_mask)
    if isinstance(max_positive_for_neighbor, bool) or max_positive_for_neighbor < 1:
        raise ValueError("max_positive_for_neighbor must be a positive integer")
    if (spread_targets is None) != (spread_mask is None):
        raise ValueError("spread_targets and spread_mask must be provided together")
    if spread_targets is None:
        spread_targets = torch.zeros_like(targets)
        spread_mask = torch.zeros_like(target_mask)
    assert spread_mask is not None
    if tuple(spread_targets.shape) != tuple(targets.shape):
        raise ValueError("spread_targets must have shape [P,19]")
    if tuple(spread_mask.shape) != tuple(target_mask.shape):
        raise ValueError("spread_mask must have shape [P,19]")
    if spread_mask.dtype != torch.bool:
        raise TypeError("spread_mask must be torch.bool")
    observed_spread = spread_targets[spread_mask]
    if not torch.isfinite(observed_spread).all() or (
        observed_spread.numel()
        and not torch.all((observed_spread == 0) | (observed_spread == 1))
    ):
        raise ValueError("Observed spread targets must be finite binary values")
    if bool((((targets == 1) & target_mask) & ((spread_targets == 1) & spread_mask)).any()):
        raise ValueError("An electrode cannot be both significant/SOZ and spread")

    strict_rows: list[torch.Tensor] = []
    relaxed_rows: list[torch.Tensor] = []
    spread_rows: list[torch.Tensor] = []
    neighbor_eligible = 0
    for sample_index in range(logits.shape[0]):
        evaluable = target_mask[sample_index]
        evaluable_indices = torch.nonzero(evaluable, as_tuple=False).flatten()
        sample_logits = logits[sample_index, evaluable_indices]
        top_value = sample_logits.max()
        tied_global_indices = evaluable_indices[sample_logits == top_value]

        positive = (targets[sample_index] == 1) & evaluable
        positive_count = int(positive.sum().item())
        acceptable = positive.clone()
        if positive_count <= max_positive_for_neighbor:
            neighbor_eligible += 1
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                acceptable[list(DEEPSOZ_STANDARD19_NEIGHBORS[index])] = True
            acceptable &= evaluable

        known_spread = (spread_targets[sample_index] == 1) & spread_mask[sample_index]
        acceptable &= ~known_spread
        strict_rows.append(positive[tied_global_indices].float().mean())
        relaxed_rows.append(acceptable[tied_global_indices].float().mean())
        spread_rows.append(known_spread[tied_global_indices].float().mean())

    strict = float(torch.stack(strict_rows).mean().detach().cpu())
    relaxed = float(torch.stack(relaxed_rows).mean().detach().cpu())
    spread = float(torch.stack(spread_rows).mean().detach().cpu())
    return Top1LocalizationMetrics(
        n_samples=int(logits.shape[0]),
        strict_accuracy=strict,
        relaxed_accuracy=relaxed,
        neighbor_only_accuracy_gain=relaxed - strict,
        spread_top1_rate=spread,
        n_neighbor_eligible_samples=neighbor_eligible,
    )


if len(DEEPSOZ_STANDARD19_NEIGHBORS) != N_STANDARD_CHANNELS:
    raise RuntimeError("DeepSOZ neighbour graph must cover standard-19")
if any(
    index in neighbours
    or any(other < 0 or other >= N_STANDARD_CHANNELS for other in neighbours)
    for index, neighbours in enumerate(DEEPSOZ_STANDARD19_NEIGHBORS)
):
    raise RuntimeError("DeepSOZ neighbour graph contains an invalid endpoint")
# The published notebook's table is not perfectly symmetric.  Preserve it
# exactly because this function is a reproduction/sensitivity endpoint: rows
# are indexed by the true electrode and list accepted predicted electrodes.
if tuple(CHANNEL_INDEX[channel] for channel in STANDARD_19) != tuple(range(19)):
    raise RuntimeError("DeepSOZ neighbour graph requires canonical standard-19 order")


@dataclass(frozen=True)
class PatientLocalizationMetrics:
    n_patients: int
    macro_average_precision: float
    mean_reciprocal_rank: float
    hit_at_k: dict[int, float]
    positive_recall_at_k: dict[int, float]
    brier: float
    nll: float


def _validate_metric_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    expected = (logits.shape[0], N_STANDARD_CHANNELS)
    if logits.ndim != 2 or tuple(logits.shape) != expected:
        raise ValueError("logits must have shape [P,19]")
    if tuple(targets.shape) != expected or tuple(target_mask.shape) != expected:
        raise ValueError("targets and target_mask must have shape [P,19]")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be torch.bool")
    if logits.shape[0] < 1:
        raise ValueError("At least one patient is required")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets[target_mask]).all():
        raise ValueError("Observed metric inputs must be finite")
    observed = targets[target_mask]
    if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("Observed targets must be binary")
    positives = ((targets == 1) & target_mask).sum(dim=1)
    if (positives == 0).any():
        bad = (positives == 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Metrics require an observed in-head positive: {bad}")


def patient_localization_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    k_values: Sequence[int] = (1, 3),
) -> PatientLocalizationMetrics:
    """Compute macro ranking and calibration metrics, one row per patient."""

    _validate_metric_inputs(logits, targets, target_mask)
    ks = tuple(sorted({int(value) for value in k_values}))
    if not ks or any(value < 1 for value in ks):
        raise ValueError("k_values must contain positive integers")

    average_precisions: list[torch.Tensor] = []
    reciprocal_ranks: list[torch.Tensor] = []
    brier_rows: list[torch.Tensor] = []
    nll_rows: list[torch.Tensor] = []
    hits: dict[int, list[torch.Tensor]] = {k: [] for k in ks}
    recalls: dict[int, list[torch.Tensor]] = {k: [] for k in ks}
    for patient_index in range(logits.shape[0]):
        observed = target_mask[patient_index]
        patient_logits = logits[patient_index][observed]
        patient_targets = targets[patient_index][observed]
        order = torch.argsort(patient_logits, descending=True, stable=True)
        ranked_logits = patient_logits[order]
        ranked_targets = patient_targets[order]
        positive_count = int(ranked_targets.sum().item())

        # Exact ties must not inherit the arbitrary canonical electrode order.
        # We report the exact expectation under a uniform random permutation
        # inside each equal-score block.  This makes constant/prevalence
        # baselines and quantized model outputs invariant to channel ordering.
        groups: list[tuple[int, int]] = []
        start = 0
        while start < ranked_logits.numel():
            stop = start + 1
            while stop < ranked_logits.numel() and bool(
                ranked_logits[stop] == ranked_logits[start]
            ):
                stop += 1
            groups.append((stop - start, int(ranked_targets[start:stop].sum().item())))
            start = stop

        expected_ap_numerator = 0.0
        higher_count = 0
        higher_positive = 0
        expected_rr: float | None = None
        for group_size, group_positive in groups:
            if group_positive:
                positive_fraction = group_positive / group_size
                for offset in range(1, group_size + 1):
                    expected_positive_prefix_term = positive_fraction
                    if group_size > 1:
                        expected_positive_prefix_term += (
                            (offset - 1)
                            * group_positive
                            * (group_positive - 1)
                            / (group_size * (group_size - 1))
                        )
                    expected_ap_numerator += (
                        higher_positive * positive_fraction
                        + expected_positive_prefix_term
                    ) / (higher_count + offset)

                if expected_rr is None:
                    denominator = math.comb(group_size, group_positive)
                    expected_rr = sum(
                        (
                            math.comb(group_size - offset, group_positive - 1)
                            / denominator
                        )
                        / (higher_count + offset)
                        for offset in range(
                            1, group_size - group_positive + 2
                        )
                    )
            higher_count += group_size
            higher_positive += group_positive

        if expected_rr is None:  # guarded by _validate_metric_inputs
            raise RuntimeError("Tie-aware MRR lost the observed positive")
        average_precisions.append(
            patient_logits.new_tensor(expected_ap_numerator / positive_count)
        )
        reciprocal_ranks.append(patient_logits.new_tensor(expected_rr))
        for k in ks:
            remaining = min(k, int(ranked_targets.numel()))
            selected_positive = 0.0
            expected_hit = 0.0
            for group_size, group_positive in groups:
                if remaining >= group_size:
                    selected_positive += group_positive
                    remaining -= group_size
                    if selected_positive > 0:
                        expected_hit = 1.0
                    if remaining == 0:
                        break
                    continue

                if remaining > 0:
                    selected_positive += remaining * group_positive / group_size
                    if selected_positive - remaining * group_positive / group_size > 0:
                        expected_hit = 1.0
                    elif group_positive > 0:
                        probability_no_positive = (
                            math.comb(group_size - group_positive, remaining)
                            / math.comb(group_size, remaining)
                            if remaining <= group_size - group_positive
                            else 0.0
                        )
                        expected_hit = 1.0 - probability_no_positive
                break
            hits[k].append(patient_logits.new_tensor(expected_hit))
            recalls[k].append(
                patient_logits.new_tensor(selected_positive / positive_count)
            )

        probabilities = patient_logits.sigmoid()
        brier_rows.append((probabilities - patient_targets).square().mean())
        nll_rows.append(
            F.binary_cross_entropy_with_logits(
                patient_logits, patient_targets, reduction="mean"
            )
        )

    def scalar_mean(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().detach().cpu())

    return PatientLocalizationMetrics(
        n_patients=int(logits.shape[0]),
        macro_average_precision=scalar_mean(average_precisions),
        mean_reciprocal_rank=scalar_mean(reciprocal_ranks),
        hit_at_k={k: scalar_mean(values) for k, values in hits.items()},
        positive_recall_at_k={k: scalar_mean(values) for k, values in recalls.items()},
        brier=scalar_mean(brier_rows),
        nll=scalar_mean(nll_rows),
    )
