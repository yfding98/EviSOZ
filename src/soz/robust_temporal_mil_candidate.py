"""Minimal robust patient pooling for the source-train LaBraM SOZ trial.

This module is intentionally independent of the historical v1--v12 runners.
It does not change the temporal-MIL event head.  It only supplies the one
previously untested intervention: compress target-free A/Q evidence to an
event/channel reliability value and use that value when robustly pooling the
complete seizure bag of each patient.

The resulting scores remain scalp-electrode SOZ *candidate* scores.  Temporal
weights, event reliability, and within-patient dispersion are observational
evidence/uncertainty descriptors; none is a cortical onset or propagation
label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import torch

from .aggregation import aggregate_patient_logits
from .development_reasoner import DevelopmentIVEvidenceBatch
from .geometry import N_STANDARD_CHANNELS, N_TIME_TILES
from .temporal_mil_recovery import (
    TemporalMILPatientBatch,
    exact_positive_set_mass_loss,
)


ROBUST_TEMPORAL_MIL_SCHEMA: Final[str] = (
    "soz_labram_robust_complete_patient_bag_v1"
)
EVENT_RELIABILITY_FLOOR: Final[float] = 0.1
WINSOR_LOWER_QUANTILE: Final[float] = 0.1
WINSOR_UPPER_QUANTILE: Final[float] = 0.9

PatientPooling = Literal["equal", "quality_winsorized"]


@dataclass(frozen=True)
class TargetFreeEventReliability:
    """Target-free event/channel quality compressed from A/Q tile evidence."""

    values: torch.Tensor
    observed_tile_fraction: torch.Tensor
    mean_observed_reliability: torch.Tensor

    def __post_init__(self) -> None:
        shape = tuple(self.values.shape)
        if len(shape) != 2 or shape[1] != N_STANDARD_CHANNELS:
            raise ValueError("event reliability must have shape [E,19]")
        for name in (
            "values",
            "observed_tile_fraction",
            "mean_observed_reliability",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f"{name} must be floating point [E,19]")
            if value.requires_grad or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and detached")
        if torch.any(
            (self.values < EVENT_RELIABILITY_FLOOR) | (self.values > 1.0)
        ):
            raise ValueError("event reliability must lie in [0.1,1]")
        for value in (
            self.observed_tile_fraction,
            self.mean_observed_reliability,
        ):
            if torch.any((value < 0.0) | (value > 1.0)):
                raise ValueError("reliability diagnostics must lie in [0,1]")


def compress_target_free_event_reliability(
    evidence: DevelopmentIVEvidenceBatch,
) -> TargetFreeEventReliability:
    """Compress A/Q tiles without consulting DeepSOZ or private labels.

    For each event and physical electrode, the score is

    ``mean(A/Q reliability on available tiles) * available-tile fraction``.

    Availability is measured only inside the event's target-free phase mask.
    A fixed 0.1 floor keeps every seizure in the complete patient bag.  The
    event-level abstention flag is intentionally *not* converted to a learned
    or hand-tuned multiplier; it remains a separately reportable warning.
    """

    if not isinstance(evidence, DevelopmentIVEvidenceBatch):
        raise TypeError("event reliability requires DevelopmentIVEvidenceBatch")
    evidence.validate()
    phase = evidence.phase_mask.unsqueeze(1)
    available = evidence.evolution_mask & phase
    available_count = available.sum(dim=-1)
    phase_count = evidence.phase_mask.sum(dim=-1).clamp_min(1).unsqueeze(1)
    observed_fraction = available_count.to(evidence.reliability.dtype) / (
        phase_count.to(evidence.reliability.dtype)
    )
    total = torch.where(
        available, evidence.reliability, torch.zeros_like(evidence.reliability)
    ).sum(dim=-1)
    mean = total / available_count.clamp_min(1).to(evidence.reliability.dtype)
    mean = torch.where(available_count > 0, mean, torch.zeros_like(mean))
    values = (mean * observed_fraction).clamp(
        min=EVENT_RELIABILITY_FLOOR, max=1.0
    )
    return TargetFreeEventReliability(
        values=values.detach().contiguous(),
        observed_tile_fraction=observed_fraction.detach().contiguous(),
        mean_observed_reliability=mean.detach().contiguous(),
    )


@dataclass(frozen=True)
class CompletePatientBagAggregation:
    """Patient logits plus within-bag observational uncertainty traces."""

    logits: torch.Tensor
    dispersion: torch.Tensor
    event_counts: torch.Tensor
    reliability_sum: torch.Tensor
    effective_event_count: torch.Tensor
    pooling: PatientPooling

    def __post_init__(self) -> None:
        if self.logits.ndim != 2 or self.logits.shape[1] != N_STANDARD_CHANNELS:
            raise ValueError("patient logits must have shape [P,19]")
        patients = int(self.logits.shape[0])
        expected_matrix = (patients, N_STANDARD_CHANNELS)
        for name in (
            "dispersion",
            "reliability_sum",
            "effective_event_count",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != expected_matrix:
                raise ValueError(f"{name} must have shape [P,19]")
        if tuple(self.event_counts.shape) != (patients,) or (
            self.event_counts.dtype != torch.long
        ):
            raise TypeError("event_counts must be long [P]")
        for value in (
            self.logits,
            self.dispersion,
            self.reliability_sum,
            self.effective_event_count,
        ):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("patient aggregation tensors must be finite floats")
        if torch.any(self.event_counts < 1):
            raise ValueError("every patient must retain a complete non-empty bag")
        if torch.any(self.effective_event_count < 1.0 - 1e-6):
            raise ValueError("effective event count cannot be below one")
        if self.pooling not in ("equal", "quality_winsorized"):
            raise ValueError("unsupported patient pooling mode")


def _validate_event_inputs(
    event_logits: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
    event_reliability: torch.Tensor,
) -> None:
    if event_logits.ndim != 2 or event_logits.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("event_logits must have shape [E,19]")
    events = int(event_logits.shape[0])
    if events < 1 or not event_logits.is_floating_point() or not torch.isfinite(
        event_logits
    ).all():
        raise ValueError("event logits must be non-empty finite floating point")
    if tuple(event_patient_index.shape) != (events,) or (
        event_patient_index.dtype != torch.long
    ):
        raise TypeError("event_patient_index must be long [E]")
    if event_patient_index.device != event_logits.device:
        raise ValueError("event logits and patient index must share a device")
    if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer")
    if int(event_patient_index.min()) != 0 or int(event_patient_index.max()) != (
        n_patients - 1
    ) or int(torch.unique(event_patient_index).numel()) != n_patients:
        raise ValueError("patient index must be a contiguous complete roster")
    if tuple(event_reliability.shape) != tuple(event_logits.shape) or (
        not event_reliability.is_floating_point()
    ):
        raise ValueError("event_reliability must be floating point [E,19]")
    if event_reliability.device != event_logits.device:
        raise ValueError("event reliability and logits must share a device")
    if event_reliability.requires_grad or not torch.isfinite(
        event_reliability
    ).all():
        raise ValueError("event reliability must be finite and detached")
    if torch.any((event_reliability < 0.0) | (event_reliability > 1.0)):
        raise ValueError("event reliability must lie in [0,1]")


def aggregate_complete_patient_bags(
    event_logits: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
    event_reliability: torch.Tensor,
    *,
    pooling: PatientPooling,
) -> CompletePatientBagAggregation:
    """Pool all seizure events without best-event selection.

    ``equal`` is the matched complete-bag arithmetic mean.  The sole new
    candidate, ``quality_winsorized``, clamps each patient/channel event bag
    to its 10th/90th empirical quantiles when at least three events exist and
    then takes a fixed target-free reliability-weighted mean.  No event is
    removed, and patients remain equally weighted by the listwise objective.
    """

    if pooling not in ("equal", "quality_winsorized"):
        raise ValueError("pooling must be 'equal' or 'quality_winsorized'")
    _validate_event_inputs(
        event_logits, event_patient_index, n_patients, event_reliability
    )
    if pooling == "equal":
        weights = torch.ones_like(event_reliability)
        historical_equal = aggregate_patient_logits(
            event_logits, event_patient_index
        )
        expected_patient_ids = torch.arange(
            n_patients, dtype=torch.long, device=event_patient_index.device
        )
        if not torch.equal(historical_equal.patient_ids, expected_patient_ids):
            raise RuntimeError("historical equal aggregation patient order changed")
    else:
        weights = event_reliability.clamp_min(EVENT_RELIABILITY_FLOOR)
        historical_equal = None

    rows: list[torch.Tensor] = []
    dispersions: list[torch.Tensor] = []
    counts: list[int] = []
    weight_sums: list[torch.Tensor] = []
    effective_counts: list[torch.Tensor] = []
    for patient in range(n_patients):
        selected = event_patient_index == patient
        values = event_logits[selected]
        patient_weights = weights[selected]
        count = int(values.shape[0])
        if pooling == "quality_winsorized" and count >= 3:
            lower = torch.quantile(values, WINSOR_LOWER_QUANTILE, dim=0)
            upper = torch.quantile(values, WINSOR_UPPER_QUANTILE, dim=0)
            values = torch.minimum(torch.maximum(values, lower), upper)
        denominator = patient_weights.sum(dim=0).clamp_min(1e-6)
        mean = (
            historical_equal.logits[patient]
            if historical_equal is not None
            else (values * patient_weights).sum(dim=0) / denominator
        )
        variance = (
            (values - mean.unsqueeze(0)).square() * patient_weights
        ).sum(dim=0) / denominator
        effective = denominator.square() / patient_weights.square().sum(
            dim=0
        ).clamp_min(1e-6)
        rows.append(mean)
        dispersions.append(variance.clamp_min(0.0).sqrt())
        counts.append(count)
        weight_sums.append(denominator)
        effective_counts.append(effective)
    pooled_logits = (
        historical_equal.logits
        if historical_equal is not None
        else torch.stack(rows).contiguous()
    )
    result = CompletePatientBagAggregation(
        logits=pooled_logits,
        dispersion=torch.stack(dispersions).contiguous(),
        event_counts=torch.tensor(
            counts, dtype=torch.long, device=event_logits.device
        ),
        reliability_sum=torch.stack(weight_sums).contiguous(),
        effective_event_count=torch.stack(effective_counts).contiguous(),
        pooling=pooling,
    )
    expected_counts = torch.bincount(event_patient_index, minlength=n_patients)
    if not torch.equal(result.event_counts, expected_counts):
        raise RuntimeError("complete patient bag lost or duplicated an event")
    if pooling == "equal":
        assert historical_equal is not None
        if result.logits.data_ptr() != historical_equal.logits.data_ptr():
            raise RuntimeError("equal pooling no longer directly reuses historical mean")
    return result


@dataclass(frozen=True)
class PositiveSetMILObjective:
    """Pure listwise partial-label objective and its patient aggregation."""

    total: torch.Tensor
    positive_set_mass: torch.Tensor
    aggregation: CompletePatientBagAggregation


def positive_set_mil_objective(
    event_logits: torch.Tensor,
    batch: TemporalMILPatientBatch,
    event_reliability: torch.Tensor,
    *,
    pooling: PatientPooling,
) -> PositiveSetMILObjective:
    """Optimize only known-positive set mass, never per-channel zero BCE."""

    if not isinstance(batch, TemporalMILPatientBatch):
        raise TypeError("positive-set MIL requires TemporalMILPatientBatch")
    aggregation = aggregate_complete_patient_bags(
        event_logits,
        batch.event_patient_index,
        len(batch.patient_ids),
        event_reliability,
        pooling=pooling,
    )
    set_mass = exact_positive_set_mass_loss(
        aggregation.logits, batch.targets, batch.target_mask
    )
    return PositiveSetMILObjective(
        total=set_mass,
        positive_set_mass=set_mass,
        aggregation=aggregation,
    )


@dataclass(frozen=True)
class ObservationalPatientUncertainty:
    """Non-Bayesian uncertainty traces suitable for selective reporting."""

    normalized_ranking_entropy: torch.Tensor
    top1_margin: torch.Tensor
    mean_event_dispersion: torch.Tensor
    event_top1_disagreement: torch.Tensor
    mean_effective_event_count: torch.Tensor

    def __post_init__(self) -> None:
        patients = int(self.normalized_ranking_entropy.numel())
        for name in (
            "normalized_ranking_entropy",
            "top1_margin",
            "mean_event_dispersion",
            "event_top1_disagreement",
            "mean_effective_event_count",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != (patients,) or not value.is_floating_point():
                raise ValueError(f"{name} must be floating point [P]")
            if value.requires_grad or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and detached")
        if torch.any(
            (self.normalized_ranking_entropy < -1e-6)
            | (self.normalized_ranking_entropy > 1.0 + 1e-6)
            | (self.event_top1_disagreement < -1e-6)
            | (self.event_top1_disagreement > 1.0 + 1e-6)
        ):
            raise ValueError("entropy/disagreement uncertainty must lie in [0,1]")


def observational_patient_uncertainty(
    event_logits: torch.Tensor,
    event_patient_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    aggregation: CompletePatientBagAggregation,
) -> ObservationalPatientUncertainty:
    """Summarize ranking ambiguity and across-event disagreement.

    ``candidate_mask`` is one deployment-time constant ``[19]`` (in the
    present experiment all standard electrodes except PZ), never a
    patient-specific target-availability mask.  Target values and target-mask
    rows are therefore unreachable here.  These quantities must not be called
    posterior epistemic uncertainty.
    """

    patients = int(aggregation.logits.shape[0])
    if tuple(candidate_mask.shape) != (N_STANDARD_CHANNELS,) or (
        candidate_mask.dtype != torch.bool
    ):
        raise TypeError("candidate_mask must be one fixed bool [19] carrier")
    if candidate_mask.device != event_logits.device:
        raise ValueError("uncertainty inputs must share a device")
    if int(candidate_mask.sum()) < 2:
        raise ValueError("uncertainty requires at least two fixed candidates")
    entropy_rows: list[torch.Tensor] = []
    margin_rows: list[torch.Tensor] = []
    disagreement_rows: list[torch.Tensor] = []
    for patient in range(patients):
        row = aggregation.logits[patient, candidate_mask]
        if row.numel() < 2:
            raise ValueError("uncertainty requires at least two candidates")
        probability = torch.softmax(row, dim=0)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum()
        entropy_rows.append(entropy / torch.log(row.new_tensor(float(row.numel()))))
        ordered = torch.topk(row, k=2).values
        margin_rows.append(ordered[0] - ordered[1])
        patient_top = int(torch.argmax(row).item())
        event_rows = event_logits[event_patient_index == patient][
            :, candidate_mask
        ]
        event_top = torch.argmax(event_rows, dim=1)
        disagreement_rows.append((event_top != patient_top).float().mean())
    return ObservationalPatientUncertainty(
        normalized_ranking_entropy=torch.stack(entropy_rows).detach().contiguous(),
        top1_margin=torch.stack(margin_rows).detach().contiguous(),
        mean_event_dispersion=aggregation.dispersion[:, candidate_mask]
        .mean(dim=1)
        .detach()
        .contiguous(),
        event_top1_disagreement=torch.stack(disagreement_rows)
        .detach()
        .contiguous(),
        mean_effective_event_count=aggregation.effective_event_count[
            :, candidate_mask
        ]
        .mean(dim=1)
        .detach()
        .contiguous(),
    )


def restore_block9_physical_tokens(prefix: torch.Tensor) -> torch.Tensor:
    """Restore detached block-9 cache to ``[E,19,15,4,200]`` for direct control."""

    if prefix.ndim != 4 or tuple(prefix.shape[1:]) != (15, 77, 200):
        raise ValueError("block-9 prefix must have shape [E,15,77,200]")
    if prefix.shape[0] < 1 or not prefix.is_floating_point() or not torch.isfinite(
        prefix
    ).all():
        raise ValueError("block-9 prefix must be non-empty finite floating point")
    if prefix.requires_grad:
        raise ValueError("block-9 prefix must remain detached")
    events = int(prefix.shape[0])
    result = (
        prefix[:, :, 1:, :]
        .reshape(events, N_TIME_TILES, N_STANDARD_CHANNELS, 4, 200)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    return result.detach()


__all__ = [
    "CompletePatientBagAggregation",
    "EVENT_RELIABILITY_FLOOR",
    "ObservationalPatientUncertainty",
    "PositiveSetMILObjective",
    "ROBUST_TEMPORAL_MIL_SCHEMA",
    "TargetFreeEventReliability",
    "WINSOR_LOWER_QUANTILE",
    "WINSOR_UPPER_QUANTILE",
    "aggregate_complete_patient_bags",
    "compress_target_free_event_reliability",
    "observational_patient_uncertainty",
    "positive_set_mil_objective",
    "restore_block9_physical_tokens",
]
