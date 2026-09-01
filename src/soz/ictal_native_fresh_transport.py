"""One-shot patient-fresh transport statistics for native TUSZ involvement.

The endpoint in this module is bipolar edge-time ictal involvement.  Nothing
here accepts or produces an SOZ label.  All summaries are patient-macro and
use only caller-supplied observed cells.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .concept_metrics import IctalConceptMetrics


def paired_patient_improvements(
    independent_logits: torch.Tensor,
    temporal_logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    patient_ids: torch.Tensor,
) -> tuple[dict[str, float | int], ...]:
    """Return per-patient independent-minus-temporal BCE/Brier effects."""

    expected = tuple(targets.shape)
    if (
        targets.ndim != 3
        or expected[1:] != (20, 60)
        or tuple(target_mask.shape) != expected
        or tuple(independent_logits.shape) != (*expected, 1)
        or tuple(temporal_logits.shape) != (*expected, 1)
    ):
        raise ValueError("Fresh-I paired tensors must align as [E,20,60]")
    if target_mask.dtype != torch.bool or patient_ids.dtype != torch.long:
        raise TypeError("Fresh-I mask must be bool and patient_ids must be long")
    if tuple(patient_ids.shape) != (expected[0],):
        raise ValueError("Fresh-I patient_ids must have shape [E]")
    devices = {
        independent_logits.device,
        temporal_logits.device,
        targets.device,
        target_mask.device,
        patient_ids.device,
    }
    if len(devices) != 1:
        raise ValueError("Fresh-I paired tensors must share one device")
    if not torch.isfinite(independent_logits).all() or not torch.isfinite(
        temporal_logits
    ).all():
        raise ValueError("Fresh-I logits must be finite")

    independent = independent_logits.squeeze(-1)
    temporal = temporal_logits.squeeze(-1)
    rows: list[dict[str, float | int]] = []
    for patient_id in torch.unique(patient_ids, sorted=True):
        selected = patient_ids == patient_id
        mask = target_mask[selected]
        if not mask.any():
            raise ValueError("Every frozen fresh-I patient must retain observed cells")
        truth = targets[selected][mask]
        independent_patient = independent[selected][mask]
        temporal_patient = temporal[selected][mask]
        independent_bce = F.binary_cross_entropy_with_logits(
            independent_patient, truth
        )
        temporal_bce = F.binary_cross_entropy_with_logits(temporal_patient, truth)
        independent_brier = ((independent_patient.sigmoid() - truth) ** 2).mean()
        temporal_brier = ((temporal_patient.sigmoid() - truth) ** 2).mean()
        positive = int(truth.sum().item())
        observed = int(truth.numel())
        rows.append(
            {
                "patient_index": int(patient_id.item()),
                "observed_labels": observed,
                "positive_labels": positive,
                "explicit_negative_labels": observed - positive,
                "bce_improvement": float((independent_bce - temporal_bce).item()),
                "brier_improvement": float(
                    (independent_brier - temporal_brier).item()
                ),
            }
        )
    if not rows:
        raise ValueError("Fresh-I paired evaluation contains no patients")
    return tuple(rows)


def patient_bootstrap_interval(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Deterministic ordinary percentile interval over frozen patients."""

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("Bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Bootstrap seed must be a non-negative integer")
    tensor = torch.tensor(tuple(float(value) for value in values), dtype=torch.float64)
    if tensor.numel() < 2 or not torch.isfinite(tensor).all():
        raise ValueError("Bootstrap requires at least two finite patient effects")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(
        0,
        tensor.numel(),
        (replicates, tensor.numel()),
        generator=generator,
    )
    means = tensor[indices].mean(dim=1)
    return {
        "patient_count": int(tensor.numel()),
        "replicates": replicates,
        "seed": seed,
        "mean": float(tensor.mean().item()),
        "lower_95": float(torch.quantile(means, 0.025).item()),
        "upper_95": float(torch.quantile(means, 0.975).item()),
        "interval": "ordinary_patient_bootstrap_percentile_uncorrected",
    }


def decide_native_fresh_transport(
    *,
    independent: IctalConceptMetrics,
    temporal: IctalConceptMetrics,
    time_only: IctalConceptMetrics,
    mask_only: IctalConceptMetrics,
    prevalence: IctalConceptMetrics,
    paired_bce_interval: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    """Apply the frozen all-or-none source-task transport qualification."""

    named = {
        "independent": independent,
        "temporal": temporal,
        "time_only": time_only,
        "mask_only": mask_only,
        "prevalence": prevalence,
    }
    reference = temporal
    for name, metrics in named.items():
        if (
            metrics.n_patients != reference.n_patients
            or metrics.n_observed_labels != reference.n_observed_labels
            or metrics.n_positive_labels != reference.n_positive_labels
            or metrics.n_negative_labels != reference.n_negative_labels
        ):
            raise ValueError(f"Fresh-I {name} metrics use a different denominator")
    if temporal.patient_macro_average_precision is None or (
        prevalence.patient_macro_average_precision is None
    ):
        raise ValueError("Fresh-I qualification requires both-class AP support")
    lower = float(paired_bce_interval.get("lower_95", math.nan))
    if not math.isfinite(lower):
        raise ValueError("Fresh-I paired BCE interval lacks a finite lower bound")

    required = {
        "minimum_discrimination_patients": int(
            thresholds["minimum_discrimination_patients"]
        ),
        "maximum_temporal_patient_macro_bce": float(
            thresholds["maximum_temporal_patient_macro_bce"]
        ),
        "maximum_temporal_patient_macro_brier": float(
            thresholds["maximum_temporal_patient_macro_brier"]
        ),
        "minimum_bce_improvement_over_independent": float(
            thresholds["minimum_bce_improvement_over_independent"]
        ),
        "minimum_brier_improvement_over_independent": float(
            thresholds["minimum_brier_improvement_over_independent"]
        ),
        "minimum_bce_improvement_over_time_only": float(
            thresholds["minimum_bce_improvement_over_time_only"]
        ),
        "minimum_bce_improvement_over_mask_only": float(
            thresholds["minimum_bce_improvement_over_mask_only"]
        ),
        "minimum_ap_lift_over_fit_prevalence": float(
            thresholds["minimum_ap_lift_over_fit_prevalence"]
        ),
        "minimum_paired_bce_improvement_lower_95": float(
            thresholds["minimum_paired_bce_improvement_lower_95"]
        ),
    }
    checks = {
        "discrimination_patient_count": temporal.n_discrimination_patients,
        "temporal_patient_macro_bce": temporal.patient_macro_bce,
        "temporal_patient_macro_brier": temporal.patient_macro_brier,
        "bce_improvement_over_independent": (
            independent.patient_macro_bce - temporal.patient_macro_bce
        ),
        "brier_improvement_over_independent": (
            independent.patient_macro_brier - temporal.patient_macro_brier
        ),
        "bce_improvement_over_time_only": (
            time_only.patient_macro_bce - temporal.patient_macro_bce
        ),
        "bce_improvement_over_mask_only": (
            mask_only.patient_macro_bce - temporal.patient_macro_bce
        ),
        "ap_lift_over_fit_prevalence": (
            temporal.patient_macro_average_precision
            - prevalence.patient_macro_average_precision
        ),
        "paired_bce_improvement_lower_95": lower,
    }
    passed_checks = {
        "minimum_discrimination_patients": checks["discrimination_patient_count"]
        >= required["minimum_discrimination_patients"],
        "maximum_temporal_patient_macro_bce": checks["temporal_patient_macro_bce"]
        <= required["maximum_temporal_patient_macro_bce"],
        "maximum_temporal_patient_macro_brier": checks[
            "temporal_patient_macro_brier"
        ]
        <= required["maximum_temporal_patient_macro_brier"],
        "minimum_bce_improvement_over_independent": checks[
            "bce_improvement_over_independent"
        ]
        >= required["minimum_bce_improvement_over_independent"],
        "minimum_brier_improvement_over_independent": checks[
            "brier_improvement_over_independent"
        ]
        >= required["minimum_brier_improvement_over_independent"],
        "minimum_bce_improvement_over_time_only": checks[
            "bce_improvement_over_time_only"
        ]
        >= required["minimum_bce_improvement_over_time_only"],
        "minimum_bce_improvement_over_mask_only": checks[
            "bce_improvement_over_mask_only"
        ]
        >= required["minimum_bce_improvement_over_mask_only"],
        "minimum_ap_lift_over_fit_prevalence": checks[
            "ap_lift_over_fit_prevalence"
        ]
        >= required["minimum_ap_lift_over_fit_prevalence"],
        "minimum_paired_bce_improvement_lower_95": checks[
            "paired_bce_improvement_lower_95"
        ]
        > required["minimum_paired_bce_improvement_lower_95"],
    }
    passed = all(passed_checks.values())
    return {
        "schema_version": "tusz_ictal_native_fresh_transport_decision_v1",
        "passed": passed,
        "qualification": (
            "source_task_transport_qualified_only" if passed else "source_task_no_go"
        ),
        "current_soz_reasoner_I_authorized": False,
        "checks": checks,
        "thresholds": required,
        "passed_checks": passed_checks,
        "metrics": {name: asdict(metrics) for name, metrics in named.items()},
        "failure_consequence": (
            None if passed else "keep_I_family_absent_no_iteration_on_this_cohort"
        ),
        "pass_consequence": (
            "requires_label_fresh_soz_downstream_evidence_before_reasoner_use"
            if passed
            else None
        ),
    }


__all__ = [
    "decide_native_fresh_transport",
    "paired_patient_improvements",
    "patient_bootstrap_interval",
]
