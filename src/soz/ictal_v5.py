"""One-cycle formal-v5 remediation for native TUSZ ictal involvement.

This module never consumes DeepSOZ SOZ vectors.  It freezes a balanced
auxiliary development/gate split from explicit TUSZ source labels and applies
the predeclared development decision to the single allowed temporal head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Mapping, Sequence

import torch

from .concept_metrics import IctalConceptMetrics, patient_macro_ictal_metrics


V5_AUXILIARY_PATIENT_COUNT = 64
V5_BOTH_CLASS_AUXILIARY_PATIENT_COUNT = 61
V5_HIGH_SUPPORT_PATIENT_COUNT = 24
V5_GROUP_PATIENT_COUNT = 12
V5_MINIMUM_POSITIVE_LABELS = 1200
V5_MINIMUM_NEGATIVE_LABELS = 1200
V5_MINIMUM_BCE_IMPROVEMENT = 0.01
V5_MINIMUM_BRIER_IMPROVEMENT = 0.01
V5_MINIMUM_CONTROL_BCE_IMPROVEMENT = 0.01
V5_MINIMUM_AP_LIFT = 0.01
V5_MINIMUM_DISCRIMINATION_PATIENTS = 10


@dataclass(frozen=True)
class IctalV5PatientSupport:
    patient_id: str
    event_count: int
    observed_labels: int
    positive_labels: int
    explicit_negative_labels: int

    def __post_init__(self) -> None:
        patient = str(self.patient_id).strip()
        if not patient or patient != self.patient_id:
            raise ValueError("V5 support patient_id must be trimmed and non-empty")
        for field in (
            "event_count",
            "observed_labels",
            "positive_labels",
            "explicit_negative_labels",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.event_count < 1:
            raise ValueError("V5 support requires at least one event")
        if self.observed_labels != (
            self.positive_labels + self.explicit_negative_labels
        ):
            raise ValueError("V5 support positive/negative counts are inconsistent")

    @property
    def prevalence(self) -> float:
        if self.observed_labels < 1:
            raise ValueError("V5 support prevalence requires observed labels")
        return self.positive_labels / self.observed_labels

    @property
    def has_both_classes(self) -> bool:
        return self.positive_labels > 0 and self.explicit_negative_labels > 0

    @property
    def high_support(self) -> bool:
        return (
            self.positive_labels >= V5_MINIMUM_POSITIVE_LABELS
            and self.explicit_negative_labels >= V5_MINIMUM_NEGATIVE_LABELS
        )


def _standardized_balance_features(
    rows: Sequence[IctalV5PatientSupport],
) -> tuple[tuple[float, float], ...]:
    raw = []
    for row in rows:
        prevalence = min(max(row.prevalence, 1e-12), 1.0 - 1e-12)
        raw.append(
            (
                math.log(prevalence / (1.0 - prevalence)),
                math.log1p(row.event_count),
            )
        )
    means = tuple(sum(values[index] for values in raw) / len(raw) for index in range(2))
    scales = []
    for index in range(2):
        variance = sum((values[index] - means[index]) ** 2 for values in raw) / len(raw)
        scales.append(math.sqrt(variance))
    if any(scale == 0.0 for scale in scales):
        raise ValueError("V5 split cannot balance a constant prevalence/event feature")
    return tuple(
        tuple((values[index] - means[index]) / scales[index] for index in range(2))
        for values in raw
    )


def choose_balanced_v5_groups(
    candidates: Sequence[IctalV5PatientSupport],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, float]]:
    """Choose the unique best 12/12 support-only balance partition.

    Complementary partitions have identical balance.  Requiring the
    lexicographically first candidate in I-dev gives the two groups stable
    names without consulting EEG features or model outcomes.
    """

    ordered = tuple(sorted(candidates, key=lambda row: row.patient_id))
    if len(ordered) != V5_HIGH_SUPPORT_PATIENT_COUNT:
        raise ValueError("V5 balancing requires exactly 24 high-support patients")
    if len({row.patient_id for row in ordered}) != len(ordered):
        raise ValueError("V5 balancing requires unique patient IDs")
    if any(not row.high_support for row in ordered):
        raise ValueError("V5 balancing received a patient below frozen support")
    features = _standardized_balance_features(ordered)
    total = tuple(sum(row[index] for row in features) for index in range(2))
    best_key: tuple[float, float, tuple[int, ...]] | None = None
    best_indices: tuple[int, ...] | None = None
    # Index zero fixes the otherwise arbitrary complement symmetry.
    for tail in combinations(range(1, len(ordered)), V5_GROUP_PATIENT_COUNT - 1):
        indices = (0, *tail)
        selected = set(indices)
        difference = tuple(
            (
                2.0 * sum(features[index][feature] for index in indices)
                - total[feature]
            )
            / V5_GROUP_PATIENT_COUNT
            for feature in range(2)
        )
        key = (
            max(abs(value) for value in difference),
            sum(value * value for value in difference),
            indices,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_indices = indices
    if best_indices is None:
        raise RuntimeError("V5 balance search produced no partition")
    dev_index = set(best_indices)
    dev = tuple(ordered[index].patient_id for index in sorted(dev_index))
    gate = tuple(
        ordered[index].patient_id
        for index in range(len(ordered))
        if index not in dev_index
    )
    dev_rows = tuple(row for row in ordered if row.patient_id in set(dev))
    gate_rows = tuple(row for row in ordered if row.patient_id in set(gate))
    balance = {
        "prevalence_mean_i_dev": sum(row.prevalence for row in dev_rows) / len(dev_rows),
        "prevalence_mean_i_gate": sum(row.prevalence for row in gate_rows) / len(gate_rows),
        "event_count_mean_i_dev": sum(row.event_count for row in dev_rows) / len(dev_rows),
        "event_count_mean_i_gate": sum(row.event_count for row in gate_rows) / len(gate_rows),
        "maximum_standardized_mean_difference": best_key[0],
    }
    if balance["maximum_standardized_mean_difference"] > 0.25:
        raise ValueError("V5 support-only split failed the prespecified balance bound")
    return dev, gate, balance


def freeze_v5_auxiliary_split(
    support_rows: Sequence[IctalV5PatientSupport],
    *,
    source_train_target_patient_ids: Sequence[object],
) -> dict[str, object]:
    ordered = tuple(sorted(support_rows, key=lambda row: row.patient_id))
    patients = tuple(row.patient_id for row in ordered)
    if len(set(patients)) != len(patients):
        raise ValueError("V5 support roster contains duplicate patients")
    target = tuple(sorted(str(value).strip() for value in source_train_target_patient_ids))
    if not target or any(not value for value in target) or len(set(target)) != len(target):
        raise ValueError("V5 source-train target roster is invalid")
    auxiliary = tuple(row for row in ordered if row.patient_id not in set(target))
    if len(auxiliary) != V5_AUXILIARY_PATIENT_COUNT:
        raise ValueError("V5 requires exactly 64 non-target auxiliary patients")
    both = tuple(row for row in auxiliary if row.has_both_classes)
    if len(both) != V5_BOTH_CLASS_AUXILIARY_PATIENT_COUNT:
        raise ValueError("V5 auxiliary both-class count changed from the frozen audit")
    high = tuple(row for row in auxiliary if row.high_support)
    if len(high) != V5_HIGH_SUPPORT_PATIENT_COUNT:
        raise ValueError("V5 high-support count changed from the frozen audit")
    dev, gate, balance = choose_balanced_v5_groups(high)
    return {
        "schema_version": "soz_ictal_formal_v5_auxiliary_split_v1",
        "target_semantics": "tusz_bipolar_edge_time_involvement_not_soz",
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "missing_tusz_cells_imputed_as_negative": False,
        "source_train_target_patient_ids": list(target),
        "auxiliary_patient_count": len(auxiliary),
        "auxiliary_both_class_patient_count": len(both),
        "high_support_patient_count": len(high),
        "minimum_positive_labels": V5_MINIMUM_POSITIVE_LABELS,
        "minimum_explicit_negative_labels": V5_MINIMUM_NEGATIVE_LABELS,
        "i_dev_patient_ids": list(dev),
        "i_gate_patient_ids": list(gate),
        "balance": balance,
        "high_support_rows": [asdict(row) | {"prevalence": row.prevalence} for row in high],
    }


def _smoothed_logit(positive: int, observed: int) -> float:
    probability = (positive + 1.0) / (observed + 2.0)
    return math.log(probability / (1.0 - probability))


def v5_shortcut_logits(
    *,
    control: str,
    training_targets: torch.Tensor,
    training_mask: torch.Tensor,
    evaluation_targets: torch.Tensor,
    evaluation_mask: torch.Tensor,
) -> torch.Tensor:
    """Fit the unchanged V4 time/mask shortcut on training labels only."""

    expected = tuple(training_targets.shape)
    if (
        training_targets.ndim != 3
        or expected[1:] != (20, 60)
        or tuple(training_mask.shape) != expected
        or evaluation_targets.ndim != 3
        or tuple(evaluation_targets.shape[1:]) != (20, 60)
        or tuple(evaluation_mask.shape) != tuple(evaluation_targets.shape)
    ):
        raise ValueError("V5 shortcut tensors must have shape [E,20,60]")
    if training_mask.dtype != torch.bool or evaluation_mask.dtype != torch.bool:
        raise TypeError("V5 shortcut masks must be bool")
    observed = training_targets[training_mask]
    if observed.numel() < 1 or not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("V5 shortcut training labels must contain observed binary cells")
    event_count = evaluation_targets.shape[0]
    if control == "time_only":
        counts = training_mask.sum(dim=(0, 1)).to(torch.int64)
        positives = (training_targets * training_mask).sum(dim=(0, 1)).to(torch.int64)
        logits = torch.tensor(
            [_smoothed_logit(int(pos), int(count)) for pos, count in zip(positives, counts)],
            dtype=torch.float32,
        )
        return logits.view(1, 1, 60, 1).expand(event_count, 20, 60, 1).clone()
    if control != "mask_only":
        raise ValueError("V5 shortcut control must be time_only or mask_only")
    boundaries = torch.tensor((0.25, 0.5, 0.75), dtype=torch.float64)
    train_bins = torch.bucketize(
        training_mask.to(torch.float64).mean(dim=(1, 2)), boundaries
    )
    global_logit = _smoothed_logit(
        int((training_targets * training_mask).sum().item()),
        int(training_mask.sum().item()),
    )
    bin_logits = []
    for bin_index in range(4):
        selected = train_bins == bin_index
        if not selected.any():
            bin_logits.append(global_logit)
            continue
        selected_mask = training_mask[selected]
        bin_logits.append(
            _smoothed_logit(
                int((training_targets[selected] * selected_mask).sum().item()),
                int(selected_mask.sum().item()),
            )
        )
    evaluation_bins = torch.bucketize(
        evaluation_mask.to(torch.float64).mean(dim=(1, 2)), boundaries
    )
    event_logits = torch.tensor(
        [bin_logits[int(index)] for index in evaluation_bins], dtype=torch.float32
    )
    return event_logits.view(-1, 1, 1, 1).expand(event_count, 20, 60, 1).clone()


def decide_v5_i_dev(
    *,
    independent_metrics: IctalConceptMetrics,
    temporal_metrics: IctalConceptMetrics,
    time_only_metrics: IctalConceptMetrics,
    mask_only_metrics: IctalConceptMetrics,
    prevalence_metrics: IctalConceptMetrics,
) -> dict[str, object]:
    metrics = {
        "independent": independent_metrics,
        "temporal": temporal_metrics,
        "time_only": time_only_metrics,
        "mask_only": mask_only_metrics,
        "prevalence": prevalence_metrics,
    }
    if any(value.n_patients != V5_GROUP_PATIENT_COUNT for value in metrics.values()):
        raise ValueError("V5 I-dev metrics must use the same 12 patients")
    if any(
        value.n_observed_labels != temporal_metrics.n_observed_labels
        or value.n_positive_labels != temporal_metrics.n_positive_labels
        or value.n_negative_labels != temporal_metrics.n_negative_labels
        for value in metrics.values()
    ):
        raise ValueError("V5 I-dev metrics must use identical observed cells")
    if temporal_metrics.patient_macro_average_precision is None or (
        prevalence_metrics.patient_macro_average_precision is None
    ):
        raise ValueError("V5 I-dev AP requires both-class patient support")
    checks = {
        "bce_improvement_over_independent": (
            independent_metrics.patient_macro_bce - temporal_metrics.patient_macro_bce
        ),
        "brier_improvement_over_independent": (
            independent_metrics.patient_macro_brier - temporal_metrics.patient_macro_brier
        ),
        "bce_improvement_over_time_only": (
            time_only_metrics.patient_macro_bce - temporal_metrics.patient_macro_bce
        ),
        "bce_improvement_over_mask_only": (
            mask_only_metrics.patient_macro_bce - temporal_metrics.patient_macro_bce
        ),
        "ap_lift_over_prevalence": (
            temporal_metrics.patient_macro_average_precision
            - prevalence_metrics.patient_macro_average_precision
        ),
        "discrimination_patient_count": temporal_metrics.n_discrimination_patients,
    }
    passed = (
        checks["bce_improvement_over_independent"] >= V5_MINIMUM_BCE_IMPROVEMENT
        and checks["brier_improvement_over_independent"] >= V5_MINIMUM_BRIER_IMPROVEMENT
        and checks["bce_improvement_over_time_only"]
        >= V5_MINIMUM_CONTROL_BCE_IMPROVEMENT
        and checks["bce_improvement_over_mask_only"]
        >= V5_MINIMUM_CONTROL_BCE_IMPROVEMENT
        and checks["ap_lift_over_prevalence"] >= V5_MINIMUM_AP_LIFT
        and checks["discrimination_patient_count"]
        >= V5_MINIMUM_DISCRIMINATION_PATIENTS
    )
    return {
        "schema_version": "soz_ictal_formal_v5_i_dev_decision_v1",
        "selected_head": "temporal_residual_k5" if passed else None,
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "minimum_bce_improvement": V5_MINIMUM_BCE_IMPROVEMENT,
            "minimum_brier_improvement": V5_MINIMUM_BRIER_IMPROVEMENT,
            "minimum_control_bce_improvement": V5_MINIMUM_CONTROL_BCE_IMPROVEMENT,
            "minimum_ap_lift": V5_MINIMUM_AP_LIFT,
            "minimum_discrimination_patients": V5_MINIMUM_DISCRIMINATION_PATIENTS,
        },
        "metrics": {name: asdict(value) for name, value in metrics.items()},
        "failure_consequence": None if passed else "remove_ictal_family_no_third_iteration",
    }


def prevalence_baseline_metrics(
    *,
    training_targets: torch.Tensor,
    training_mask: torch.Tensor,
    evaluation_targets: torch.Tensor,
    evaluation_mask: torch.Tensor,
    evaluation_patient_ids: torch.Tensor,
) -> IctalConceptMetrics:
    observed = training_targets[training_mask]
    positive = int(observed.sum().item())
    total = int(observed.numel())
    if positive < 1 or positive >= total:
        raise ValueError("V5 prevalence baseline requires both training classes")
    logit = math.log((positive / total) / (1.0 - positive / total))
    logits = torch.full(
        (*evaluation_targets.shape, 1), logit, dtype=torch.float32
    )
    return patient_macro_ictal_metrics(
        logits,
        evaluation_targets,
        evaluation_mask,
        evaluation_patient_ids,
    )


__all__ = [
    "IctalV5PatientSupport",
    "choose_balanced_v5_groups",
    "decide_v5_i_dev",
    "freeze_v5_auxiliary_split",
    "prevalence_baseline_metrics",
    "v5_shortcut_logits",
]
