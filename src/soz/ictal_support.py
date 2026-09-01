"""Explicit-support audit for sparse TUSZ ictal source supervision.

Unknown annotation cells are excluded before every count.  The audit keeps
training learnability separate from native discrimination/calibration metric
eligibility; callers must provide their prespecified minimum-support gate
instead of inheriting an arbitrary threshold from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .geometry import N_TCP_EDGES


@dataclass(frozen=True)
class IctalEventLabelSupport:
    event_id: str
    patient_id: str
    observed_labels: int
    positive_labels: int
    explicit_negative_labels: int

    def __post_init__(self) -> None:
        if not self.event_id or not self.patient_id:
            raise ValueError("Ictal support identities cannot be blank")
        counts = (
            self.observed_labels,
            self.positive_labels,
            self.explicit_negative_labels,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("Ictal event support counts must be non-negative integers")
        if self.observed_labels != (
            self.positive_labels + self.explicit_negative_labels
        ):
            raise ValueError("Ictal event support counts are inconsistent")

    @property
    def has_both_classes(self) -> bool:
        return self.positive_labels > 0 and self.explicit_negative_labels > 0


@dataclass(frozen=True)
class IctalPatientLabelSupport:
    patient_id: str
    event_count: int
    observed_labels: int
    positive_labels: int
    explicit_negative_labels: int
    events_with_both_classes: int

    def __post_init__(self) -> None:
        if not self.patient_id:
            raise ValueError("Ictal support patient_id cannot be blank")
        counts = (
            self.event_count,
            self.observed_labels,
            self.positive_labels,
            self.explicit_negative_labels,
            self.events_with_both_classes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("Ictal patient support counts must be non-negative integers")
        if self.event_count < 1 or self.events_with_both_classes > self.event_count:
            raise ValueError("Ictal patient event support is inconsistent")
        if self.observed_labels != (
            self.positive_labels + self.explicit_negative_labels
        ):
            raise ValueError("Ictal patient label support is inconsistent")

    @property
    def has_both_classes(self) -> bool:
        return self.positive_labels > 0 and self.explicit_negative_labels > 0


@dataclass(frozen=True)
class IctalSupportGateDecision:
    passed: bool
    minimum_explicit_negative_labels: int
    minimum_events_with_explicit_negative: int
    minimum_patients_with_both_classes: int
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        thresholds = (
            self.minimum_explicit_negative_labels,
            self.minimum_events_with_explicit_negative,
            self.minimum_patients_with_both_classes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in thresholds
        ):
            raise ValueError("Every ictal support threshold must be a positive integer")
        if self.passed != (not self.failures):
            raise ValueError("Ictal support gate result disagrees with its failures")


@dataclass(frozen=True)
class IctalLabelSupportAudit:
    events: tuple[IctalEventLabelSupport, ...]
    patients: tuple[IctalPatientLabelSupport, ...]

    def __post_init__(self) -> None:
        if not self.events or not self.patients:
            raise ValueError("Ictal support audit requires events and patients")
        event_ids = tuple(item.event_id for item in self.events)
        patient_ids = tuple(item.patient_id for item in self.patients)
        if event_ids != tuple(sorted(set(event_ids))):
            raise ValueError("Ictal support events must be unique and sorted")
        if patient_ids != tuple(sorted(set(patient_ids))):
            raise ValueError("Ictal support patients must be unique and sorted")
        if {item.patient_id for item in self.events} != set(patient_ids):
            raise ValueError("Ictal event and patient support rosters disagree")

    @property
    def observed_labels(self) -> int:
        return sum(item.observed_labels for item in self.events)

    @property
    def positive_labels(self) -> int:
        return sum(item.positive_labels for item in self.events)

    @property
    def explicit_negative_labels(self) -> int:
        return sum(item.explicit_negative_labels for item in self.events)

    @property
    def events_with_both_classes(self) -> int:
        return sum(item.has_both_classes for item in self.events)

    @property
    def events_with_explicit_negative(self) -> int:
        return sum(item.explicit_negative_labels > 0 for item in self.events)

    @property
    def patients_with_both_classes(self) -> int:
        return sum(item.has_both_classes for item in self.patients)

    @property
    def patients_with_explicit_negative(self) -> int:
        return sum(item.explicit_negative_labels > 0 for item in self.patients)

    def assess_native_discrimination_support(
        self,
        *,
        minimum_explicit_negative_labels: int,
        minimum_events_with_explicit_negative: int,
        minimum_patients_with_both_classes: int,
    ) -> IctalSupportGateDecision:
        """Apply caller-frozen support minima before reporting discrimination.

        A failed gate means AUROC/AP/calibration promotion must be reported as
        unsupported/``NA`` for that cohort.  It does not erase positive-only
        events or turn their unknown cells into negatives.
        """

        thresholds = (
            minimum_explicit_negative_labels,
            minimum_events_with_explicit_negative,
            minimum_patients_with_both_classes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in thresholds
        ):
            raise ValueError("Every ictal support threshold must be a positive integer")
        failures: list[str] = []
        if self.explicit_negative_labels < minimum_explicit_negative_labels:
            failures.append("insufficient_explicit_negative_labels")
        if (
            self.events_with_explicit_negative
            < minimum_events_with_explicit_negative
        ):
            failures.append("insufficient_events_with_explicit_negative")
        if self.patients_with_both_classes < minimum_patients_with_both_classes:
            failures.append("insufficient_patients_with_both_classes")
        ordered = tuple(failures)
        return IctalSupportGateDecision(
            passed=not ordered,
            minimum_explicit_negative_labels=minimum_explicit_negative_labels,
            minimum_events_with_explicit_negative=(
                minimum_events_with_explicit_negative
            ),
            minimum_patients_with_both_classes=minimum_patients_with_both_classes,
            failures=ordered,
        )


def audit_ictal_source_support(
    targets: torch.Tensor,
    source_target_mask: torch.Tensor,
    *,
    event_ids: Sequence[object],
    patient_ids: Sequence[object],
) -> IctalLabelSupportAudit:
    """Count only explicit positive/background source labels by event/patient."""

    if not isinstance(targets, torch.Tensor) or not isinstance(
        source_target_mask, torch.Tensor
    ):
        raise TypeError("Ictal targets and source_target_mask must be tensors")
    if targets.ndim != 3 or targets.shape[1] != N_TCP_EDGES:
        raise ValueError("Ictal targets must have shape [E,20,S]")
    if tuple(source_target_mask.shape) != tuple(targets.shape):
        raise ValueError("source_target_mask must match [E,20,S] targets")
    if not targets.is_floating_point() or source_target_mask.dtype != torch.bool:
        raise TypeError("Ictal targets must be float and source_target_mask bool")
    if targets.device != source_target_mask.device:
        raise ValueError("Ictal targets and source_target_mask must share a device")
    if targets.requires_grad or source_target_mask.requires_grad:
        raise ValueError("Ictal support inputs must be detached")
    observed = targets[source_target_mask]
    if not torch.isfinite(observed).all() or (
        observed.numel() and not torch.all((observed == 0) | (observed == 1))
    ):
        raise ValueError("Observed ictal source targets must be finite binary values")

    normalized_events = tuple(str(value).strip() for value in event_ids)
    normalized_patients = tuple(str(value).strip() for value in patient_ids)
    n_events = targets.shape[0]
    if len(normalized_events) != n_events or len(normalized_patients) != n_events:
        raise ValueError("Ictal support identities must align with events")
    if any(not value for value in (*normalized_events, *normalized_patients)):
        raise ValueError("Ictal support identities cannot be blank")
    if len(set(normalized_events)) != n_events:
        raise ValueError("Ictal support event IDs must be unique")

    event_rows: list[IctalEventLabelSupport] = []
    for event_index, (event_id, patient_id) in enumerate(
        zip(normalized_events, normalized_patients)
    ):
        mask = source_target_mask[event_index]
        values = targets[event_index][mask]
        positive = int(values.sum().item())
        total = int(mask.sum().item())
        event_rows.append(
            IctalEventLabelSupport(
                event_id=event_id,
                patient_id=patient_id,
                observed_labels=total,
                positive_labels=positive,
                explicit_negative_labels=total - positive,
            )
        )
    ordered_events = tuple(sorted(event_rows, key=lambda item: item.event_id))

    patient_rows: list[IctalPatientLabelSupport] = []
    for patient_id in sorted(set(normalized_patients)):
        rows = tuple(item for item in ordered_events if item.patient_id == patient_id)
        patient_rows.append(
            IctalPatientLabelSupport(
                patient_id=patient_id,
                event_count=len(rows),
                observed_labels=sum(item.observed_labels for item in rows),
                positive_labels=sum(item.positive_labels for item in rows),
                explicit_negative_labels=sum(
                    item.explicit_negative_labels for item in rows
                ),
                events_with_both_classes=sum(item.has_both_classes for item in rows),
            )
        )
    return IctalLabelSupportAudit(
        events=ordered_events,
        patients=tuple(patient_rows),
    )


__all__ = [
    "IctalEventLabelSupport",
    "IctalLabelSupportAudit",
    "IctalPatientLabelSupport",
    "IctalSupportGateDecision",
    "audit_ictal_source_support",
]
