"""Patient-level aggregation for repeated seizure observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import N_STANDARD_CHANNELS


@dataclass(frozen=True)
class PatientAggregation:
    """Mean event-logit aggregation with one output row per patient."""

    logits: torch.Tensor
    patient_ids: torch.Tensor
    event_counts: torch.Tensor


def aggregate_patient_logits(
    event_logits: torch.Tensor,
    event_patient_ids: torch.Tensor,
) -> PatientAggregation:
    """Equally average event logits within patient.

    This operation intentionally has no learned attention: DeepSOZ labels only
    identify the patient aggregate, so learned event weights would not be
    separately identifiable from the available supervision.
    """

    if event_logits.ndim != 2 or event_logits.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError(
            "event_logits must have shape [E,19], got "
            f"{tuple(event_logits.shape)}"
        )
    if event_patient_ids.ndim != 1 or event_patient_ids.shape[0] != event_logits.shape[0]:
        raise ValueError("event_patient_ids must have shape [E]")
    if event_patient_ids.dtype != torch.long:
        raise TypeError("event_patient_ids must be torch.long")
    if event_patient_ids.device != event_logits.device:
        raise ValueError("event logits and patient IDs must share a device")
    if event_logits.shape[0] < 1:
        raise ValueError("At least one event is required")
    if not torch.isfinite(event_logits).all():
        raise ValueError("event_logits must be finite")

    patient_ids, inverse = torch.unique(
        event_patient_ids, sorted=True, return_inverse=True
    )
    n_patients = int(patient_ids.numel())
    sums = event_logits.new_zeros((n_patients, N_STANDARD_CHANNELS))
    sums.index_add_(0, inverse, event_logits)
    counts = torch.bincount(inverse, minlength=n_patients)
    logits = sums / counts.to(dtype=event_logits.dtype).unsqueeze(1)
    return PatientAggregation(logits=logits, patient_ids=patient_ids, event_counts=counts)

