"""Offset-aware wrapper around the frozen direct evolution implementation.

This module is intentionally separate from :mod:`src.soz.evolution`: existing
direct-descriptor/scaler artifacts bind that implementation's exact source
bytes, while phase semantics are orthogonal event metadata and must not alter
the descriptor computation lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .evolution import (
    TemporalEvolutionDescriptors,
    compute_temporal_evolution_descriptors,
)
from .temporal_masks import (
    OffsetAwarePhaseMasks,
    build_offset_aware_phase_masks,
)


@dataclass(frozen=True)
class TemporalEvolutionEvidence:
    """Direct descriptors plus separate offset-aware clinical phase semantics."""

    evolution: TemporalEvolutionDescriptors
    phase_masks: OffsetAwarePhaseMasks

    def __post_init__(self) -> None:
        if not isinstance(self.evolution, TemporalEvolutionDescriptors):
            raise TypeError("evolution must be TemporalEvolutionDescriptors")
        if not isinstance(self.phase_masks, OffsetAwarePhaseMasks):
            raise TypeError("phase_masks must be OffsetAwarePhaseMasks")
        if (
            self.evolution.descriptors.shape[0]
            != self.phase_masks.ictal_phase_mask.shape[0]
        ):
            raise ValueError(
                "Evolution descriptors and phase masks must share batch size"
            )


def compute_temporal_evolution_evidence(
    eeg_volts: torch.Tensor,
    seizure_duration_sec: Sequence[float | None],
    *,
    offset_trustworthy: Sequence[bool],
    previous_seizure_gap_sec: Sequence[float | None],
    previous_timeline_trustworthy: Sequence[bool],
) -> TemporalEvolutionEvidence:
    """Compute direct features and their independent offset-aware phase mask.

    ``seizure_duration_sec`` must be the global seizure stop relative to t0;
    the crop's fixed ``window_stop_sec`` is not a seizure offset.
    """

    evolution = compute_temporal_evolution_descriptors(eeg_volts)
    phase_masks = build_offset_aware_phase_masks(
        seizure_duration_sec,
        offset_trustworthy=offset_trustworthy,
        previous_seizure_gap_sec=previous_seizure_gap_sec,
        previous_timeline_trustworthy=previous_timeline_trustworthy,
    )
    return TemporalEvolutionEvidence(
        evolution=evolution,
        phase_masks=phase_masks,
    )


__all__ = [
    "TemporalEvolutionEvidence",
    "compute_temporal_evolution_evidence",
]
