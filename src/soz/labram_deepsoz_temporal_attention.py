"""Low-capacity DeepSOZ-style temporal attention on frozen LaBraM tokens.

The module deliberately separates source-native seizure detection from
patient-level SOZ localization.  It never accepts raw EEG and contains no
foundation-model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import N_STANDARD_CHANNELS
from .v11_reasoner import V11_CANDIDATE_MASK


N_SECONDS: Final[int] = 60
PREICTAL_SECONDS: Final[int] = 12
TOKEN_DIM: Final[int] = 200
DETECTOR_HIDDEN: Final[int] = 100
DETECTION_CLASS_WEIGHTS: Final[tuple[float, float]] = (0.2, 0.8)


def _require_finite_float(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def validate_node_time(node_time: torch.Tensor) -> None:
    _require_finite_float(node_time, name="frozen LaBraM node-time tokens")
    if node_time.ndim != 4 or node_time.shape[1] != N_STANDARD_CHANNELS or (
        node_time.shape[3] != TOKEN_DIM
    ):
        raise ValueError("node_time must have shape [E,19,T,200]")
    if node_time.shape[0] < 1:
        raise ValueError("node_time must contain at least one event")
    if node_time.shape[2] < 1 or node_time.shape[2] > N_SECONDS:
        raise ValueError("node_time T must lie in [1,60]")


def build_detection_targets(duration_seconds: torch.Tensor) -> torch.Tensor:
    """Build conservative one-second TUSZ interval targets.

    A positive cell must be fully contained in ``[t0, stop)``.  The first 12
    cells are the pre-anchor baseline and all cells after the audited stop are
    negative.  These labels describe seizure interval membership, not SOZ.
    """

    _require_finite_float(duration_seconds, name="event durations")
    if duration_seconds.ndim != 1 or duration_seconds.numel() < 1:
        raise ValueError("duration_seconds must be non-empty [E]")
    full_ictal = torch.floor(duration_seconds).long().clamp(
        min=0, max=N_SECONDS - PREICTAL_SECONDS
    )
    if bool((full_ictal < 1).any()):
        raise ValueError("every event must contain at least one full ictal second")
    relative = torch.arange(
        N_SECONDS - PREICTAL_SECONDS,
        device=duration_seconds.device,
    )
    targets = torch.zeros(
        (duration_seconds.numel(), N_SECONDS),
        dtype=torch.long,
        device=duration_seconds.device,
    )
    targets[:, PREICTAL_SECONDS:] = (
        relative.unsqueeze(0) < full_ictal.unsqueeze(1)
    ).long()
    return targets


@dataclass(frozen=True)
class DetectionAttentionOutput:
    detection_logits: torch.Tensor
    attention: torch.Tensor
    normalized_node_time: torch.Tensor

    def __post_init__(self) -> None:
        events = int(self.detection_logits.shape[0])
        if self.detection_logits.ndim != 3 or self.detection_logits.shape[2] != 2:
            raise ValueError("detection_logits must have shape [E,T,2]")
        time_steps = int(self.detection_logits.shape[1])
        if tuple(self.attention.shape) != (events, time_steps):
            raise ValueError("attention must have shape [E,T]")
        if tuple(self.normalized_node_time.shape) != (
            events,
            N_STANDARD_CHANNELS,
            time_steps,
            TOKEN_DIM,
        ):
            raise ValueError("normalized_node_time has an invalid shape")
        for value in (
            self.detection_logits,
            self.attention,
            self.normalized_node_time,
        ):
            _require_finite_float(value, name="detection-attention output")
        expected = torch.ones(
            events,
            dtype=self.attention.dtype,
            device=self.attention.device,
        )
        if not torch.allclose(
            self.attention.sum(dim=1), expected, atol=1e-6, rtol=1e-6
        ):
            raise ValueError("temporal attention must sum to one per event")


class FrozenLaBraMTemporalDetector(nn.Module):
    """BiLSTM seizure detector over channel-mean frozen LaBraM tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int = DETECTOR_HIDDEN,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.temporal_lstm = nn.LSTM(
            input_size=TOKEN_DIM,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.detection_head = nn.Linear(2 * self.hidden_dim, 2)

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(self, node_time: torch.Tensor) -> DetectionAttentionOutput:
        validate_node_time(node_time)
        normalized = F.layer_norm(node_time, (TOKEN_DIM,))
        global_time = normalized.mean(dim=1)
        temporal, _ = self.temporal_lstm(global_time)
        logits = self.detection_head(self.dropout(temporal))
        seizure_probability = torch.softmax(logits, dim=-1)[..., 1]
        attention = seizure_probability / seizure_probability.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        return DetectionAttentionOutput(
            detection_logits=logits,
            attention=attention,
            normalized_node_time=normalized,
        )


def pool_event_features(
    normalized_node_time: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    """Pool frozen channel-time tokens to one feature per event and channel."""

    validate_node_time(normalized_node_time)
    events = int(normalized_node_time.shape[0])
    time_steps = int(normalized_node_time.shape[2])
    _require_finite_float(attention, name="temporal attention")
    if tuple(attention.shape) != (events, time_steps):
        raise ValueError("attention must have shape [E,T]")
    expected = torch.ones(events, dtype=attention.dtype, device=attention.device)
    if not torch.allclose(attention.sum(dim=1), expected, atol=1e-6, rtol=1e-6):
        raise ValueError("attention must sum to one per event")
    pooled = torch.einsum("et,ectd->ecd", attention, normalized_node_time)
    if tuple(pooled.shape) != (events, N_STANDARD_CHANNELS, TOKEN_DIM):
        raise RuntimeError("event feature pooling shape drifted")
    return pooled.contiguous()


def uniform_pool_event_features(node_time: torch.Tensor) -> torch.Tensor:
    """Matched control: normalize identically and average all 60 seconds."""

    validate_node_time(node_time)
    normalized = F.layer_norm(node_time, (TOKEN_DIM,))
    return normalized.mean(dim=2).contiguous()


class SharedChannelSOZHead(nn.Module):
    """Shared residual channel scorer added to an outer-train-only prior."""

    def __init__(self, prior_logits: torch.Tensor) -> None:
        super().__init__()
        _require_finite_float(prior_logits, name="channel prior logits")
        if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("prior_logits must have shape [19]")
        self.projection = nn.Linear(TOKEN_DIM, 1, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.register_buffer("prior_logits", prior_logits.detach().float().clone())
        self.register_buffer("candidate_mask", V11_CANDIDATE_MASK.clone())

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(self, event_features: torch.Tensor) -> torch.Tensor:
        _require_finite_float(event_features, name="event channel features")
        if event_features.ndim != 3 or tuple(event_features.shape[1:]) != (
            N_STANDARD_CHANNELS,
            TOKEN_DIM,
        ):
            raise ValueError("event_features must have shape [E,19,200]")
        residual = self.projection(event_features).squeeze(-1)
        prior = self.prior_logits.to(
            device=event_features.device,
            dtype=event_features.dtype,
        )
        return residual + prior.unsqueeze(0)


def aggregate_patient_probabilities(
    event_logits: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
) -> torch.Tensor:
    """Equally average every seizure probability within each patient."""

    _require_finite_float(event_logits, name="event SOZ logits")
    if event_logits.ndim != 2 or event_logits.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("event_logits must have shape [E,19]")
    events = int(event_logits.shape[0])
    if (
        not isinstance(event_patient_index, torch.Tensor)
        or event_patient_index.dtype != torch.long
        or tuple(event_patient_index.shape) != (events,)
        or event_patient_index.device != event_logits.device
    ):
        raise TypeError("event_patient_index must be aligned torch.long [E]")
    if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer")
    if events < n_patients or int(event_patient_index.min()) != 0 or (
        int(event_patient_index.max()) != n_patients - 1
    ):
        raise ValueError("event_patient_index must cover a contiguous patient roster")
    if torch.unique(event_patient_index).numel() != n_patients:
        raise ValueError("every patient must own at least one event")
    probabilities = torch.sigmoid(event_logits)
    sums = probabilities.new_zeros((n_patients, N_STANDARD_CHANNELS))
    sums.index_add_(0, event_patient_index, probabilities)
    counts = torch.bincount(event_patient_index, minlength=n_patients).to(
        dtype=probabilities.dtype
    )
    return sums / counts.unsqueeze(1)


def masked_patient_bce_l1(
    patient_probabilities: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    l1_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSOZ benchmark-membership BCE plus probability sparsity penalty."""

    _require_finite_float(patient_probabilities, name="patient probabilities")
    _require_finite_float(targets, name="patient targets")
    if patient_probabilities.ndim != 2 or patient_probabilities.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("patient probabilities must have shape [P,19]")
    if tuple(targets.shape) != tuple(patient_probabilities.shape) or (
        tuple(target_mask.shape) != tuple(patient_probabilities.shape)
    ):
        raise ValueError("targets and target_mask must align with probabilities")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be torch.bool")
    if bool((target_mask & ~V11_CANDIDATE_MASK.to(target_mask.device)).any()):
        raise ValueError("target_mask cannot include PZ")
    observed = targets[target_mask]
    if observed.numel() < 1 or not bool(((observed == 0) | (observed == 1)).all()):
        raise ValueError("observed targets must be non-empty binary values")
    if not 0.0 <= float(l1_weight):
        raise ValueError("l1_weight must be non-negative")
    clipped = patient_probabilities.clamp(1.0e-6, 1.0 - 1.0e-6)
    bce = F.binary_cross_entropy(clipped[target_mask], observed)
    sparsity = clipped[target_mask].mean()
    total = bce + float(l1_weight) * sparsity
    return total, bce, sparsity


def weighted_detection_loss(
    detection_logits: torch.Tensor,
    detection_targets: torch.Tensor,
) -> torch.Tensor:
    _require_finite_float(detection_logits, name="detection logits")
    if detection_logits.ndim != 3 or detection_logits.shape[2] != 2 or (
        detection_logits.shape[1] < 1
    ):
        raise ValueError("detection_logits must have shape [E,T,2]")
    if detection_targets.dtype != torch.long or tuple(detection_targets.shape) != (
        detection_logits.shape[0],
        detection_logits.shape[1],
    ):
        raise TypeError("detection_targets must be aligned torch.long [E,T]")
    if not bool(((detection_targets == 0) | (detection_targets == 1)).all()):
        raise ValueError("detection targets must be binary")
    weight = detection_logits.new_tensor(DETECTION_CLASS_WEIGHTS)
    return F.cross_entropy(
        detection_logits.reshape(-1, 2),
        detection_targets.reshape(-1),
        weight=weight,
    )


__all__ = [
    "DETECTION_CLASS_WEIGHTS",
    "DETECTOR_HIDDEN",
    "DetectionAttentionOutput",
    "FrozenLaBraMTemporalDetector",
    "N_SECONDS",
    "PREICTAL_SECONDS",
    "SharedChannelSOZHead",
    "TOKEN_DIM",
    "aggregate_patient_probabilities",
    "build_detection_targets",
    "masked_patient_bce_l1",
    "pool_event_features",
    "uniform_pool_event_features",
    "validate_node_time",
    "weighted_detection_loss",
]
