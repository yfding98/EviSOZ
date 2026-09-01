"""Time-resolved, node-native LaBraM localization head for v12.

This module is deliberately independent of the frozen v11/v11.1 recovery
implementations.  It restores the one-second LaBraM node tokens that were
previously collapsed into coarse phase contrasts, fits every feature
transform on outer-fold training patients only, and learns a very small
channel-shared endpoint scorer.

The learned temporal gate is an onset-*anchored* positive-change weighting:
it is useful for emphasizing scalp-visible changes near the global TUSZ
seizure anchor, but it is neither a cortical onset estimate nor propagation
supervision.  The uniform gate is the matched control.  PZ remains a finite
signal carrier, but is excluded from the fixed 18-candidate gate, ranking,
and loss normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS
from .v11_reasoner import (
    V11_CANDIDATE_INDICES,
    V11_CANDIDATE_MASK,
    apply_fixed_candidate_mask,
    jeffreys_reference_prior_logits,
    positive_set_mass_loss,
)


V12_TIME_RESOLVED_SCHEMA: Final[str] = (
    "soz_labram_block9_time_resolved_node_native_v12"
)
V12_N_CALLS: Final[int] = 15
V12_SECONDS_PER_CALL: Final[int] = 4
V12_N_SECONDS: Final[int] = V12_N_CALLS * V12_SECONDS_PER_CALL
V12_PREICTAL_SECONDS: Final[int] = 12
V12_NODE_RAW_DIM: Final[int] = 200
V12_NODE_PCA_DIM: Final[int] = 16
V12_GATE_FLOOR: Final[float] = 1.0e-3
V12_RELIABILITY_FLOOR: Final[float] = 0.1
V12_CANDIDATE_INDICES: Final[tuple[int, ...]] = V11_CANDIDATE_INDICES
V12_CANDIDATE_MASK: Final[torch.Tensor] = V11_CANDIDATE_MASK.clone()

GateMode = Literal["uniform", "learned"]


def _require_float(
    value: torch.Tensor,
    *,
    name: str,
    detached: bool,
) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if detached and value.requires_grad:
        raise ValueError(f"{name} must be detached")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def restore_prefix_node_time(prefix: torch.Tensor) -> torch.Tensor:
    """Restore frozen LaBraM block-9 prefixes to ``[E,19,60,200]``.

    Each of the 15 independent four-second calls contains one CLS token and
    ``19 * 4`` electrode-second tokens.  CLS is excluded before the exact
    call/channel/second permutation.  The fixed time axis is ``[-12, 48)``
    seconds relative to the global scalp-visible seizure anchor.
    """

    _require_float(prefix, name="LaBraM block-9 prefix", detached=True)
    if prefix.ndim != 4 or tuple(prefix.shape[1:]) != (
        V12_N_CALLS,
        1 + N_STANDARD_CHANNELS * V12_SECONDS_PER_CALL,
        V12_NODE_RAW_DIM,
    ):
        raise ValueError("prefix must have shape [E,15,77,200]")
    events = int(prefix.shape[0])
    if events < 1:
        raise ValueError("prefix must contain at least one event")
    restored = (
        prefix[:, :, 1:, :]
        .reshape(
            events,
            V12_N_CALLS,
            N_STANDARD_CHANNELS,
            V12_SECONDS_PER_CALL,
            V12_NODE_RAW_DIM,
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(
            events,
            N_STANDARD_CHANNELS,
            V12_N_SECONDS,
            V12_NODE_RAW_DIM,
        )
    )
    return restored.detach().contiguous()


def baseline_difference_node_time(
    raw_node_time: torch.Tensor,
    temporal_masks: "V12TemporalMasks",
) -> torch.Tensor:
    """Form explicit preictal-baseline differences before fold fitting.

    For every event, channel, and feature, the reference is the mean of the
    12 audited pre-anchor tokens.  Only true-stop-aware ictal seconds retain
    ``token - baseline``; pre-anchor and invalid/post-stop rows are exactly
    zero.  Consequently neither baseline rows nor padding rows may dominate
    the robust scaler/PCA fit.
    """

    if not isinstance(temporal_masks, V12TemporalMasks):
        raise TypeError("baseline differencing requires V12TemporalMasks")
    _validate_node_time(
        raw_node_time,
        feature_dim=V12_NODE_RAW_DIM,
        name="raw node-time tokens",
    )
    if temporal_masks.n_events != int(raw_node_time.shape[0]) or (
        temporal_masks.ictal_valid_mask.device != raw_node_time.device
    ):
        raise ValueError("raw node-time tokens and temporal masks must align")
    baseline = raw_node_time[:, :, :V12_PREICTAL_SECONDS].mean(
        dim=2, keepdim=True
    )
    valid = temporal_masks.ictal_valid_mask[:, None, :, None]
    delta = torch.where(valid, raw_node_time - baseline, torch.zeros_like(raw_node_time))
    return delta.detach().contiguous()


def _validate_node_time(
    node_time: torch.Tensor,
    *,
    feature_dim: int,
    name: str,
) -> None:
    _require_float(node_time, name=name, detached=True)
    if node_time.ndim != 4 or tuple(node_time.shape[1:]) != (
        N_STANDARD_CHANNELS,
        V12_N_SECONDS,
        feature_dim,
    ):
        raise ValueError(
            f"{name} must have shape [E,19,60,{feature_dim}]"
        )
    if node_time.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one event")


def _validate_masked_delta(
    node_time: torch.Tensor,
    temporal_masks: "V12TemporalMasks",
    *,
    name: str,
) -> None:
    if not isinstance(temporal_masks, V12TemporalMasks):
        raise TypeError(f"{name} requires V12TemporalMasks")
    if temporal_masks.n_events != int(node_time.shape[0]) or (
        temporal_masks.ictal_valid_mask.device != node_time.device
    ):
        raise ValueError(f"{name} and temporal masks must align")
    # Reduce first instead of materializing an expanded [E,19,60,D] mask.
    # This validation runs on the full cache and must not duplicate it.
    time_nonzero = (node_time.amax(dim=(1, 3)) != 0) | (
        node_time.amin(dim=(1, 3)) != 0
    )
    if bool((time_nonzero & ~temporal_masks.ictal_valid_mask).any()):
        raise ValueError(f"{name} must be exactly zero outside valid ictal seconds")


def _validate_patient_roster(
    event_patient_index: torch.Tensor,
    *,
    events: int,
    device: torch.device,
) -> int:
    if not isinstance(event_patient_index, torch.Tensor) or (
        event_patient_index.dtype != torch.long
    ):
        raise TypeError("event_patient_index must be torch.long")
    if tuple(event_patient_index.shape) != (events,):
        raise ValueError("event_patient_index must have shape [E]")
    if event_patient_index.device != device:
        raise ValueError("event_patient_index and event tensors must share a device")
    if events < 1 or int(event_patient_index.min().item()) != 0:
        raise ValueError("event_patient_index must begin at patient 0")
    n_patients = int(event_patient_index.max().item()) + 1
    if torch.unique(event_patient_index).numel() != n_patients:
        raise ValueError("event_patient_index must be a contiguous complete roster")
    return n_patients


def _robust_center_scale(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = rows.median(dim=0).values
    mad = (rows - center).abs().median(dim=0).values
    standard = rows.std(dim=0, unbiased=False)
    scale = torch.maximum(1.4826 * mad, 0.1 * standard).clamp_min(1.0e-4)
    return center, scale


@dataclass(frozen=True)
class NodeFeatureFoldTransformV12:
    """Outer-train-only robust scaling and PCA16 for LaBraM node tokens."""

    center: torch.Tensor
    scale: torch.Tensor
    pca_mean: torch.Tensor
    components: torch.Tensor
    train_patient_indices: tuple[int, ...]
    train_event_count: int
    schema_version: str = V12_TIME_RESOLVED_SCHEMA

    def __post_init__(self) -> None:
        expected = {
            "center": (V12_NODE_RAW_DIM,),
            "scale": (V12_NODE_RAW_DIM,),
            "pca_mean": (V12_NODE_RAW_DIM,),
            "components": (V12_NODE_RAW_DIM, V12_NODE_PCA_DIM),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not value.is_floating_point() or (
                not torch.isfinite(value).all()
            ):
                raise ValueError(f"fold transform {name} has invalid shape/value")
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError("fold transform tensors must be detached CPU tensors")
        if torch.any(self.scale <= 0):
            raise ValueError("fold transform scale must be positive")
        if not self.train_patient_indices or len(set(self.train_patient_indices)) != len(
            self.train_patient_indices
        ):
            raise ValueError("fold transform needs unique outer-train patients")
        if isinstance(self.train_event_count, bool) or self.train_event_count < 1:
            raise ValueError("fold transform needs a positive training-event count")
        if self.schema_version != V12_TIME_RESOLVED_SCHEMA:
            raise ValueError("unsupported v12 fold-transform schema")

    def apply(
        self,
        node_time: torch.Tensor,
        temporal_masks: "V12TemporalMasks",
    ) -> torch.Tensor:
        """Apply the frozen transform without changing node or time axes."""

        _validate_node_time(
            node_time, feature_dim=V12_NODE_RAW_DIM, name="raw node-time tokens"
        )
        _validate_masked_delta(node_time, temporal_masks, name="raw delta-H")
        center = self.center.to(device=node_time.device, dtype=node_time.dtype)
        scale = self.scale.to(device=node_time.device, dtype=node_time.dtype)
        pca_mean = self.pca_mean.to(device=node_time.device, dtype=node_time.dtype)
        components = self.components.to(
            device=node_time.device, dtype=node_time.dtype
        )
        standardized = (node_time - center) / scale
        result = torch.matmul(standardized - pca_mean, components)
        result = torch.where(
            temporal_masks.ictal_valid_mask[:, None, :, None],
            result,
            torch.zeros_like(result),
        )
        if tuple(result.shape[1:]) != (
            N_STANDARD_CHANNELS,
            V12_N_SECONDS,
            V12_NODE_PCA_DIM,
        ):
            raise RuntimeError("v12 transformed node-time shape drifted")
        return result.detach().contiguous()

    def tensor_state(self) -> Mapping[str, torch.Tensor]:
        """Return a serialization-friendly, detached tensor state."""

        return {
            "center": self.center.clone(),
            "scale": self.scale.clone(),
            "pca_mean": self.pca_mean.clone(),
            "components": self.components.clone(),
            "train_patient_indices": torch.tensor(
                self.train_patient_indices, dtype=torch.long
            ),
            "train_event_count": torch.tensor(self.train_event_count, dtype=torch.long),
        }


def fit_node_feature_transform(
    node_time: torch.Tensor,
    event_patient_index: torch.Tensor,
    train_patient_indices: Sequence[int],
    temporal_masks: "V12TemporalMasks",
) -> NodeFeatureFoldTransformV12:
    """Fit robust scaling and deterministic PCA16 on outer-train patients only."""

    _validate_node_time(
        node_time, feature_dim=V12_NODE_RAW_DIM, name="raw node-time tokens"
    )
    _validate_masked_delta(node_time, temporal_masks, name="raw delta-H")
    n_patients = _validate_patient_roster(
        event_patient_index,
        events=int(node_time.shape[0]),
        device=node_time.device,
    )
    selected = tuple(int(value) for value in train_patient_indices)
    if not selected or len(set(selected)) != len(selected) or any(
        value < 0 or value >= n_patients for value in selected
    ):
        raise ValueError("train_patient_indices must be unique valid patients")
    patient_selector = torch.zeros(
        n_patients, dtype=torch.bool, device=node_time.device
    )
    patient_selector[
        torch.tensor(selected, dtype=torch.long, device=node_time.device)
    ] = True
    event_selector = patient_selector[event_patient_index]
    train_event_count = int(event_selector.sum().item())
    selected_tokens = node_time[event_selector][
        :, V12_CANDIDATE_INDICES
    ].permute(0, 2, 1, 3)
    selected_valid = temporal_masks.ictal_valid_mask[event_selector]
    rows = selected_tokens[selected_valid].reshape(-1, V12_NODE_RAW_DIM).float()
    if rows.shape[0] <= V12_NODE_PCA_DIM:
        raise ValueError("not enough outer-train node-time rows for PCA16")

    center, scale = _robust_center_scale(rows)
    standardized = (rows - center) / scale
    pca_mean = standardized.mean(dim=0)
    centered = standardized - pca_mean
    covariance = centered.transpose(0, 1).matmul(centered) / float(
        centered.shape[0] - 1
    )
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -V12_NODE_PCA_DIM:].flip(dims=(1,)).contiguous()
    # Pin the arbitrary eigenvector sign for reproducible fold artifacts.
    for column in range(V12_NODE_PCA_DIM):
        vector = components[:, column]
        pivot = int(vector.abs().argmax().item())
        if vector[pivot] < 0:
            components[:, column] = -vector

    return NodeFeatureFoldTransformV12(
        center=center.detach().cpu(),
        scale=scale.detach().cpu(),
        pca_mean=pca_mean.detach().cpu(),
        components=components.detach().cpu(),
        train_patient_indices=selected,
        train_event_count=train_event_count,
    )


@dataclass(frozen=True)
class V12TemporalMasks:
    """Strict one-second baseline and true-stop-aware ictal gate masks.

    The preictal baseline contains all 12 audited seconds; events without that
    clean context must be excluded before v12.  Ictal validity is a non-empty
    prefix beginning at the global t0 (index 12); no transition or postictal
    second may be re-enabled after the audited stop boundary.
    """

    preictal_baseline_mask: torch.Tensor
    ictal_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("preictal_baseline_mask", self.preictal_baseline_mask),
            ("ictal_valid_mask", self.ictal_valid_mask),
        ):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                raise TypeError(f"{name} must be a bool tensor")
            if value.ndim != 2 or value.shape[1] != V12_N_SECONDS:
                raise ValueError(f"{name} must have shape [E,60]")
        if self.preictal_baseline_mask.shape != self.ictal_valid_mask.shape:
            raise ValueError("v12 temporal masks must align")
        if self.preictal_baseline_mask.shape[0] < 1:
            raise ValueError("v12 temporal masks need at least one event")
        if self.preictal_baseline_mask.device != self.ictal_valid_mask.device:
            raise ValueError("v12 temporal masks must share a device")
        pre = self.preictal_baseline_mask
        ictal = self.ictal_valid_mask
        if pre[:, V12_PREICTAL_SECONDS:].any():
            raise ValueError("preictal baseline cannot extend past global t0")
        pre_count = pre[:, :V12_PREICTAL_SECONDS].sum(dim=1)
        if not torch.all(pre_count == V12_PREICTAL_SECONDS):
            raise ValueError("preictal baseline must contain all 12 audited seconds")
        if ictal[:, :V12_PREICTAL_SECONDS].any():
            raise ValueError("ictal gate mask cannot include pre-anchor seconds")
        post = ictal[:, V12_PREICTAL_SECONDS:]
        if not post[:, 0].all():
            raise ValueError("every ictal mask must begin at the global t0")
        # Once the true-stop mask becomes false, it must remain false.
        if ((~post[:, :-1]) & post[:, 1:]).any():
            raise ValueError("ictal mask must be a contiguous true-stop prefix")

    @property
    def n_events(self) -> int:
        return int(self.ictal_valid_mask.shape[0])

    @property
    def ictal_stop_index_exclusive(self) -> torch.Tensor:
        return V12_PREICTAL_SECONDS + self.ictal_valid_mask[
            :, V12_PREICTAL_SECONDS:
        ].sum(dim=1)

    def to(self, device: str | torch.device) -> "V12TemporalMasks":
        return V12TemporalMasks(
            preictal_baseline_mask=self.preictal_baseline_mask.to(device=device),
            ictal_valid_mask=self.ictal_valid_mask.to(device=device),
        )


@dataclass(frozen=True)
class TimeResolvedEventOutputV12:
    """Finite event logits plus complete temporal-gate audit traces."""

    event_logits: torch.Tensor
    evidence_logits: torch.Tensor
    node_time_logits: torch.Tensor
    global_trajectory: torch.Tensor
    onset_derivative: torch.Tensor
    gate_weights: torch.Tensor
    uniform_gate_weights: torch.Tensor
    baseline_score: torch.Tensor
    ictal_valid_mask: torch.Tensor
    gate_mode: GateMode

    def __post_init__(self) -> None:
        events = int(self.event_logits.shape[0])
        expected = {
            "event_logits": (events, N_STANDARD_CHANNELS),
            "evidence_logits": (events, N_STANDARD_CHANNELS),
            "node_time_logits": (
                events,
                N_STANDARD_CHANNELS,
                V12_N_SECONDS,
            ),
            "global_trajectory": (events, V12_N_SECONDS),
            "onset_derivative": (events, V12_N_SECONDS),
            "gate_weights": (events, V12_N_SECONDS),
            "uniform_gate_weights": (events, V12_N_SECONDS),
            "baseline_score": (events,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not value.is_floating_point() or (
                not torch.isfinite(value).all()
            ):
                raise ValueError(f"v12 event output {name} has invalid shape/value")
        if tuple(self.ictal_valid_mask.shape) != (events, V12_N_SECONDS) or (
            self.ictal_valid_mask.dtype != torch.bool
        ):
            raise TypeError("ictal_valid_mask must be bool [E,60]")
        if self.gate_mode not in ("uniform", "learned"):
            raise ValueError("unsupported v12 gate mode")
        for weights in (self.gate_weights, self.uniform_gate_weights):
            if torch.any(weights < 0) or not torch.allclose(
                weights.sum(dim=1),
                torch.ones(events, dtype=weights.dtype, device=weights.device),
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise ValueError("v12 gate weights must be nonnegative and normalized")
            if torch.any(weights.masked_select(~self.ictal_valid_mask) != 0):
                raise ValueError("v12 gate mass cannot cross the true-stop mask")

    def ranking_logits(self) -> torch.Tensor:
        """Return 18-candidate event logits; PZ is ``-inf`` only in this view."""

        return apply_fixed_candidate_mask(self.event_logits)


class TimeResolvedNodeLocalizerV12(nn.Module):
    """A 16/32-parameter shared node scorer with a matched temporal gate.

    ``uniform`` has one trainable ``node_weight`` and a same-shaped frozen-zero
    ``gate_weight``.  ``learned`` makes that gate vector trainable.  There are
    no biases, channel identities, graph terms, region heads, or hidden MLPs.
    """

    def __init__(self, prior_logits: torch.Tensor, *, gate_mode: GateMode) -> None:
        super().__init__()
        if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,) or (
            not prior_logits.is_floating_point()
        ) or not torch.isfinite(prior_logits).all():
            raise ValueError("prior_logits must be finite floating point [19]")
        if gate_mode not in ("uniform", "learned"):
            raise ValueError("gate_mode must be 'uniform' or 'learned'")
        self.gate_mode: GateMode = gate_mode
        self.node_weight = nn.Parameter(torch.zeros(V12_NODE_PCA_DIM))
        self.gate_weight = nn.Parameter(
            torch.zeros(V12_NODE_PCA_DIM),
            requires_grad=gate_mode == "learned",
        )
        self.register_buffer(
            "prior_logits", prior_logits.detach().float().contiguous()
        )
        self.register_buffer("candidate_mask", V12_CANDIDATE_MASK.clone())

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(
        self,
        node_time_features: torch.Tensor,
        temporal_masks: V12TemporalMasks,
    ) -> TimeResolvedEventOutputV12:
        if not isinstance(temporal_masks, V12TemporalMasks):
            raise TypeError("v12 localizer requires V12TemporalMasks")
        _validate_node_time(
            node_time_features,
            feature_dim=V12_NODE_PCA_DIM,
            name="transformed node-time features",
        )
        _validate_masked_delta(
            node_time_features,
            temporal_masks,
            name="transformed delta-H",
        )
        events = int(node_time_features.shape[0])
        if temporal_masks.n_events != events:
            raise ValueError("v12 features and temporal masks have different events")
        if temporal_masks.ictal_valid_mask.device != node_time_features.device:
            raise ValueError("v12 features and temporal masks must share a device")
        if self.gate_mode == "uniform" and (
            self.gate_weight.requires_grad or bool((self.gate_weight != 0).any())
        ):
            raise RuntimeError("uniform matched control requires a frozen-zero gate")

        dtype = node_time_features.dtype
        node_weight = self.node_weight.to(dtype=dtype)
        gate_weight = self.gate_weight.to(dtype=dtype)
        node_time_logits = torch.einsum(
            "ectd,d->ect", node_time_features, node_weight
        )

        # PZ is a signal carrier only.  Its features cannot alter the common
        # trajectory or any of the 18 evaluable candidate logits.
        candidate_features = node_time_features[:, V12_CANDIDATE_INDICES]
        global_features = candidate_features.mean(dim=1)
        trajectory = torch.sigmoid(
            torch.einsum("etd,d->et", global_features, gate_weight)
        )

        # Delta-H is zero outside the valid ictal support, hence sigmoid(0)
        # gives the fixed no-change reference q=0.5 for the first ictal second.
        baseline = torch.full(
            (events,), 0.5, dtype=dtype, device=node_time_features.device
        )

        derivative = torch.zeros_like(trajectory)
        onset_index = V12_PREICTAL_SECONDS
        derivative[:, onset_index] = trajectory[:, onset_index] - baseline
        derivative[:, onset_index + 1 :] = (
            trajectory[:, onset_index + 1 :] - trajectory[:, onset_index:-1]
        )
        derivative = torch.where(
            temporal_masks.ictal_valid_mask,
            derivative,
            torch.zeros_like(derivative),
        )

        valid = temporal_masks.ictal_valid_mask.to(dtype)
        uniform = valid / valid.sum(dim=1, keepdim=True)
        learned_mass = (F.softplus(derivative) + V12_GATE_FLOOR) * valid
        learned = learned_mass / learned_mass.sum(dim=1, keepdim=True)
        weights = uniform if self.gate_mode == "uniform" else learned
        evidence_logits = torch.einsum("ect,et->ec", node_time_logits, weights)
        prior = self.prior_logits.to(
            device=node_time_features.device, dtype=dtype
        ).expand(events, -1)
        event_logits = prior + evidence_logits
        if not torch.isfinite(event_logits).all():
            raise RuntimeError("v12 event logits must remain finite on all 19 carriers")
        return TimeResolvedEventOutputV12(
            event_logits=event_logits,
            evidence_logits=evidence_logits,
            node_time_logits=node_time_logits,
            global_trajectory=trajectory,
            onset_derivative=derivative,
            gate_weights=weights,
            uniform_gate_weights=uniform,
            baseline_score=baseline,
            ictal_valid_mask=temporal_masks.ictal_valid_mask,
            gate_mode=self.gate_mode,
        )


@dataclass(frozen=True)
class V12PatientAggregation:
    """Differentiable robust aggregation of complete event bags."""

    logits: torch.Tensor
    dispersion: torch.Tensor
    event_counts: torch.Tensor
    reliability_sum: torch.Tensor

    def __post_init__(self) -> None:
        if self.logits.ndim != 2 or self.logits.shape[1] != N_STANDARD_CHANNELS:
            raise ValueError("patient logits must have shape [P,19]")
        patients = int(self.logits.shape[0])
        if tuple(self.dispersion.shape) != tuple(self.logits.shape):
            raise ValueError("patient dispersion must match logits")
        if tuple(self.reliability_sum.shape) != tuple(self.logits.shape):
            raise ValueError("patient reliability_sum must have shape [P,19]")
        if tuple(self.event_counts.shape) != (patients,) or (
            self.event_counts.dtype != torch.long
        ):
            raise TypeError("patient event_counts must be long [P]")
        for value in (self.logits, self.dispersion, self.reliability_sum):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("patient aggregation must remain finite")
        if torch.any(self.event_counts < 1):
            raise ValueError("every patient needs a complete non-empty event bag")

    def ranking_logits(self) -> torch.Tensor:
        return apply_fixed_candidate_mask(self.logits)


def robust_aggregate_patient_logits(
    event_logits: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
    reliability: torch.Tensor,
) -> V12PatientAggregation:
    """Aggregate event logits using external, target-free quality reliability.

    ``reliability`` must be produced without DeepSOZ/private targets (the v12
    runner derives it from fine-evidence artifact burden).  It is deliberately
    an explicit input rather than an embedding-derived learned confidence.
    Every event remains in its complete bag through the fixed 0.1 weight floor.
    Bags of at least three events are winsorized at their per-channel 10th and
    90th percentiles before the reliability-weighted mean.
    """

    _require_float(event_logits, name="event logits", detached=False)
    if event_logits.ndim != 2 or event_logits.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("event_logits must have shape [E,19]")
    if isinstance(n_patients, bool) or not isinstance(n_patients, int) or (
        n_patients < 1
    ):
        raise ValueError("n_patients must be a positive integer")
    roster_patients = _validate_patient_roster(
        event_patient_index,
        events=int(event_logits.shape[0]),
        device=event_logits.device,
    )
    if roster_patients != n_patients:
        raise ValueError("n_patients disagrees with event_patient_index")
    _require_float(reliability, name="target-free reliability", detached=True)
    if tuple(reliability.shape) != tuple(event_logits.shape):
        raise ValueError("target-free reliability must have shape [E,19]")
    if reliability.device != event_logits.device:
        raise ValueError("reliability and event logits must share a device")
    if torch.any((reliability < 0) | (reliability > 1)):
        raise ValueError("target-free reliability must lie in [0,1]")
    weights = reliability.clamp_min(V12_RELIABILITY_FLOOR)

    patient_logits: list[torch.Tensor] = []
    patient_dispersion: list[torch.Tensor] = []
    event_counts: list[int] = []
    reliability_sums: list[torch.Tensor] = []
    for patient in range(n_patients):
        selected = event_patient_index == patient
        values = event_logits[selected]
        patient_weights = weights[selected]
        count = int(values.shape[0])
        if count >= 3:
            lower = torch.quantile(values, 0.1, dim=0)
            upper = torch.quantile(values, 0.9, dim=0)
            values = torch.minimum(torch.maximum(values, lower), upper)
        denominator = patient_weights.sum(dim=0).clamp_min(1.0e-6)
        mean = (values * patient_weights).sum(dim=0) / denominator
        variance = (
            (values - mean.unsqueeze(0)).square() * patient_weights
        ).sum(dim=0) / denominator
        patient_logits.append(mean)
        patient_dispersion.append(variance.clamp_min(0).sqrt())
        event_counts.append(count)
        reliability_sums.append(denominator)
    return V12PatientAggregation(
        logits=torch.stack(patient_logits).contiguous(),
        dispersion=torch.stack(patient_dispersion).contiguous(),
        event_counts=torch.tensor(
            event_counts, dtype=torch.long, device=event_logits.device
        ),
        reliability_sum=torch.stack(reliability_sums).contiguous(),
    )


__all__ = [
    "NodeFeatureFoldTransformV12",
    "TimeResolvedEventOutputV12",
    "TimeResolvedNodeLocalizerV12",
    "V12PatientAggregation",
    "V12TemporalMasks",
    "V12_CANDIDATE_INDICES",
    "V12_CANDIDATE_MASK",
    "V12_GATE_FLOOR",
    "V12_NODE_PCA_DIM",
    "V12_NODE_RAW_DIM",
    "V12_N_SECONDS",
    "V12_PREICTAL_SECONDS",
    "V12_RELIABILITY_FLOOR",
    "V12_TIME_RESOLVED_SCHEMA",
    "apply_fixed_candidate_mask",
    "baseline_difference_node_time",
    "fit_node_feature_transform",
    "jeffreys_reference_prior_logits",
    "positive_set_mass_loss",
    "restore_prefix_node_time",
    "robust_aggregate_patient_logits",
]
