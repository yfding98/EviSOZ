"""Low-capacity patient-level reasoner for the v11 LaBraM recovery trial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import torch
import torch.nn as nn

from .fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES
from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS


V11_BLOCK9_TOKEN_DIM = 200
V11_H_RAW_DIM = 3 * V11_BLOCK9_TOKEN_DIM
V11_H_PCA_DIM = 16
V11_FINE_DIM = len(FINE_TEMPORAL_FEATURE_NAMES)
V11_CANDIDATE_INDICES: Final[tuple[int, ...]] = tuple(
    index for index in range(N_STANDARD_CHANNELS) if index != CHANNEL_INDEX["PZ"]
)
V11_CANDIDATE_MASK: Final[torch.Tensor] = torch.tensor(
    tuple(index in V11_CANDIDATE_INDICES for index in range(N_STANDARD_CHANNELS)),
    dtype=torch.bool,
)


def _validate_candidate_mask(
    target_mask: torch.Tensor,
    *,
    allow_candidate_subset: bool,
) -> None:
    if target_mask.ndim != 2 or target_mask.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("candidate mask must have shape [P,19]")
    if target_mask.dtype != torch.bool:
        raise TypeError("candidate mask must be torch.bool")
    fixed = V11_CANDIDATE_MASK.to(device=target_mask.device).expand_as(target_mask)
    if allow_candidate_subset:
        if bool((target_mask & ~fixed).any()):
            raise ValueError("candidate subset cannot include PZ")
    elif not torch.equal(target_mask, fixed):
        raise ValueError("every primary target row must use the fixed 18-candidate mask")


def apply_fixed_candidate_mask(logits: torch.Tensor) -> torch.Tensor:
    """Mask the non-evaluable PZ carrier before any deployment ranking."""

    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("candidate logits must be floating point")
    if logits.ndim < 1 or logits.shape[-1] != N_STANDARD_CHANNELS:
        raise ValueError("candidate logits must end in the 19-channel carrier")
    if not torch.isfinite(logits).all():
        raise ValueError("candidate logits must be finite before masking")
    mask = V11_CANDIDATE_MASK.to(device=logits.device)
    return logits.masked_fill(~mask, -torch.inf)


def _require_finite_float(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def extract_block9_phase_contrasts(prefix: torch.Tensor) -> torch.Tensor:
    """Return three node-wise temporal contrasts from a block-9 prefix.

    Input is ``[E,15,77,200]`` (15 independent four-second calls).  Output is
    detached ``[E,19,600]`` with onset-minus-baseline, early-minus-baseline,
    and late-minus-early contrasts.  These phases are signal descriptors, not
    SOZ-onset or propagation targets.
    """

    _require_finite_float(prefix, name="block-9 prefix")
    if prefix.ndim != 4 or tuple(prefix.shape[1:]) != (15, 77, 200):
        raise ValueError("block-9 prefix must have shape [E,15,77,200]")
    if prefix.shape[0] < 1:
        raise ValueError("block-9 prefix must contain at least one event")
    events = int(prefix.shape[0])
    tokens = (
        prefix[:, :, 1:, :]
        .reshape(events, 15, N_STANDARD_CHANNELS, 4, V11_BLOCK9_TOKEN_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(events, N_STANDARD_CHANNELS, 60, V11_BLOCK9_TOKEN_DIM)
    )
    baseline = tokens[:, :, 0:12].mean(dim=2)
    onset = tokens[:, :, 12:20].mean(dim=2)
    early = tokens[:, :, 12:28].mean(dim=2)
    late = tokens[:, :, 28:52].mean(dim=2)
    result = torch.cat((onset - baseline, early - baseline, late - early), dim=-1)
    if tuple(result.shape) != (events, N_STANDARD_CHANNELS, V11_H_RAW_DIM):
        raise RuntimeError("block-9 phase contrast shape drifted")
    return result.detach().contiguous()


@dataclass(frozen=True)
class PatientPooledEvidence:
    features: torch.Tensor
    dispersion: torch.Tensor
    event_counts: torch.Tensor
    reliability_sum: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 3 or self.features.shape[1] != N_STANDARD_CHANNELS:
            raise ValueError("patient-pooled features must have shape [P,19,D]")
        if tuple(self.dispersion.shape) != tuple(self.features.shape):
            raise ValueError("patient-pooled dispersion must match features")
        patients = int(self.features.shape[0])
        if tuple(self.event_counts.shape) != (patients,) or (
            self.event_counts.dtype != torch.long
        ):
            raise TypeError("event_counts must be long [P]")
        if tuple(self.reliability_sum.shape) != (patients, N_STANDARD_CHANNELS):
            raise ValueError("reliability_sum must have shape [P,19]")
        if torch.any(self.event_counts < 1):
            raise ValueError("every patient must own a complete non-empty event bag")
        for value in (self.features, self.dispersion, self.reliability_sum):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("patient-pooled values must be finite floating point")


def robust_pool_complete_patient_bags(
    event_features: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
    reliability: torch.Tensor | None = None,
) -> PatientPooledEvidence:
    """Reliability-weight and winsorize complete event bags per patient.

    Patients, rather than events, are the independent SOZ units.  This
    operation never selects a best seizure.  For bags with three or more
    events, values are winsorized at the 10th/90th percentiles before the
    weighted mean; smaller bags use the weighted mean directly.
    """

    _require_finite_float(event_features, name="event_features")
    if event_features.ndim != 3 or event_features.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("event_features must have shape [E,19,D]")
    events = int(event_features.shape[0])
    if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer")
    if tuple(event_patient_index.shape) != (events,) or (
        event_patient_index.dtype != torch.long
    ):
        raise TypeError("event_patient_index must be long [E]")
    if event_patient_index.device != event_features.device:
        raise ValueError("event_patient_index and features must share a device")
    if events < n_patients or event_patient_index.min().item() != 0 or (
        event_patient_index.max().item() != n_patients - 1
    ):
        raise ValueError("event_patient_index is not a contiguous complete roster")
    if torch.unique(event_patient_index).numel() != n_patients:
        raise ValueError("every patient must have at least one event")
    if reliability is None:
        weights = torch.ones(
            (events, N_STANDARD_CHANNELS),
            dtype=event_features.dtype,
            device=event_features.device,
        )
    else:
        _require_finite_float(reliability, name="reliability")
        if tuple(reliability.shape) != (events, N_STANDARD_CHANNELS):
            raise ValueError("reliability must have shape [E,19]")
        if reliability.device != event_features.device:
            raise ValueError("reliability and features must share a device")
        if torch.any((reliability < 0) | (reliability > 1)):
            raise ValueError("reliability must lie in [0,1]")
        weights = reliability.clamp_min(0.1)

    pooled = []
    dispersion = []
    counts = []
    weight_sums = []
    for patient in range(n_patients):
        selector = event_patient_index == patient
        values = event_features[selector]
        patient_weights = weights[selector]
        count = int(values.shape[0])
        if count >= 3:
            lower = torch.quantile(values, 0.1, dim=0)
            upper = torch.quantile(values, 0.9, dim=0)
            values = torch.minimum(torch.maximum(values, lower), upper)
        denominator = patient_weights.sum(dim=0).clamp_min(1e-6)
        mean = (
            values * patient_weights.unsqueeze(-1)
        ).sum(dim=0) / denominator.unsqueeze(-1)
        variance = (
            (values - mean.unsqueeze(0)).square()
            * patient_weights.unsqueeze(-1)
        ).sum(dim=0) / denominator.unsqueeze(-1)
        pooled.append(mean)
        dispersion.append(variance.clamp_min(0.0).sqrt())
        counts.append(count)
        weight_sums.append(denominator)
    return PatientPooledEvidence(
        features=torch.stack(pooled).contiguous(),
        dispersion=torch.stack(dispersion).contiguous(),
        event_counts=torch.tensor(
            counts, dtype=torch.long, device=event_features.device
        ),
        reliability_sum=torch.stack(weight_sums).contiguous(),
    )


def _robust_center_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = values.median(dim=0).values
    mad = (values - center).abs().median(dim=0).values
    standard = values.std(dim=0, unbiased=False)
    scale = torch.maximum(1.4826 * mad, 0.1 * standard).clamp_min(1e-4)
    return center, scale


@dataclass(frozen=True)
class TransformedPatientFeatures:
    h: torch.Tensor
    fine: torch.Tensor

    def __post_init__(self) -> None:
        if self.h.ndim != 3 or tuple(self.h.shape[1:]) != (
            N_STANDARD_CHANNELS,
            V11_H_PCA_DIM,
        ):
            raise ValueError("transformed H must have shape [P,19,16]")
        if self.fine.ndim != 3 or tuple(self.fine.shape[1:]) != (
            N_STANDARD_CHANNELS,
            V11_FINE_DIM,
        ):
            raise ValueError("transformed fine features must have shape [P,19,20]")
        if self.h.shape[0] != self.fine.shape[0]:
            raise ValueError("transformed H/fine patient counts differ")
        for value in (self.h, self.fine):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("transformed features must be finite floating point")

    def index_select(self, indices: torch.Tensor) -> "TransformedPatientFeatures":
        if indices.dtype != torch.long or indices.ndim != 1:
            raise TypeError("patient indices must be long [K]")
        return TransformedPatientFeatures(
            h=self.h.index_select(0, indices),
            fine=self.fine.index_select(0, indices),
        )


@dataclass(frozen=True)
class FoldFeatureTransform:
    h_center: torch.Tensor
    h_scale: torch.Tensor
    h_pca_mean: torch.Tensor
    h_components: torch.Tensor
    fine_center: torch.Tensor
    fine_scale: torch.Tensor
    train_patient_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = {
            "h_center": (V11_H_RAW_DIM,),
            "h_scale": (V11_H_RAW_DIM,),
            "h_pca_mean": (V11_H_RAW_DIM,),
            "h_components": (V11_H_RAW_DIM, V11_H_PCA_DIM),
            "fine_center": (V11_FINE_DIM,),
            "fine_scale": (V11_FINE_DIM,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not value.is_floating_point() or (
                not torch.isfinite(value).all()
            ):
                raise ValueError(f"fold transform {name} has an invalid shape/value")
        if torch.any(self.h_scale <= 0) or torch.any(self.fine_scale <= 0):
            raise ValueError("fold transform scales must be positive")
        if not self.train_patient_indices or len(set(self.train_patient_indices)) != len(
            self.train_patient_indices
        ):
            raise ValueError("fold transform requires unique training patients")

    def apply(
        self,
        h_patient: torch.Tensor,
        fine_patient: torch.Tensor,
    ) -> TransformedPatientFeatures:
        _validate_patient_feature_inputs(h_patient, fine_patient)
        device = h_patient.device
        dtype = h_patient.dtype
        h_center = self.h_center.to(device=device, dtype=dtype)
        h_scale = self.h_scale.to(device=device, dtype=dtype)
        h_mean = self.h_pca_mean.to(device=device, dtype=dtype)
        components = self.h_components.to(device=device, dtype=dtype)
        fine_center = self.fine_center.to(device=device, dtype=fine_patient.dtype)
        fine_scale = self.fine_scale.to(device=device, dtype=fine_patient.dtype)
        h_z = (h_patient - h_center) / h_scale
        h = torch.matmul(h_z - h_mean, components)
        fine = (fine_patient - fine_center) / fine_scale
        return TransformedPatientFeatures(h=h.contiguous(), fine=fine.contiguous())


def _validate_patient_feature_inputs(
    h_patient: torch.Tensor,
    fine_patient: torch.Tensor,
) -> None:
    _require_finite_float(h_patient, name="patient H")
    _require_finite_float(fine_patient, name="patient fine features")
    if h_patient.ndim != 3 or tuple(h_patient.shape[1:]) != (
        N_STANDARD_CHANNELS,
        V11_H_RAW_DIM,
    ):
        raise ValueError("patient H must have shape [P,19,600]")
    if fine_patient.ndim != 3 or tuple(fine_patient.shape[1:]) != (
        N_STANDARD_CHANNELS,
        V11_FINE_DIM,
    ):
        raise ValueError("patient fine features must have shape [P,19,20]")
    if h_patient.shape[0] != fine_patient.shape[0] or (
        h_patient.device != fine_patient.device
    ):
        raise ValueError("patient H/fine carriers must align")


def fit_fold_transform(
    h_patient: torch.Tensor,
    fine_patient: torch.Tensor,
    train_indices: Sequence[int],
) -> FoldFeatureTransform:
    """Fit robust scaling and PCA from training patients only."""

    _validate_patient_feature_inputs(h_patient, fine_patient)
    selected = tuple(int(value) for value in train_indices)
    patients = int(h_patient.shape[0])
    if not selected or len(set(selected)) != len(selected) or any(
        value < 0 or value >= patients for value in selected
    ):
        raise ValueError("train_indices must be unique valid patient indices")
    index = torch.tensor(selected, dtype=torch.long, device=h_patient.device)
    h_rows = h_patient.index_select(0, index).reshape(-1, V11_H_RAW_DIM)
    fine_rows = fine_patient.index_select(0, index).reshape(-1, V11_FINE_DIM)
    if h_rows.shape[0] <= V11_H_PCA_DIM:
        raise ValueError("not enough training patient-channel rows for PCA16")
    h_center, h_scale = _robust_center_scale(h_rows)
    h_z = (h_rows - h_center) / h_scale
    h_pca_mean = h_z.mean(dim=0)
    centered = h_z - h_pca_mean
    # Full deterministic SVD is affordable for the small patient-level matrix
    # and avoids stochastic PCA leakage/reproducibility ambiguity.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:V11_H_PCA_DIM].transpose(0, 1).contiguous()
    # SVD signs are arbitrary.  Pin each component by its largest loading.
    for column in range(V11_H_PCA_DIM):
        vector = components[:, column]
        pivot = int(vector.abs().argmax().item())
        if vector[pivot] < 0:
            components[:, column] = -vector
    fine_center, fine_scale = _robust_center_scale(fine_rows)
    return FoldFeatureTransform(
        h_center=h_center.detach().cpu(),
        h_scale=h_scale.detach().cpu(),
        h_pca_mean=h_pca_mean.detach().cpu(),
        h_components=components.detach().cpu(),
        fine_center=fine_center.detach().cpu(),
        fine_scale=fine_scale.detach().cpu(),
        train_patient_indices=selected,
    )


def jeffreys_reference_prior_logits(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    allow_candidate_subset: bool = False,
) -> torch.Tensor:
    if targets.ndim != 2 or tuple(targets.shape[1:]) != (N_STANDARD_CHANNELS,) or (
        tuple(target_mask.shape) != tuple(targets.shape)
    ):
        raise ValueError("targets/mask must have shape [P,19]")
    if target_mask.dtype != torch.bool or not targets.is_floating_point():
        raise TypeError("targets must be floating point and mask bool")
    _validate_candidate_mask(
        target_mask, allow_candidate_subset=allow_candidate_subset
    )
    observed_values = targets[target_mask]
    if not torch.isfinite(observed_values).all() or (
        observed_values.numel()
        and not torch.all((observed_values == 0) | (observed_values == 1))
    ):
        raise ValueError("observed targets must be finite binary values")
    positive = ((targets == 1) & target_mask).sum(dim=0).float()
    observed = target_mask.sum(dim=0).float()
    prevalence = (positive + 0.5) / (observed + 1.0)
    return torch.logit(prevalence.clamp(1e-4, 1 - 1e-4)).detach()


@dataclass(frozen=True)
class V11ReasonerOutput:
    logits: torch.Tensor
    prior: torch.Tensor
    h_contribution: torch.Tensor
    fine_contribution: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return self.prior + self.h_contribution + self.fine_contribution


class SharedPositiveSetReasoner(nn.Module):
    """Shared scorer on a 19-node carrier with a fixed 18-candidate output."""

    def __init__(
        self,
        prior_logits: torch.Tensor,
        *,
        use_h: bool = True,
        use_fine: bool = True,
    ) -> None:
        super().__init__()
        if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,) or (
            not prior_logits.is_floating_point()
        ) or not torch.isfinite(prior_logits).all():
            raise ValueError("prior_logits must be finite [19]")
        if not use_h and not use_fine:
            raise ValueError("reasoner must use at least one evidence family")
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
        self.register_buffer("prior_logits", prior_logits.detach().float().contiguous())
        self.register_buffer("candidate_mask", V11_CANDIDATE_MASK.clone())

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(self, evidence: TransformedPatientFeatures) -> V11ReasonerOutput:
        if not isinstance(evidence, TransformedPatientFeatures):
            raise TypeError("v11 reasoner accepts transformed evidence only")
        patients = int(evidence.h.shape[0])
        dtype = evidence.h.dtype
        device = evidence.h.device
        prior = self.prior_logits.to(device=device, dtype=dtype).expand(patients, -1)
        h = torch.zeros_like(prior)
        fine = torch.zeros_like(prior)
        if self.h_weight is not None:
            h = torch.einsum("pcd,d->pc", evidence.h, self.h_weight.to(dtype))
        if self.fine_weight is not None:
            fine = torch.einsum(
                "pcd,d->pc", evidence.fine, self.fine_weight.to(evidence.fine.dtype)
            ).to(dtype)
        output = V11ReasonerOutput(
            logits=prior + h + fine,
            prior=prior,
            h_contribution=h,
            fine_contribution=fine,
        )
        if not torch.allclose(
            output.logits,
            output.reconstructed_logits(),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("v11 reasoner contribution decomposition drifted")
        return output


def positive_set_mass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    allow_candidate_subset: bool = False,
) -> torch.Tensor:
    """Put ranking mass on each patient's observed clinical positive set."""

    if logits.ndim != 2 or logits.shape[1] != N_STANDARD_CHANNELS or (
        tuple(targets.shape) != tuple(logits.shape)
    ) or tuple(target_mask.shape) != tuple(logits.shape):
        raise ValueError("set-mass inputs must have aligned shape [P,19]")
    if target_mask.dtype != torch.bool or not targets.is_floating_point():
        raise TypeError("set-mass targets must be floating point and mask bool")
    _validate_candidate_mask(
        target_mask, allow_candidate_subset=allow_candidate_subset
    )
    if not torch.isfinite(logits).all() or not torch.isfinite(targets[target_mask]).all():
        raise ValueError("set-mass observed inputs must be finite")
    observed_targets = targets[target_mask]
    if not ((observed_targets == 0) | (observed_targets == 1)).all():
        raise ValueError("set-mass observed targets must be binary")
    rows = []
    for patient in range(logits.shape[0]):
        observed = target_mask[patient]
        positive = observed & (targets[patient] == 1)
        if not observed.any() or not positive.any():
            raise ValueError("set-mass loss requires an observed positive per patient")
        rows.append(
            torch.logsumexp(logits[patient, observed], dim=0)
            - torch.logsumexp(logits[patient, positive], dim=0)
        )
    return torch.stack(rows).mean()


__all__ = [
    "FoldFeatureTransform",
    "PatientPooledEvidence",
    "SharedPositiveSetReasoner",
    "TransformedPatientFeatures",
    "V11ReasonerOutput",
    "V11_FINE_DIM",
    "V11_H_PCA_DIM",
    "V11_H_RAW_DIM",
    "V11_CANDIDATE_INDICES",
    "V11_CANDIDATE_MASK",
    "apply_fixed_candidate_mask",
    "extract_block9_phase_contrasts",
    "fit_fold_transform",
    "jeffreys_reference_prior_logits",
    "positive_set_mass_loss",
    "robust_pool_complete_patient_bags",
]
