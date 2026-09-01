"""Target-free utilities for patient-disjoint temporal-future qualification.

The functions in this module operate only on directly computed temporal
descriptors and frozen foundation tokens.  They do not accept SOZ targets or
TUSZ channel-involvement labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


N_FEATURES = 6


@dataclass(frozen=True)
class EvolutionScalerSpec:
    """Frozen six-feature robust scaler used by one OOF fold."""

    center: torch.Tensor
    scale: torch.Tensor
    clip: float

    def validate(self) -> None:
        if tuple(self.center.shape) != (N_FEATURES,):
            raise ValueError("scaler center must have shape [6]")
        if tuple(self.scale.shape) != (N_FEATURES,):
            raise ValueError("scaler scale must have shape [6]")
        if not self.center.is_floating_point() or not self.scale.is_floating_point():
            raise TypeError("scaler center/scale must be floating point")
        if not torch.isfinite(self.center).all() or not torch.isfinite(self.scale).all():
            raise ValueError("scaler center/scale must be finite")
        if not torch.all(self.scale > 0):
            raise ValueError("scaler scale must be positive")
        if not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("scaler clip must be positive and finite")

    def transform(self, descriptors: torch.Tensor) -> torch.Tensor:
        self.validate()
        if descriptors.ndim != 4 or descriptors.shape[-1] != N_FEATURES:
            raise ValueError("descriptors must have shape [E,19,15,6]")
        center = self.center.to(device=descriptors.device, dtype=descriptors.dtype)
        scale = self.scale.to(device=descriptors.device, dtype=descriptors.dtype)
        transformed = (descriptors - center) / scale
        return transformed.clamp(min=-float(self.clip), max=float(self.clip))


def scaler_from_receipt(payload: Mapping[str, object]) -> EvolutionScalerSpec:
    """Parse the closed scaler receipt embedded in a verified artifact."""

    receipt = payload.get("scaler_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("scaler artifact lacks scaler_receipt")
    names = receipt.get("feature_names")
    expected_names = [
        "log_rms",
        "log_line_length",
        "spectral_centroid",
        "normalized_spectral_entropy",
        "rhythmicity",
        "mean_neighbor_coherence",
    ]
    if names != expected_names:
        raise ValueError("evolution scaler feature order changed")
    center = receipt.get("center")
    scale = receipt.get("scale")
    clip = receipt.get("clip")
    if not isinstance(center, list) or not isinstance(scale, list):
        raise TypeError("scaler center/scale must be JSON arrays")
    if isinstance(clip, bool) or not isinstance(clip, (int, float)):
        raise TypeError("scaler clip must be numeric")
    spec = EvolutionScalerSpec(
        center=torch.tensor(center, dtype=torch.float64),
        scale=torch.tensor(scale, dtype=torch.float64),
        clip=float(clip),
    )
    spec.validate()
    return spec


def future_targets_and_mask(
    descriptors: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if descriptors.ndim != 4 or descriptors.shape[-1] != N_FEATURES:
        raise ValueError("descriptors must have shape [E,19,T,6]")
    if tuple(mask.shape) != tuple(descriptors.shape[:-1]) or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [E,19,T]")
    if descriptors.shape[2] < 2:
        raise ValueError("at least two temporal tiles are required")
    target = descriptors[:, :, 1:] - descriptors[:, :, :-1]
    target_mask = mask[:, :, 1:] & mask[:, :, :-1]
    return target, target_mask


def _validate_patient_vector(patient_ids: torch.Tensor, n_events: int) -> None:
    if patient_ids.ndim != 1 or patient_ids.shape[0] != n_events:
        raise ValueError("patient_ids must have shape [E]")
    if patient_ids.dtype != torch.long:
        raise TypeError("patient_ids must be torch.long")


def patient_macro_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patient_ids: torch.Tensor,
) -> tuple[float, dict[int, float]]:
    """Return patient-macro Smooth-L1 and individual patient losses."""

    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[-1] != N_FEATURES:
        raise ValueError("prediction must have shape [E,19,T,6]")
    if tuple(mask.shape) != tuple(prediction.shape[:-1]) or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [E,19,T]")
    _validate_patient_vector(patient_ids, prediction.shape[0])
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("observed prediction/target values must be finite")
    element = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    losses: dict[int, float] = {}
    for patient in torch.unique(patient_ids, sorted=True).tolist():
        rows = patient_ids == int(patient)
        observed = mask[rows]
        if observed.any():
            losses[int(patient)] = float(element[rows][observed].mean().item())
    if not losses:
        raise ValueError("no observed patient losses")
    return float(np.mean(list(losses.values()))), losses


def patient_macro_feature_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patient_ids: torch.Tensor,
) -> list[float]:
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 4 or prediction.shape[-1] != N_FEATURES:
        raise ValueError("prediction must have shape [E,19,T,6]")
    if tuple(mask.shape) != tuple(prediction.shape[:-1]) or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [E,19,T]")
    _validate_patient_vector(patient_ids, prediction.shape[0])
    element = F.smooth_l1_loss(prediction, target, reduction="none")
    values: list[float] = []
    for feature in range(N_FEATURES):
        patient_values = []
        for patient in torch.unique(patient_ids, sorted=True):
            rows = patient_ids == patient
            observed = mask[rows]
            if observed.any():
                patient_values.append(
                    float(element[rows, ..., feature][observed].mean().item())
                )
        if not patient_values:
            raise ValueError("feature has no observed patient losses")
        values.append(float(np.mean(patient_values)))
    return values


def fit_patient_balanced_linear_ar(
    descriptors: torch.Tensor,
    mask: torch.Tensor,
    patient_ids: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Fit delta(t+1) from descriptor(t) with equal total patient weight."""

    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be positive and finite")
    _validate_patient_vector(patient_ids, descriptors.shape[0])
    if train_indices.ndim != 1 or train_indices.dtype != torch.long:
        raise TypeError("train_indices must be a long vector")
    current = descriptors.index_select(0, train_indices)[:, :, :-1]
    delta, future_mask = future_targets_and_mask(
        descriptors.index_select(0, train_indices), mask.index_select(0, train_indices)
    )
    train_patients = patient_ids.index_select(0, train_indices)
    x_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    for patient in torch.unique(train_patients, sorted=True):
        event_rows = train_patients == patient
        observed = future_mask[event_rows]
        x_patient = current[event_rows][observed]
        y_patient = delta[event_rows][observed]
        if x_patient.numel() == 0:
            continue
        x_rows.append(x_patient)
        y_rows.append(y_patient)
        weight_rows.append(
            torch.full(
                (x_patient.shape[0],),
                1.0 / float(x_patient.shape[0]),
                dtype=torch.float64,
            )
        )
    if not x_rows:
        raise ValueError("linear AR has no observed training rows")
    x = torch.cat(x_rows).to(torch.float64)
    y = torch.cat(y_rows).to(torch.float64)
    weights = torch.cat(weight_rows).to(torch.float64)
    design = torch.cat([torch.ones((x.shape[0], 1), dtype=x.dtype), x], dim=1)
    sqrt_weight = weights.sqrt().unsqueeze(1)
    weighted_x = design * sqrt_weight
    weighted_y = y * sqrt_weight
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * float(ridge)
    penalty[0, 0] = 0.0
    gram = weighted_x.T @ weighted_x + penalty
    rhs = weighted_x.T @ weighted_y
    return torch.linalg.solve(gram, rhs)


def predict_linear_ar(descriptors: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    if tuple(coefficients.shape) != (N_FEATURES + 1, N_FEATURES):
        raise ValueError("linear AR coefficients must have shape [7,6]")
    current = descriptors[:, :, :-1].to(torch.float64)
    design = torch.cat([torch.ones_like(current[..., :1]), current], dim=-1)
    return design @ coefficients.to(device=design.device, dtype=design.dtype)


def fit_patient_balanced_time_only(
    descriptors: torch.Tensor,
    mask: torch.Tensor,
    patient_ids: torch.Tensor,
    train_indices: torch.Tensor,
) -> torch.Tensor:
    """Fit a [T-1,6] relative-time-only mean with equal patient weight."""

    selected = descriptors.index_select(0, train_indices)
    selected_mask = mask.index_select(0, train_indices)
    selected_patients = patient_ids.index_select(0, train_indices)
    delta, future_mask = future_targets_and_mask(selected, selected_mask)
    patient_means: list[torch.Tensor] = []
    for patient in torch.unique(selected_patients, sorted=True):
        rows = selected_patients == patient
        means = []
        for tile in range(delta.shape[2]):
            observed = future_mask[rows, :, tile]
            values = delta[rows, :, tile][observed]
            if values.numel() == 0:
                means.append(torch.full((N_FEATURES,), float("nan"), dtype=delta.dtype))
            else:
                means.append(values.mean(dim=0))
        patient_means.append(torch.stack(means))
    stacked = torch.stack(patient_means)
    if not torch.isfinite(stacked).all():
        raise ValueError("time-only baseline has an empty patient/tile stratum")
    return stacked.mean(dim=0)


def bootstrap_paired_difference(
    candidate: Mapping[int, float],
    comparator: Mapping[int, float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Patient bootstrap for candidate-minus-comparator loss."""

    if replicates < 100:
        raise ValueError("bootstrap requires at least 100 replicates")
    patients = sorted(set(candidate) & set(comparator))
    if set(candidate) != set(comparator) or len(patients) < 2:
        raise ValueError("paired losses must cover the same patients")
    differences = np.asarray(
        [float(candidate[p]) - float(comparator[p]) for p in patients], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(patients), size=(int(replicates), len(patients)))
    bootstrap = differences[draws].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "lower95": float(np.quantile(bootstrap, 0.025)),
        "upper95": float(np.quantile(bootstrap, 0.975)),
    }


def temporal_predictability_decision(
    *,
    ensemble_loss: float,
    persistence_loss: float,
    linear_ar_loss: float,
    shuffled_loss: float,
    versus_persistence: Mapping[str, float],
    versus_linear_ar: Mapping[str, float],
    coverage_complete: bool,
    scaler_replay_exact: bool,
    safety_passed: bool,
) -> tuple[bool, dict[str, bool]]:
    gates = {
        "coverage_complete": bool(coverage_complete),
        "scaler_replay_exact": bool(scaler_replay_exact),
        "safety_passed": bool(safety_passed),
        "ensemble_better_than_persistence": ensemble_loss < persistence_loss,
        "ensemble_better_than_linear_ar": ensemble_loss < linear_ar_loss,
        "paired_upper95_below_zero_vs_persistence": float(
            versus_persistence["upper95"]
        )
        < 0.0,
        "paired_upper95_below_zero_vs_linear_ar": float(
            versus_linear_ar["upper95"]
        )
        < 0.0,
        "correct_future_better_than_shuffled": ensemble_loss < shuffled_loss,
    }
    return all(gates.values()), gates


def encode_patient_ids(values: Sequence[str]) -> tuple[torch.Tensor, dict[str, int]]:
    ordered = sorted(set(values))
    mapping = {patient: index for index, patient in enumerate(ordered)}
    encoded = torch.tensor([mapping[value] for value in values], dtype=torch.long)
    return encoded, mapping


__all__ = [
    "EvolutionScalerSpec",
    "bootstrap_paired_difference",
    "encode_patient_ids",
    "fit_patient_balanced_linear_ar",
    "fit_patient_balanced_time_only",
    "future_targets_and_mask",
    "patient_macro_feature_smooth_l1",
    "patient_macro_smooth_l1",
    "predict_linear_ar",
    "scaler_from_receipt",
    "temporal_predictability_decision",
]
