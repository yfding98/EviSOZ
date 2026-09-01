"""Frozen target-free numerical primitives for the v13 LaBraM I-gate probes.

Only detached score tensors, signal-derived masks, and anonymous public
patient IDs are accepted.  No target loader, label, recovery artifact,
promotion receipt, clinical identity, or clinical outcome is reachable from
this module.
"""

from __future__ import annotations

import numpy as np
import torch


ICTAL_SCALE_QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ICTAL_SCALE_QUANTILE_ESTIMATOR = (
    "mean_of_within_patient_linear_quantiles_equal_patient_weight_v1"
)
ICTAL_FOLD_IDENTITY_FEATURE_POLICY = (
    "oof_ictal_four_second_mean_max_phase_masked_patient_summary_v1"
)
ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION = 52


def patient_macro_scale_summary(
    scores: torch.Tensor,
    mask: torch.Tensor,
    patient_ids_by_event: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Compute equal-patient quantiles over signal-visible score cells."""

    patient_ids = tuple(sorted(set(patient_ids_by_event)))
    levels = torch.tensor(ICTAL_SCALE_QUANTILE_LEVELS, dtype=torch.float64)
    quantiles: list[torch.Tensor] = []
    counts: list[int] = []
    for patient_id in patient_ids:
        event_index = torch.tensor(
            [
                index
                for index, value in enumerate(patient_ids_by_event)
                if value == patient_id
            ],
            dtype=torch.long,
        )
        patient_values = scores.index_select(0, event_index)
        patient_mask = mask.index_select(0, event_index)
        observed = patient_values[patient_mask].to(torch.float64)
        if observed.numel() < 1:
            raise ValueError("Every shared-probe patient needs observed score cells")
        quantiles.append(torch.quantile(observed, levels))
        counts.append(int(observed.numel()))
    patient_macro = torch.stack(quantiles, dim=0).mean(dim=0)
    return tuple(float(value) for value in patient_macro), tuple(counts)


def balanced_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    """Five-class macro recall used by the frozen producer-ID probe."""

    recalls = []
    for label in range(5):
        selected = true == label
        if not np.any(selected):
            raise ValueError("Fold-identity metric requires all five classes")
        recalls.append(float(np.mean(predicted[selected] == label)))
    return float(np.mean(recalls))


def fixed_ridge_oof_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    probe_splits: np.ndarray,
) -> np.ndarray:
    """Fit the fixed five-split L2 multinomial ridge probe."""

    predictions = np.empty(labels.shape[0], dtype=np.int64)
    for split in range(5):
        test = probe_splits == split
        train = ~test
        if not np.any(test) or int(train.sum()) < 5:
            raise ValueError("Fold-identity probe split lacks train/test patients")
        train_x = features[train]
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized_train = (train_x - mean) / scale
        normalized_test = (features[test] - mean) / scale
        design_train = np.concatenate(
            [normalized_train, np.ones((normalized_train.shape[0], 1))], axis=1
        )
        design_test = np.concatenate(
            [normalized_test, np.ones((normalized_test.shape[0], 1))], axis=1
        )
        targets = np.eye(5, dtype=np.float64)[labels[train]]
        regularizer = np.eye(design_train.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        matrix = design_train.T @ design_train + regularizer
        weights = np.linalg.solve(matrix, design_train.T @ targets)
        predictions[test] = np.argmax(design_test @ weights, axis=1)
    return predictions


def masked_patient_fold_identity_features(
    scores: torch.Tensor,
    deployment_mask: torch.Tensor,
    phase_mask: torch.Tensor,
    patient_ids_by_event: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build the frozen 52-D patient rows from reasoner-visible I scores."""

    if (
        scores.dtype != torch.float32
        or scores.ndim != 3
        or tuple(scores.shape[1:]) != (20, 60)
        or not torch.isfinite(scores).all()
        or torch.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("Fold-ID scores must be finite float32 [E,20,60]")
    if (
        deployment_mask.dtype != torch.bool
        or tuple(deployment_mask.shape) != tuple(scores.shape)
        or phase_mask.dtype != torch.bool
        or tuple(phase_mask.shape) != (scores.shape[0], 15)
    ):
        raise ValueError("Fold-ID deployment/phase masks have invalid shape")
    if len(patient_ids_by_event) != scores.shape[0]:
        raise ValueError("Fold-ID event patients do not align with score rows")
    tile_values = scores.reshape(scores.shape[0], 20, 15, 4)
    evidence = torch.stack(
        (tile_values.mean(dim=-1), tile_values.amax(dim=-1)), dim=-1
    ).to(torch.float64)
    tile_available = deployment_mask.reshape(
        deployment_mask.shape[0], 20, 15, 4
    ).all(dim=-1)
    visible = tile_available & phase_mask[:, None, :]
    patients = tuple(sorted(set(patient_ids_by_event)))
    rows: list[torch.Tensor] = []
    quantile_levels = torch.tensor(
        ICTAL_SCALE_QUANTILE_LEVELS, dtype=torch.float64
    )
    for patient_id in patients:
        indices = torch.tensor(
            [
                index
                for index, value in enumerate(patient_ids_by_event)
                if value == patient_id
            ],
            dtype=torch.long,
        )
        patient_evidence = evidence.index_select(0, indices)
        patient_visible = visible.index_select(0, indices)
        denominator = patient_visible.sum(dim=(0, 2)).to(torch.float64)
        if torch.any(denominator < 1):
            raise ValueError(
                "Every fold-ID patient/edge needs reasoner-visible score support"
            )
        edge_summary = (
            (
                patient_evidence
                * patient_visible.unsqueeze(-1).to(torch.float64)
            ).sum(dim=(0, 2))
            / denominator.unsqueeze(-1)
        ).reshape(-1)
        distribution_summary: list[torch.Tensor] = []
        for feature_index in range(2):
            observed = patient_evidence[..., feature_index][patient_visible]
            if observed.numel() < 2:
                raise ValueError("Fold-ID patient has insufficient visible scores")
            distribution_summary.append(
                torch.quantile(observed, quantile_levels)
            )
            distribution_summary.append(observed.std(unbiased=False).reshape(1))
        row = torch.cat((edge_summary, *distribution_summary), dim=0)
        if row.numel() != ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION:
            raise RuntimeError("Frozen fold-ID feature width changed")
        rows.append(row)
    matrix = torch.stack(rows, dim=0).numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("Fold-ID patient feature matrix is non-finite")
    return patients, matrix


def patient_grouped_fold_identity_statistics(
    matrix: np.ndarray,
    labels: np.ndarray,
    patient_groups: np.ndarray,
    probe_splits: np.ndarray,
    *,
    seed: int,
    bootstrap_count: int,
    permutation_count: int,
) -> tuple[float, float, float, float, float, float]:
    """Compute patient-grouped bootstrap and within-patient permutation tests."""

    if (
        matrix.ndim != 2
        or labels.ndim != 1
        or patient_groups.ndim != 1
        or probe_splits.ndim != 1
        or not (
            matrix.shape[0]
            == labels.size
            == patient_groups.size
            == probe_splits.size
        )
    ):
        raise ValueError("Patient-grouped Fold-ID arrays do not align")
    patients = np.unique(patient_groups)
    if any(np.sum(patient_groups == patient) != 5 for patient in patients):
        raise ValueError("Each Fold-ID patient must contribute five producer rows")
    if any(
        set(labels[patient_groups == patient].tolist()) != set(range(5))
        for patient in patients
    ):
        raise ValueError(
            "Each patient must contribute exactly one row per fold producer"
        )
    if any(
        len(set(probe_splits[patient_groups == patient].tolist())) != 1
        for patient in patients
    ):
        raise ValueError("All producer rows for one patient must share one CV split")

    predictions = fixed_ridge_oof_predictions(matrix, labels, probe_splits)
    observed = balanced_accuracy(labels, predictions)
    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(bootstrap_count):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        sampled_rows = np.concatenate(
            [
                np.flatnonzero(patient_groups == patient)
                for patient in sampled_patients
            ]
        )
        bootstrap_values.append(
            balanced_accuracy(labels[sampled_rows], predictions[sampled_rows])
        )
    permutation_values: list[float] = []
    for _ in range(permutation_count):
        permuted = labels.copy()
        for patient in patients:
            rows = np.flatnonzero(patient_groups == patient)
            permuted[rows] = rng.permutation(labels[rows])
        permuted_predictions = fixed_ridge_oof_predictions(
            matrix, permuted, probe_splits
        )
        permutation_values.append(
            balanced_accuracy(permuted, permuted_predictions)
        )
    bootstrap = np.asarray(bootstrap_values, dtype=np.float64)
    permutation = np.asarray(permutation_values, dtype=np.float64)
    p_value = float(
        (1 + int(np.sum(permutation >= observed))) / (permutation_count + 1)
    )
    return (
        observed,
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
        float(permutation.mean()),
        float(np.quantile(permutation, 0.95)),
        p_value,
    )


__all__ = (
    "ICTAL_FOLD_IDENTITY_FEATURE_DIMENSION",
    "ICTAL_FOLD_IDENTITY_FEATURE_POLICY",
    "ICTAL_SCALE_QUANTILE_ESTIMATOR",
    "ICTAL_SCALE_QUANTILE_LEVELS",
    "balanced_accuracy",
    "fixed_ridge_oof_predictions",
    "masked_patient_fold_identity_features",
    "patient_grouped_fold_identity_statistics",
    "patient_macro_scale_summary",
)
