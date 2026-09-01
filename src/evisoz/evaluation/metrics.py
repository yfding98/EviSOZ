"""Small, deterministic metrics for EviSOZ prediction and report audits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def _probability_rows(probabilities: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,C] with N>=1 and C>=2")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    sums = values.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    return values


def _positive_sets(positive_sets: Iterable[Iterable[int]], *, count: int, classes: int) -> list[set[int]]:
    rows = [set(int(item) for item in values) for values in positive_sets]
    if len(rows) != count or any(not row or any(item < 0 or item >= classes for item in row) for row in rows):
        raise ValueError("positive sets must be non-empty and within class range")
    return rows


def top_k_candidate_hit(
    probabilities: Sequence[Sequence[float]],
    positive_sets: Iterable[Iterable[int]],
    *,
    k: int = 1,
) -> float:
    """Return candidate-set hit rate, treating a positive set as incomplete-safe."""

    if k < 1:
        raise ValueError("k must be positive")
    values = _probability_rows(probabilities)
    positives = _positive_sets(positive_sets, count=values.shape[0], classes=values.shape[1])
    hits = 0
    for row, allowed in zip(values, positives):
        top = np.argsort(-row, kind="stable")[:k]
        hits += int(bool(set(int(item) for item in top) & allowed))
    return float(hits / values.shape[0])


def mean_reciprocal_rank(
    probabilities: Sequence[Sequence[float]],
    positive_sets: Iterable[Iterable[int]],
) -> float:
    """Return MRR where the first member of an incomplete positive set counts."""

    values = _probability_rows(probabilities)
    positives = _positive_sets(positive_sets, count=values.shape[0], classes=values.shape[1])
    reciprocal = []
    for row, allowed in zip(values, positives):
        order = np.argsort(-row, kind="stable")
        ranks = [index for index, class_id in enumerate(order, start=1) if int(class_id) in allowed]
        reciprocal.append(1.0 / min(ranks))
    return float(np.mean(reciprocal))


def expected_calibration_error(
    confidence: Sequence[float],
    correctness: Sequence[bool],
    *,
    bins: int = 10,
) -> float:
    """Compute equal-width ECE without silently clipping invalid confidence."""

    conf = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correctness, dtype=np.float64)
    if conf.ndim != 1 or correct.ndim != 1 or conf.size == 0 or conf.size != correct.size:
        raise ValueError("confidence and correctness must be non-empty matching vectors")
    if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("confidence must lie in [0,1]")
    if not np.isin(correct, [0.0, 1.0]).all():
        raise ValueError("correctness must be boolean")
    if bins < 1:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (conf >= edges[index]) & (conf <= edges[index + 1] if index == bins - 1 else conf < edges[index + 1])
        if mask.any():
            total += float(mask.mean()) * abs(float(conf[mask].mean()) - float(correct[mask].mean()))
    return float(total)


def brier_score_multiclass(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> float:
    """Compute multiclass Brier score for one gold class per sample."""

    values = _probability_rows(probabilities)
    target = np.asarray(labels, dtype=np.int64)
    if target.ndim != 1 or target.size != values.shape[0] or ((target < 0) | (target >= values.shape[1])).any():
        raise ValueError("labels must match rows and be valid class indices")
    one_hot = np.zeros_like(values)
    one_hot[np.arange(values.shape[0]), target] = 1.0
    return float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))


def risk_coverage_curve(
    confidence: Sequence[float],
    correctness: Sequence[bool],
) -> list[dict[str, float | int]]:
    """Return deterministic accept-high-confidence risk/coverage points."""

    conf = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correctness, dtype=np.float64)
    if conf.ndim != 1 or correct.ndim != 1 or conf.size == 0 or conf.size != correct.size:
        raise ValueError("confidence and correctness must be non-empty matching vectors")
    if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any() or not np.isin(correct, [0.0, 1.0]).all():
        raise ValueError("invalid confidence/correctness values")
    order = np.argsort(-conf, kind="stable")
    accepted = correct[order]
    result: list[dict[str, float | int]] = []
    cumulative_errors = 0.0
    for count, value in enumerate(accepted, start=1):
        cumulative_errors += 1.0 - float(value)
        result.append({
            "accepted_count": count,
            "coverage": float(count / len(accepted)),
            "risk": float(cumulative_errors / count),
        })
    return result


def onset_spread_order_accuracy(
    onset_times: Sequence[float | None],
    spread_times: Sequence[float | None],
) -> float:
    """Measure the fraction of assessable events with onset strictly before spread."""

    if len(onset_times) != len(spread_times) or not onset_times:
        raise ValueError("onset and spread arrays must be non-empty and matching")
    assessable = [(float(onset), float(spread)) for onset, spread in zip(onset_times, spread_times) if onset is not None and spread is not None]
    if not assessable:
        return float("nan")
    return float(sum(onset < spread for onset, spread in assessable) / len(assessable))


def unsupported_claim_rate(
    claims: Sequence[MappingLike],
    evidence_ids: Iterable[str],
) -> float:
    """Return the fraction of report claims whose support IDs are absent."""

    if not claims:
        return 0.0
    available = {str(item) for item in evidence_ids}
    unsupported = 0
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("claims must contain objects")
        support = claim.get("evidence_ids", claim.get("support_ids", []))
        if not isinstance(support, list) or not support or any(str(item) not in available for item in support):
            unsupported += 1
    return float(unsupported / len(claims))


def correction_corruption_rates(
    base_correct: Sequence[bool],
    new_correct: Sequence[bool],
) -> dict[str, float]:
    """Compute correction and corruption conditional rates for a feedback loop."""

    base = np.asarray(base_correct, dtype=bool)
    new = np.asarray(new_correct, dtype=bool)
    if base.ndim != 1 or new.ndim != 1 or base.size == 0 or base.size != new.size:
        raise ValueError("base_correct and new_correct must be non-empty matching vectors")
    wrong_base = ~base
    right_base = base
    return {
        "correction_rate": float(np.mean(new[wrong_base])) if wrong_base.any() else 0.0,
        "corruption_rate": float(np.mean(~new[right_base])) if right_base.any() else 0.0,
    }


MappingLike = dict[str, Any]


__all__ = [
    "brier_score_multiclass",
    "correction_corruption_rates",
    "expected_calibration_error",
    "mean_reciprocal_rank",
    "onset_spread_order_accuracy",
    "risk_coverage_curve",
    "top_k_candidate_hit",
    "unsupported_claim_rate",
]
