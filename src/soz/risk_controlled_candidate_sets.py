from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True, order=True)
class CandidatePolicy:
    tau: float
    k: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau):
            raise ValueError("tau must be finite")
        if self.k <= 0:
            raise ValueError("k must be positive")


def binomial_upper(errors: int, total: int, alpha: float) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    if errors < 0 or errors > total:
        raise ValueError("errors must lie in [0,total]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if errors == total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, errors + 1, total - errors))


def binomial_lower(successes: int, total: int, alpha: float) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie in [0,total]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def _as_bool_matrix(value: np.ndarray, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _validate_inputs(
    scores: np.ndarray,
    strict_positive: np.ndarray,
    relaxed_acceptable: np.ndarray,
    contralateral_far: np.ndarray,
    spread: np.ndarray,
    spread_known: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 2 or score_array.shape[0] == 0 or score_array.shape[1] < 2:
        raise ValueError("scores must be a non-empty [patients,channels] matrix")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    shape = score_array.shape
    strict = _as_bool_matrix(strict_positive, name="strict_positive", shape=shape)
    relaxed = _as_bool_matrix(relaxed_acceptable, name="relaxed_acceptable", shape=shape)
    contra = _as_bool_matrix(contralateral_far, name="contralateral_far", shape=shape)
    spread_array = _as_bool_matrix(spread, name="spread", shape=shape)
    known = np.asarray(spread_known, dtype=bool)
    if known.shape != (shape[0],):
        raise ValueError(f"spread_known must have shape {(shape[0],)}, got {known.shape}")
    if np.any(strict.sum(axis=1) == 0):
        raise ValueError("each calibration patient must have at least one strict positive")
    if np.any(relaxed.sum(axis=1) == 0):
        raise ValueError("each calibration patient must have at least one relaxed acceptable electrode")
    if np.any(strict & ~relaxed):
        raise ValueError("relaxed_acceptable must contain every strict positive")
    return score_array, strict, relaxed, contra, spread_array, known


def _policy_rows(
    *,
    scores: np.ndarray,
    strict_positive: np.ndarray,
    relaxed_acceptable: np.ndarray,
    contralateral_far: np.ndarray,
    spread: np.ndarray,
    spread_known: np.ndarray,
    policy: CandidatePolicy,
) -> dict[str, Any]:
    n_patients, n_channels = scores.shape
    if policy.k > n_channels:
        raise ValueError("policy k exceeds channel count")
    order = np.argsort(-scores, axis=1, kind="stable")
    top1 = order[:, 0]
    top2 = order[:, 1]
    margin = scores[np.arange(n_patients), top1] - scores[np.arange(n_patients), top2]
    accepted = margin >= policy.tau
    candidate_idx = order[:, : policy.k]
    candidate_mask = np.zeros_like(strict_positive, dtype=bool)
    candidate_mask[np.arange(n_patients)[:, None], candidate_idx] = True

    accepted_indices = np.flatnonzero(accepted)
    strict_miss = ~np.any(candidate_mask & strict_positive, axis=1)
    relaxed_miss = ~np.any(candidate_mask & relaxed_acceptable, axis=1)
    contra_top1 = contralateral_far[np.arange(n_patients), top1]
    spread_top1 = spread[np.arange(n_patients), top1]
    spread_denominator = accepted & spread_known
    return {
        "accepted": accepted,
        "accepted_count": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "strict_miss_count": int(strict_miss[accepted_indices].sum()),
        "relaxed_miss_count": int(relaxed_miss[accepted_indices].sum()),
        "contralateral_far_count": int(contra_top1[accepted_indices].sum()),
        "spread_top1_count": int(spread_top1[spread_denominator].sum()),
        "spread_denominator": int(spread_denominator.sum()),
        "candidate_burden": float(policy.k / n_channels),
        "top1": top1,
        "margin": margin,
    }


def calibrate_candidate_policy(
    *,
    scores: np.ndarray,
    strict_positive: np.ndarray,
    relaxed_acceptable: np.ndarray,
    contralateral_far: np.ndarray,
    spread: np.ndarray,
    spread_known: np.ndarray,
    policies: Sequence[CandidatePolicy],
    risk_limits: Mapping[str, float],
    coverage_floor: float,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    """Select a policy using simultaneous exact-binomial bounds on frozen calibration patients.

    The caller must supply patient-level, label-fresh S1-C rows. This function does not
    train a model and deliberately returns fail-closed when no policy is qualified.
    """

    score_array, strict, relaxed, contra, spread_array, known = _validate_inputs(
        scores,
        strict_positive,
        relaxed_acceptable,
        contralateral_far,
        spread,
        spread_known,
    )
    if not policies:
        raise ValueError("at least one policy is required")
    unique_policies = tuple(sorted(set(policies)))
    required_risks = {
        "strict_miss_all",
        "neighborhood4_miss_all",
        "contralateral_far_top1",
        "spread_top1",
        "candidate_burden",
    }
    if set(risk_limits) != required_risks:
        raise ValueError(f"risk_limits must contain exactly {sorted(required_risks)}")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0,1)")
    if not 0.0 <= coverage_floor <= 1.0:
        raise ValueError("coverage_floor must lie in [0,1]")
    if any(not 0.0 <= float(value) <= 1.0 for value in risk_limits.values()):
        raise ValueError("risk limits must lie in [0,1]")

    # Four binomial risks plus one binomial coverage constraint are protected for
    # every policy. Candidate burden is deterministic given k and is not assigned
    # an additional tail probability.
    simultaneous_tests = len(unique_policies) * 5
    local_alpha = familywise_alpha / simultaneous_tests
    n_patients = score_array.shape[0]
    rows: list[dict[str, Any]] = []
    for policy in unique_policies:
        counts = _policy_rows(
            scores=score_array,
            strict_positive=strict,
            relaxed_acceptable=relaxed,
            contralateral_far=contra,
            spread=spread_array,
            spread_known=known,
            policy=policy,
        )
        accepted = int(counts["accepted_count"])
        coverage_lower = binomial_lower(accepted, n_patients, local_alpha)
        bounds: dict[str, float | None] = {
            "strict_miss_all": None,
            "neighborhood4_miss_all": None,
            "contralateral_far_top1": None,
            "spread_top1": None,
            "candidate_burden": float(counts["candidate_burden"]),
        }
        estimates: dict[str, float | None] = {
            "strict_miss_all": None,
            "neighborhood4_miss_all": None,
            "contralateral_far_top1": None,
            "spread_top1": None,
            "candidate_burden": float(counts["candidate_burden"]),
        }
        if accepted > 0:
            for name, count_key in (
                ("strict_miss_all", "strict_miss_count"),
                ("neighborhood4_miss_all", "relaxed_miss_count"),
                ("contralateral_far_top1", "contralateral_far_count"),
            ):
                count = int(counts[count_key])
                estimates[name] = count / accepted
                bounds[name] = binomial_upper(count, accepted, local_alpha)
        spread_denominator = int(counts["spread_denominator"])
        if spread_denominator > 0:
            spread_count = int(counts["spread_top1_count"])
            estimates["spread_top1"] = spread_count / spread_denominator
            bounds["spread_top1"] = binomial_upper(spread_count, spread_denominator, local_alpha)

        constraint_pass = {
            "coverage": coverage_lower >= coverage_floor,
            **{
                name: bounds[name] is not None and float(bounds[name]) <= float(risk_limits[name])
                for name in (
                    "strict_miss_all",
                    "neighborhood4_miss_all",
                    "contralateral_far_top1",
                    "spread_top1",
                )
            },
            "candidate_burden": float(bounds["candidate_burden"]) <= float(risk_limits["candidate_burden"]),
        }
        rows.append(
            {
                "tau": policy.tau,
                "k": policy.k,
                "accepted_count": accepted,
                "patient_count": n_patients,
                "coverage": counts["coverage"],
                "coverage_lower_simultaneous": coverage_lower,
                "spread_known_accepted_count": spread_denominator,
                "risk_estimates": estimates,
                "risk_upper_simultaneous": bounds,
                "constraint_pass": constraint_pass,
                "qualified": all(constraint_pass.values()),
            }
        )

    qualified = [row for row in rows if row["qualified"]]
    if qualified:
        # Simultaneous bounds make data-dependent selection within this finite
        # family valid. Prefer coverage, then lower candidate burden, then larger
        # margin threshold as an explicit conservative tie-break.
        selected = sorted(
            qualified,
            key=lambda row: (-float(row["coverage"]), int(row["k"]), -float(row["tau"])),
        )[0]
        status = "QUALIFIED"
        action = "release_selected_candidate_policy"
    else:
        selected = None
        status = "NO_QUALIFIED_POLICY"
        action = "fail_closed_no_localization_display"
    return {
        "schema_version": "trustworthy_soz_risk_controlled_candidate_policy_v1",
        "status": status,
        "action": action,
        "patient_count": n_patients,
        "policy_count": len(unique_policies),
        "familywise_alpha": familywise_alpha,
        "simultaneous_test_count": simultaneous_tests,
        "local_alpha": local_alpha,
        "coverage_floor": coverage_floor,
        "risk_limits": dict(risk_limits),
        "selected_policy": selected,
        "policy_rows": rows,
        "semantic_boundary": {
            "calibration_unit": "patient",
            "strict_positive_is_multi_label_set": True,
            "unknown_is_negative": False,
            "spread_missing_is_negative": False,
            "nonexchangeable_shift_is_covered": False,
        },
    }
