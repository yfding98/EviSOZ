"""Input-dependent, fixed-budget inner ragged router for BA-IEG.

The outer BA-IEG controller decides which physical EEG support is available.
This module never opens an EDF and never requests new support.  It only chooses
the temporal resolution used to encode the already acquired support.

One candidate cell is deliberately *channel neutral*: it represents every
eligible analysis unit/reference row for one physical interval and one
temporal-permission lane.  The router may therefore refine time, but it cannot
silently select a channel and turn its routing decision into an SOZ claim.

The decoder starts from the coarsest complete partition and replaces a parent
only by a complete, non-overlapping child cover.  A learned or deterministic
score is interpreted as utility density per physical second, so splitting one
cell into more rows cannot mechanically increase utility.  Selection is bound
by both token cost and resolution-weighted EEG seconds and is deterministic
under candidate row reordering.

The trainable scorer below is a target-free primitive.  It consumes only the
registered signal-derived routing features.  Scores, attention, cell choices
and route receipts are not clinical Findings and never authorize onset, SOZ,
EZ, diagnosis or report text.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


BA_IEG_INNER_RAGGED_ROUTER_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_ba_ieg_inner_ragged_router_v1"
BA_IEG_INNER_RAGGED_ROUTER_METHOD_ID: Final[
    str
] = "ba_ieg_channel_neutral_hierarchical_ragged_resolution_router_v1"
BA_IEG_INNER_RAGGED_ROUTER_MODEL_ID: Final[
    str
] = "ba_ieg_source_only_cell_utility_density_mlp_v1"

BA_IEG_INNER_ROUTER_SCALES: Final[tuple[str, ...]] = (
    "fine",
    "coarse",
    "context",
)
BA_IEG_INNER_ROUTER_PERMISSIONS: Final[tuple[str, ...]] = (
    "onset_causal",
    "morphology_native",
    "context_offline",
)
BA_IEG_INNER_ROUTER_SCORE_SOURCES: Final[frozenset[str]] = frozenset(
    {"deterministic_signal_policy", "source_trained_model"}
)
BA_IEG_INNER_ROUTER_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log1p_duration_seconds",
    "boundary_uncertainty",
    "log1p_change_density",
    "cross_channel_disagreement",
    "reference_instability",
    "quality_fraction",
    "opportunity_fraction",
    "fine_scale",
    "coarse_scale",
    "context_scale",
    "onset_causal_permission",
    "morphology_native_permission",
    "context_offline_permission",
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_INTERVAL_TOLERANCE_SECONDS: Final[float] = 1e-8
_CANDIDATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "cell_id",
        "parent_cell_id",
        "scale",
        "permission",
        "nominal_interval_seconds",
        "actual_interval_seconds",
        "future_sample_access",
        "onset_evidence_authorized",
        "view_ids",
        "unit_ids",
        "reference_families",
        "source_token_indices",
        "raw_dependency_sha256s",
        "view_receipt_sha256s",
        "transform_sha256s",
        "boundary_uncertainty",
        "change_density",
        "cross_channel_disagreement",
        "reference_instability",
        "quality_fraction",
        "opportunity_fraction",
        "router_score",
        "score_source",
        "score_receipt_sha256",
        "token_cost",
        "resolution_weighted_eeg_seconds_cost",
    }
)
_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "method_id",
        "router_id",
        "route_boundary",
        "source_binding",
        "policy",
        "outer_support_union",
        "candidate_cells",
        "selected_cells",
        "budget",
        "coverage",
        "permission_partition",
        "replay",
        "router_sha256",
    }
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in _SHA256_CHARACTERS for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} exceeds its maximum")
    return result


def _interval(value: object, name: str) -> tuple[float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a two-item interval")
    start = _finite(value[0], f"{name}[0]")
    stop = _finite(value[1], f"{name}[1]")
    if stop <= start:
        raise ValueError(f"{name} must have positive duration")
    return start, stop


def _duration(interval: Sequence[float]) -> float:
    return float(interval[1]) - float(interval[0])


def _contains(outer: Sequence[float], inner: Sequence[float]) -> bool:
    return bool(
        float(inner[0]) >= float(outer[0]) - _INTERVAL_TOLERANCE_SECONDS
        and float(inner[1]) <= float(outer[1]) + _INTERVAL_TOLERANCE_SECONDS
    )


def _overlaps(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(
        min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0]))
        > _INTERVAL_TOLERANCE_SECONDS
    )


def _normalize_interval_union(
    value: object,
    name: str,
) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an interval array")
    intervals = sorted(
        (_interval(item, f"{name}[{index}]") for index, item in enumerate(value)),
        key=lambda item: (item[0], item[1]),
    )
    merged: list[list[float]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1] + _INTERVAL_TOLERANCE_SECONDS:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return merged


def _union_seconds(intervals: Sequence[Sequence[float]]) -> float:
    return sum(
        _duration(item) for item in _normalize_interval_union(intervals, "intervals")
    )


def _same_interval_union(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> bool:
    first = _normalize_interval_union(left, "left_interval_union")
    second = _normalize_interval_union(right, "right_interval_union")
    return len(first) == len(second) and all(
        abs(a[0] - b[0]) <= _INTERVAL_TOLERANCE_SECONDS
        and abs(a[1] - b[1]) <= _INTERVAL_TOLERANCE_SECONDS
        for a, b in zip(first, second)
    )


def _sorted_unique_strings(value: object, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result = sorted(_identifier(item, f"{name}[]") for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _sorted_unique_hashes(value: object, name: str) -> list[str]:
    result = _sorted_unique_strings(value, name)
    for index, item in enumerate(result):
        _sha256(item, f"{name}[{index}]")
    return result


def _sorted_unique_indices(value: object, name: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    result: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or item < 0:
            raise ValueError(f"{name}[{index}] must be a non-negative integer")
        result.append(item)
    result.sort()
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


@dataclass(frozen=True)
class BAIEGInnerRaggedRouterPolicyV1:
    """Frozen budgets and admission thresholds for the inner router."""

    maximum_token_cost: int = 4096
    maximum_resolution_weighted_eeg_seconds: float = 512.0
    minimum_quality_fraction: float = 0.0
    minimum_opportunity_fraction: float = 0.0
    positive_upgrade_gain_epsilon: float = 0.0
    fine_resolution_weight: float = 1.0
    coarse_resolution_weight: float = 0.25
    context_resolution_weight: float = 0.0625
    onset_causal_budget_fraction: float = 0.5
    morphology_native_budget_fraction: float = 0.2
    context_offline_budget_fraction: float = 0.3

    def __post_init__(self) -> None:
        if type(self.maximum_token_cost) is not int or self.maximum_token_cost < 1:
            raise ValueError("maximum_token_cost must be an integer >= 1")
        _finite(
            self.maximum_resolution_weighted_eeg_seconds,
            "maximum_resolution_weighted_eeg_seconds",
            minimum=0.0,
        )
        for name in (
            "minimum_quality_fraction",
            "minimum_opportunity_fraction",
        ):
            _finite(getattr(self, name), name, minimum=0.0, maximum=1.0)
        _finite(
            self.positive_upgrade_gain_epsilon,
            "positive_upgrade_gain_epsilon",
            minimum=0.0,
        )
        weights = (
            float(self.fine_resolution_weight),
            float(self.coarse_resolution_weight),
            float(self.context_resolution_weight),
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("resolution weights must be finite and positive")
        if not weights[0] > weights[1] > weights[2]:
            raise ValueError("resolution weights must decrease from fine to context")
        if any(
            not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
            for value, expected in zip(weights, (1.0, 0.25, 0.0625))
        ):
            raise ValueError("v1 resolution weights are frozen at 1, 1/4 and 1/16")
        fractions = tuple(
            _finite(getattr(self, name), name, minimum=0.0, maximum=1.0)
            for name in (
                "onset_causal_budget_fraction",
                "morphology_native_budget_fraction",
                "context_offline_budget_fraction",
            )
        )
        if any(value <= 0.0 for value in fractions) or not math.isclose(
            sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "permission-lane budget fractions must be positive and sum to one"
            )

    @property
    def scale_resolution_weights(self) -> dict[str, float]:
        return {
            "fine": float(self.fine_resolution_weight),
            "coarse": float(self.coarse_resolution_weight),
            "context": float(self.context_resolution_weight),
        }

    @property
    def permission_budget_fractions(self) -> dict[str, float]:
        return {
            "onset_causal": float(self.onset_causal_budget_fraction),
            "morphology_native": float(self.morphology_native_budget_fraction),
            "context_offline": float(self.context_offline_budget_fraction),
        }

    @property
    def permission_token_budgets(self) -> dict[str, int]:
        raw = {
            permission: math.floor(
                self.maximum_token_cost * self.permission_budget_fractions[permission]
            )
            for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
        }
        remainder = self.maximum_token_cost - sum(raw.values())
        raw["onset_causal"] += remainder
        return raw

    @property
    def permission_resolution_budgets(self) -> dict[str, float]:
        return {
            permission: self.maximum_resolution_weighted_eeg_seconds
            * self.permission_budget_fractions[permission]
            for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "ba_ieg_inner_ragged_router_policy_v1",
            "maximum_token_cost": int(self.maximum_token_cost),
            "maximum_resolution_weighted_eeg_seconds": float(
                self.maximum_resolution_weighted_eeg_seconds
            ),
            "minimum_quality_fraction": float(self.minimum_quality_fraction),
            "minimum_opportunity_fraction": float(self.minimum_opportunity_fraction),
            "positive_upgrade_gain_epsilon": float(self.positive_upgrade_gain_epsilon),
            "scale_resolution_weights": self.scale_resolution_weights,
            "permission_budget_fractions": self.permission_budget_fractions,
            "permission_token_budgets": self.permission_token_budgets,
            "permission_resolution_weighted_eeg_seconds_budgets": (
                self.permission_resolution_budgets
            ),
            "router_score_semantics": "per_physical_second_utility_density",
            "selection_algorithm": (
                "coarsest_complete_partition_then_positive_gain_child_cover_v1"
            ),
            "channel_or_reference_subset_selection_allowed": False,
            "new_physical_eeg_acquisition_allowed": False,
            "clinical_claim_authorized": False,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BAIEGInnerRaggedRouterPolicyV1":
        if type(value) is not dict:
            raise TypeError("inner-router policy must be an object")
        expected = cls().to_dict()
        if set(value) != set(expected):
            raise ValueError("inner-router policy has missing or unknown fields")
        if value["schema_version"] != expected["schema_version"]:
            raise ValueError("inner-router policy schema drifted")
        if value["router_score_semantics"] != expected["router_score_semantics"]:
            raise ValueError("inner-router score semantics drifted")
        if value["selection_algorithm"] != expected["selection_algorithm"]:
            raise ValueError("inner-router selection algorithm drifted")
        for name in (
            "channel_or_reference_subset_selection_allowed",
            "new_physical_eeg_acquisition_allowed",
            "clinical_claim_authorized",
        ):
            if value[name] is not False:
                raise ValueError(f"inner-router policy illegally opened {name}")
        weights = value["scale_resolution_weights"]
        if type(weights) is not dict or set(weights) != set(BA_IEG_INNER_ROUTER_SCALES):
            raise ValueError("inner-router scale weights are invalid")
        fractions = value["permission_budget_fractions"]
        if type(fractions) is not dict or set(fractions) != set(
            BA_IEG_INNER_ROUTER_PERMISSIONS
        ):
            raise ValueError("inner-router permission budget fractions are invalid")
        result = cls(
            maximum_token_cost=value["maximum_token_cost"],
            maximum_resolution_weighted_eeg_seconds=value[
                "maximum_resolution_weighted_eeg_seconds"
            ],
            minimum_quality_fraction=value["minimum_quality_fraction"],
            minimum_opportunity_fraction=value["minimum_opportunity_fraction"],
            positive_upgrade_gain_epsilon=value["positive_upgrade_gain_epsilon"],
            fine_resolution_weight=weights["fine"],
            coarse_resolution_weight=weights["coarse"],
            context_resolution_weight=weights["context"],
            onset_causal_budget_fraction=fractions["onset_causal"],
            morphology_native_budget_fraction=fractions["morphology_native"],
            context_offline_budget_fraction=fractions["context_offline"],
        )
        if value["permission_token_budgets"] != result.permission_token_budgets:
            raise ValueError(
                "inner-router permission token budgets are not reproducible"
            )
        if (
            value["permission_resolution_weighted_eeg_seconds_budgets"]
            != result.permission_resolution_budgets
        ):
            raise ValueError(
                "inner-router permission resolution budgets are not reproducible"
            )
        return result


@dataclass(frozen=True)
class BAIEGInnerRaggedRouterModelOutputV1:
    """Signal-derived candidate scores without clinical semantics."""

    utility_density: torch.Tensor
    harm_logit: torch.Tensor


class BAIEGInnerRaggedRouterScorerV1(nn.Module):
    """Small trainable scorer over target-free registered cell features."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 4:
            raise ValueError("hidden_dim must be an integer >= 4")
        self.network = nn.Sequential(
            nn.Linear(len(BA_IEG_INNER_ROUTER_FEATURE_NAMES), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> BAIEGInnerRaggedRouterModelOutputV1:
        if features.ndim != 2 or features.shape[1] != len(
            BA_IEG_INNER_ROUTER_FEATURE_NAMES
        ):
            raise ValueError("inner-router features have the wrong shape")
        if not torch.isfinite(features).all():
            raise ValueError("inner-router features must be finite")
        output = self.network(features)
        return BAIEGInnerRaggedRouterModelOutputV1(
            utility_density=output[:, 0],
            harm_logit=output[:, 1],
        )


def risk_adjusted_inner_router_utility_density_v1(
    output: BAIEGInnerRaggedRouterModelOutputV1,
    *,
    harm_weight: float = 1.0,
) -> torch.Tensor:
    """Convert model heads to one selection score without clinical meaning."""

    if not isinstance(output, BAIEGInnerRaggedRouterModelOutputV1):
        raise TypeError("output must be BAIEGInnerRaggedRouterModelOutputV1")
    weight = _finite(harm_weight, "harm_weight", minimum=0.0)
    if output.utility_density.ndim != 1 or tuple(output.harm_logit.shape) != tuple(
        output.utility_density.shape
    ):
        raise ValueError("inner-router model heads must be aligned vectors")
    return output.utility_density - weight * torch.sigmoid(output.harm_logit)


def compute_ba_ieg_inner_router_training_loss_v1(
    output: BAIEGInnerRaggedRouterModelOutputV1,
    *,
    target_signed_utility_density: torch.Tensor,
    target_mask: torch.Tensor,
    context_group_index: torch.Tensor,
    split: str,
    regression_weight: float = 1.0,
    harm_weight: float = 1.0,
    ranking_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Train on frozen counterfactual utility targets in ``source_train`` only.

    Targets are supervision-only sidecars.  They must be generated by a
    frozen, patient-separated downstream evaluator after hiding/revealing the
    candidate cell; this function never accepts a physician label, report or
    clinical term.  ``context_group_index`` limits pairwise ranking to actions
    originating from the same visible event/context.
    """

    if split != "source_train":
        raise ValueError("inner-router gradients are allowed only on source_train")
    if not isinstance(output, BAIEGInnerRaggedRouterModelOutputV1):
        raise TypeError("output must be BAIEGInnerRaggedRouterModelOutputV1")
    prediction = output.utility_density
    harm_logit = output.harm_logit
    target = target_signed_utility_density
    if prediction.ndim != 1 or tuple(harm_logit.shape) != tuple(prediction.shape):
        raise ValueError("inner-router prediction heads must be aligned vectors")
    if (
        tuple(target.shape) != tuple(prediction.shape)
        or target_mask.dtype != torch.bool
        or tuple(target_mask.shape) != tuple(prediction.shape)
        or context_group_index.dtype != torch.long
        or tuple(context_group_index.shape) != tuple(prediction.shape)
    ):
        raise ValueError("inner-router supervision tensors are not aligned")
    if not torch.isfinite(target[target_mask]).all():
        raise ValueError("evaluable inner-router targets must be finite")
    if torch.any(context_group_index < 0):
        raise ValueError("context_group_index cannot be negative")
    if not bool(target_mask.any()):
        raise ValueError("inner-router batch has no evaluable target")
    weights = tuple(
        _finite(value, name, minimum=0.0)
        for value, name in (
            (regression_weight, "regression_weight"),
            (harm_weight, "harm_weight"),
            (ranking_weight, "ranking_weight"),
        )
    )
    regression = F.smooth_l1_loss(
        prediction[target_mask], target[target_mask], reduction="mean"
    )
    harm_target = (target[target_mask] < 0.0).to(harm_logit.dtype)
    harm = F.binary_cross_entropy_with_logits(
        harm_logit[target_mask], harm_target, reduction="mean"
    )
    ranking_terms: list[torch.Tensor] = []
    active_indices = torch.nonzero(target_mask, as_tuple=False).flatten()
    for group in torch.unique(context_group_index[active_indices]):
        indices = active_indices[context_group_index[active_indices] == group]
        for left_position in range(int(indices.numel())):
            for right_position in range(left_position + 1, int(indices.numel())):
                left = int(indices[left_position])
                right = int(indices[right_position])
                target_difference = target[left] - target[right]
                if float(torch.abs(target_difference).detach().cpu()) <= 1e-12:
                    continue
                direction = torch.sign(target_difference).detach()
                prediction_difference = prediction[left] - prediction[right]
                ranking_terms.append(F.softplus(-direction * prediction_difference))
    ranking = (
        torch.stack(ranking_terms).mean() if ranking_terms else prediction.sum() * 0.0
    )
    total = weights[0] * regression + weights[1] * harm + weights[2] * ranking
    return {
        "loss": total,
        "utility_density_regression_loss": regression,
        "harm_classification_loss": harm,
        "within_context_pairwise_ranking_loss": ranking,
        "evaluable_cell_count": target_mask.sum().to(dtype=torch.float32),
        "ranking_pair_count": torch.tensor(
            float(len(ranking_terms)), device=prediction.device
        ),
    }


def _normalize_candidate(
    value: object,
    *,
    outer_support_union: Sequence[Sequence[float]],
    policy: BAIEGInnerRaggedRouterPolicyV1,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _CANDIDATE_KEYS:
        raise ValueError("inner-router candidate cell has missing or unknown fields")
    cell = deepcopy(value)
    cell_id = _identifier(cell["cell_id"], "candidate.cell_id")
    parent = cell["parent_cell_id"]
    if parent is not None:
        parent = _identifier(parent, "candidate.parent_cell_id")
        if parent == cell_id:
            raise ValueError("candidate cell cannot be its own parent")
    scale = str(cell["scale"])
    permission = str(cell["permission"])
    if scale not in BA_IEG_INNER_ROUTER_SCALES:
        raise ValueError("candidate cell scale is unsupported")
    if permission not in BA_IEG_INNER_ROUTER_PERMISSIONS:
        raise ValueError("candidate cell permission is unsupported")
    nominal = _interval(cell["nominal_interval_seconds"], "candidate.nominal_interval")
    actual = _interval(cell["actual_interval_seconds"], "candidate.actual_interval")
    if not _contains(nominal, actual):
        raise ValueError("candidate actual interval escapes its nominal interval")
    if not any(_contains(item, nominal) for item in outer_support_union):
        raise ValueError("candidate interval escapes outer acquired support")
    if (
        type(cell["future_sample_access"]) is not bool
        or type(cell["onset_evidence_authorized"]) is not bool
    ):
        raise TypeError("candidate temporal permissions must be boolean")
    if permission == "onset_causal" and cell["future_sample_access"] is not False:
        raise ValueError("future-access cell cannot enter the onset-causal partition")
    if permission != "onset_causal" and cell["onset_evidence_authorized"] is not False:
        raise ValueError("non-onset partition cannot authorize onset evidence")
    if cell["onset_evidence_authorized"] and cell["future_sample_access"]:
        raise ValueError("future-access cell cannot authorize onset evidence")

    views = _sorted_unique_strings(cell["view_ids"], "candidate.view_ids")
    units = _sorted_unique_strings(cell["unit_ids"], "candidate.unit_ids")
    references = _sorted_unique_strings(
        cell["reference_families"], "candidate.reference_families"
    )
    token_indices = _sorted_unique_indices(
        cell["source_token_indices"], "candidate.source_token_indices"
    )
    raw_hashes = _sorted_unique_hashes(
        cell["raw_dependency_sha256s"], "candidate.raw_dependency_sha256s"
    )
    view_hashes = _sorted_unique_hashes(
        cell["view_receipt_sha256s"], "candidate.view_receipt_sha256s"
    )
    transform_hashes = _sorted_unique_hashes(
        cell["transform_sha256s"], "candidate.transform_sha256s"
    )
    features = {
        "boundary_uncertainty": _finite(
            cell["boundary_uncertainty"],
            "candidate.boundary_uncertainty",
            minimum=0.0,
            maximum=1.0,
        ),
        "change_density": _finite(
            cell["change_density"], "candidate.change_density", minimum=0.0
        ),
        "cross_channel_disagreement": _finite(
            cell["cross_channel_disagreement"],
            "candidate.cross_channel_disagreement",
            minimum=0.0,
            maximum=1.0,
        ),
        "reference_instability": _finite(
            cell["reference_instability"],
            "candidate.reference_instability",
            minimum=0.0,
            maximum=1.0,
        ),
        "quality_fraction": _finite(
            cell["quality_fraction"],
            "candidate.quality_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        "opportunity_fraction": _finite(
            cell["opportunity_fraction"],
            "candidate.opportunity_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        "router_score": _finite(cell["router_score"], "candidate.router_score"),
    }
    score_source = str(cell["score_source"])
    if score_source not in BA_IEG_INNER_ROUTER_SCORE_SOURCES:
        raise ValueError("candidate score source is unsupported")
    score_receipt = _sha256(
        cell["score_receipt_sha256"], "candidate.score_receipt_sha256"
    )
    if type(cell["token_cost"]) is not int or cell["token_cost"] != len(token_indices):
        raise ValueError(
            "candidate token_cost must equal its unique source-token count"
        )
    expected_resolution_cost = (
        _duration(nominal) * policy.scale_resolution_weights[scale]
    )
    supplied_resolution_cost = _finite(
        cell["resolution_weighted_eeg_seconds_cost"],
        "candidate.resolution_weighted_eeg_seconds_cost",
        minimum=0.0,
    )
    if not math.isclose(
        expected_resolution_cost,
        supplied_resolution_cost,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("candidate resolution cost is not reproducible from policy")
    return {
        "cell_id": cell_id,
        "parent_cell_id": parent,
        "scale": scale,
        "permission": permission,
        "nominal_interval_seconds": [nominal[0], nominal[1]],
        "actual_interval_seconds": [actual[0], actual[1]],
        "future_sample_access": bool(cell["future_sample_access"]),
        "onset_evidence_authorized": bool(cell["onset_evidence_authorized"]),
        "view_ids": views,
        "unit_ids": units,
        "reference_families": references,
        "source_token_indices": token_indices,
        "raw_dependency_sha256s": raw_hashes,
        "view_receipt_sha256s": view_hashes,
        "transform_sha256s": transform_hashes,
        **features,
        "score_source": score_source,
        "score_receipt_sha256": score_receipt,
        "token_cost": int(cell["token_cost"]),
        "resolution_weighted_eeg_seconds_cost": expected_resolution_cost,
    }


def tensorize_ba_ieg_inner_router_candidates_v1(
    candidate_cells: Sequence[Mapping[str, Any]],
    *,
    outer_support_union: Sequence[Sequence[float]],
    policy: BAIEGInnerRaggedRouterPolicyV1,
) -> tuple[tuple[str, ...], torch.Tensor, str]:
    """Return canonical target-free cell features and their input receipt."""

    support = _normalize_interval_union(outer_support_union, "outer_support_union")
    normalized = [
        _normalize_candidate(item, outer_support_union=support, policy=policy)
        for item in candidate_cells
    ]
    normalized.sort(key=lambda item: item["cell_id"])
    ids = tuple(item["cell_id"] for item in normalized)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate cell IDs must be unique")
    rows: list[list[float]] = []
    for item in normalized:
        scale = item["scale"]
        permission = item["permission"]
        rows.append(
            [
                math.log1p(_duration(item["nominal_interval_seconds"])),
                float(item["boundary_uncertainty"]),
                math.log1p(float(item["change_density"])),
                float(item["cross_channel_disagreement"]),
                float(item["reference_instability"]),
                float(item["quality_fraction"]),
                float(item["opportunity_fraction"]),
                float(scale == "fine"),
                float(scale == "coarse"),
                float(scale == "context"),
                float(permission == "onset_causal"),
                float(permission == "morphology_native"),
                float(permission == "context_offline"),
            ]
        )
    tensor = torch.tensor(rows, dtype=torch.float32)
    receipt = _canonical_sha256(
        {
            "schema_version": "ba_ieg_inner_router_target_free_features_v1",
            "feature_names": list(BA_IEG_INNER_ROUTER_FEATURE_NAMES),
            "candidate_cell_ids": list(ids),
            "candidate_cells_sha256": _canonical_sha256(normalized),
            "outer_support_union": support,
            "policy_sha256": policy.sha256,
            "labels_or_clinical_text_used": False,
        }
    )
    return ids, tensor, receipt


def validate_ba_ieg_inner_router_score_receipt_v1(
    payload: object,
) -> dict[str, Any]:
    """Validate a source-model score receipt used by candidate cells."""

    required = {
        "schema_version",
        "method_id",
        "model_receipt_sha256",
        "feature_input_receipt_sha256",
        "candidate_cell_ids",
        "harm_weight",
        "utility_density",
        "harm_probability",
        "risk_adjusted_score",
        "training_targets_read_at_inference",
        "labels_annotations_spreadsheets_or_clinical_text_used",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("inner-router score receipt has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != "ba_ieg_inner_router_source_model_scores_v1":
        raise ValueError("inner-router score receipt schema drifted")
    if data["method_id"] != BA_IEG_INNER_RAGGED_ROUTER_MODEL_ID:
        raise ValueError("inner-router score model identity drifted")
    _sha256(data["model_receipt_sha256"], "score.model_receipt_sha256")
    _sha256(
        data["feature_input_receipt_sha256"],
        "score.feature_input_receipt_sha256",
    )
    ids = _sorted_unique_strings(data["candidate_cell_ids"], "score.candidate_ids")
    if ids != data["candidate_cell_ids"]:
        raise ValueError("score candidate IDs must use canonical order")
    count = len(ids)
    for name in (
        "utility_density",
        "harm_probability",
        "risk_adjusted_score",
    ):
        values = data[name]
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"score {name} does not align with candidate IDs")
        data[name] = [_finite(item, f"score.{name}[]") for item in values]
    if any(value < 0.0 or value > 1.0 for value in data["harm_probability"]):
        raise ValueError("score harm probabilities must lie in [0,1]")
    _finite(data["harm_weight"], "score.harm_weight", minimum=0.0)
    if data["training_targets_read_at_inference"] is not False:
        raise ValueError("inner-router inference cannot read training targets")
    if data["labels_annotations_spreadsheets_or_clinical_text_used"] is not False:
        raise ValueError("inner-router score receipt violates the EEG-only firewall")
    _sha256(data["receipt_sha256"], "score.receipt_sha256")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("inner-router score receipt hash does not bind its content")
    return data


def score_ba_ieg_inner_router_candidates_v1(
    model: BAIEGInnerRaggedRouterScorerV1,
    candidate_cells: Sequence[Mapping[str, Any]],
    *,
    outer_support_union: Sequence[Sequence[float]],
    policy: BAIEGInnerRaggedRouterPolicyV1,
    model_receipt_sha256: str,
    harm_weight: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score canonical candidates with a frozen source-trained model."""

    if not isinstance(model, BAIEGInnerRaggedRouterScorerV1):
        raise TypeError("model must be BAIEGInnerRaggedRouterScorerV1")
    model_receipt = _sha256(model_receipt_sha256, "model_receipt_sha256")
    weight = _finite(harm_weight, "harm_weight", minimum=0.0)
    cell_ids, features, input_receipt = tensorize_ba_ieg_inner_router_candidates_v1(
        candidate_cells,
        outer_support_union=outer_support_union,
        policy=policy,
    )
    if not cell_ids:
        raise ValueError("a source model cannot score an empty candidate roster")
    parameter = next(model.parameters())
    with torch.no_grad():
        output = model(features.to(device=parameter.device))
        scores = risk_adjusted_inner_router_utility_density_v1(
            output, harm_weight=weight
        )
        utility = output.utility_density.detach().cpu().to(torch.float64)
        harm_probability = (
            torch.sigmoid(output.harm_logit).detach().cpu().to(torch.float64)
        )
        risk_adjusted = scores.detach().cpu().to(torch.float64)
    if not (
        torch.isfinite(utility).all()
        and torch.isfinite(harm_probability).all()
        and torch.isfinite(risk_adjusted).all()
    ):
        raise ValueError("inner-router source model emitted a non-finite score")
    receipt: dict[str, Any] = {
        "schema_version": "ba_ieg_inner_router_source_model_scores_v1",
        "method_id": BA_IEG_INNER_RAGGED_ROUTER_MODEL_ID,
        "model_receipt_sha256": model_receipt,
        "feature_input_receipt_sha256": input_receipt,
        "candidate_cell_ids": list(cell_ids),
        "harm_weight": weight,
        "utility_density": [float(item) for item in utility],
        "harm_probability": [float(item) for item in harm_probability],
        "risk_adjusted_score": [float(item) for item in risk_adjusted],
        "training_targets_read_at_inference": False,
        "labels_annotations_spreadsheets_or_clinical_text_used": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    receipt = validate_ba_ieg_inner_router_score_receipt_v1(receipt)
    score_by_id = dict(zip(cell_ids, receipt["risk_adjusted_score"]))
    scored: list[dict[str, Any]] = []
    for item in candidate_cells:
        row = deepcopy(dict(item))
        cell_id = str(row["cell_id"])
        if cell_id not in score_by_id:
            raise ValueError("source-model score roster lost a candidate cell")
        row["router_score"] = float(score_by_id[cell_id])
        row["score_source"] = "source_trained_model"
        row["score_receipt_sha256"] = receipt["receipt_sha256"]
        scored.append(row)
    scored.sort(key=lambda item: str(item["cell_id"]))
    return scored, receipt


def _eligible_cell(
    cell: Mapping[str, Any], policy: BAIEGInnerRaggedRouterPolicyV1
) -> bool:
    return bool(
        float(cell["quality_fraction"]) >= policy.minimum_quality_fraction
        and float(cell["opportunity_fraction"]) >= policy.minimum_opportunity_fraction
    )


def _validate_candidate_tree(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, tuple[str, ...]],
    tuple[str, ...],
    str,
    str,
]:
    by_id: dict[str, Mapping[str, Any]] = {}
    token_owner: dict[int, str] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        if cell_id in by_id:
            raise ValueError("candidate cell IDs must be unique")
        by_id[cell_id] = cell
        for token_index in cell["source_token_indices"]:
            previous = token_owner.setdefault(int(token_index), cell_id)
            if previous != cell_id:
                raise ValueError(
                    "one source token cannot belong to multiple candidate cells"
                )

    score_sources = {str(item["score_source"]) for item in cells}
    score_receipts = {str(item["score_receipt_sha256"]) for item in cells}
    if len(score_sources) != 1 or len(score_receipts) != 1:
        raise ValueError("one route must use one frozen score source and receipt")
    score_source = next(iter(score_sources))
    score_receipt = next(iter(score_receipts))

    children: dict[str, list[str]] = {cell_id: [] for cell_id in by_id}
    roots: list[str] = []
    scale_rank = {name: index for index, name in enumerate(BA_IEG_INNER_ROUTER_SCALES)}
    for cell_id, cell in by_id.items():
        parent_id = cell["parent_cell_id"]
        if parent_id is None:
            roots.append(cell_id)
            continue
        if parent_id not in by_id:
            raise ValueError("candidate parent_cell_id is missing from the roster")
        parent = by_id[parent_id]
        if parent["permission"] != cell["permission"]:
            raise ValueError("candidate parent and child cross a permission lane")
        if scale_rank[str(parent["scale"])] != scale_rank[str(cell["scale"])] + 1:
            raise ValueError(
                "candidate tree must refine one registered scale at a time"
            )
        if not _contains(
            parent["nominal_interval_seconds"], cell["nominal_interval_seconds"]
        ):
            raise ValueError("candidate child interval escapes its parent")
        children[str(parent_id)].append(cell_id)

    for parent_id, child_ids in children.items():
        child_ids.sort(
            key=lambda item: (
                float(by_id[item]["nominal_interval_seconds"][0]),
                float(by_id[item]["nominal_interval_seconds"][1]),
                item,
            )
        )
        if not child_ids:
            continue
        child_intervals = [
            by_id[item]["nominal_interval_seconds"] for item in child_ids
        ]
        for left, right in zip(child_intervals, child_intervals[1:]):
            if _overlaps(left, right):
                raise ValueError("sibling candidate cells cannot overlap")
        if not _same_interval_union(
            child_intervals, [by_id[parent_id]["nominal_interval_seconds"]]
        ):
            raise ValueError("candidate children must form a complete parent cover")

    root_by_permission: dict[str, list[Mapping[str, Any]]] = {
        permission: [] for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
    }
    for root_id in roots:
        root_by_permission[str(by_id[root_id]["permission"])].append(by_id[root_id])
    for permission, permission_roots in root_by_permission.items():
        permission_roots.sort(
            key=lambda item: (
                float(item["nominal_interval_seconds"][0]),
                float(item["nominal_interval_seconds"][1]),
                str(item["cell_id"]),
            )
        )
        for left, right in zip(permission_roots, permission_roots[1:]):
            if _overlaps(
                left["nominal_interval_seconds"], right["nominal_interval_seconds"]
            ):
                raise ValueError(
                    f"root candidate cells overlap inside permission {permission}"
                )

    return (
        by_id,
        {key: tuple(value) for key, value in children.items()},
        tuple(sorted(roots)),
        score_source,
        score_receipt,
    )


def _selected_costs(
    selected_ids: Sequence[str], by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[int, float]:
    return (
        sum(int(by_id[cell_id]["token_cost"]) for cell_id in selected_ids),
        sum(
            float(by_id[cell_id]["resolution_weighted_eeg_seconds_cost"])
            for cell_id in selected_ids
        ),
    )


def _integrated_utility(cell: Mapping[str, Any]) -> float:
    return float(cell["router_score"]) * _duration(cell["nominal_interval_seconds"])


def _candidate_upgrade(
    *,
    parent_id: str,
    child_ids: Sequence[str],
    by_id: Mapping[str, Mapping[str, Any]],
    policy: BAIEGInnerRaggedRouterPolicyV1,
    used_token_cost: int,
    used_resolution_cost: float,
    maximum_token_cost: int,
    maximum_resolution_cost: float,
) -> dict[str, Any] | None:
    parent = by_id[parent_id]
    children = [by_id[item] for item in child_ids]
    if not children or not all(_eligible_cell(item, policy) for item in children):
        return None
    child_token_cost = sum(int(item["token_cost"]) for item in children)
    child_resolution_cost = sum(
        float(item["resolution_weighted_eeg_seconds_cost"]) for item in children
    )
    token_delta = child_token_cost - int(parent["token_cost"])
    resolution_delta = child_resolution_cost - float(
        parent["resolution_weighted_eeg_seconds_cost"]
    )
    if token_delta < 0 or resolution_delta < -1e-8:
        raise ValueError(
            "finer child cover cannot have lower registered resolution cost"
        )
    if used_token_cost + token_delta > maximum_token_cost:
        return None
    if used_resolution_cost + resolution_delta > maximum_resolution_cost + 1e-8:
        return None
    gain = sum(_integrated_utility(item) for item in children) - _integrated_utility(
        parent
    )
    if gain <= policy.positive_upgrade_gain_epsilon:
        return None
    normalized_cost = token_delta / max(1, maximum_token_cost) + resolution_delta / max(
        1e-12, maximum_resolution_cost
    )
    priority = gain / max(normalized_cost, 1e-12)
    return {
        "parent_cell_id": parent_id,
        "child_cell_ids": list(child_ids),
        "integrated_utility_gain": float(gain),
        "token_cost_delta": int(token_delta),
        "resolution_weighted_eeg_seconds_delta": float(resolution_delta),
        "gain_per_normalized_incremental_cost": float(priority),
    }


def _select_permission_lane(
    *,
    permission: str,
    roots: Sequence[str],
    by_id: Mapping[str, Mapping[str, Any]],
    children: Mapping[str, Sequence[str]],
    policy: BAIEGInnerRaggedRouterPolicyV1,
) -> dict[str, Any]:
    """Select one lane without any computational path from another lane."""

    eligible_roots = tuple(
        cell_id
        for cell_id in roots
        if by_id[cell_id]["permission"] == permission
        and _eligible_cell(by_id[cell_id], policy)
    )
    maximum_token_cost = policy.permission_token_budgets[permission]
    maximum_resolution_cost = policy.permission_resolution_budgets[permission]
    minimum_token_cost, minimum_resolution_cost = _selected_costs(eligible_roots, by_id)
    feasible = bool(
        eligible_roots
        and minimum_token_cost <= maximum_token_cost
        and minimum_resolution_cost <= maximum_resolution_cost + 1e-8
    )
    selected: set[str] = set(eligible_roots if feasible else ())
    trace: list[dict[str, Any]] = []
    if feasible:
        while True:
            used_token_cost, used_resolution_cost = _selected_costs(
                sorted(selected), by_id
            )
            upgrades: list[dict[str, Any]] = []
            for parent_id in sorted(selected):
                upgrade = _candidate_upgrade(
                    parent_id=parent_id,
                    child_ids=children[parent_id],
                    by_id=by_id,
                    policy=policy,
                    used_token_cost=used_token_cost,
                    used_resolution_cost=used_resolution_cost,
                    maximum_token_cost=maximum_token_cost,
                    maximum_resolution_cost=maximum_resolution_cost,
                )
                if upgrade is not None:
                    upgrades.append(upgrade)
            if not upgrades:
                break
            upgrades.sort(
                key=lambda item: (
                    -float(item["gain_per_normalized_incremental_cost"]),
                    -float(item["integrated_utility_gain"]),
                    str(item["parent_cell_id"]),
                )
            )
            chosen = upgrades[0]
            selected.remove(str(chosen["parent_cell_id"]))
            selected.update(str(item) for item in chosen["child_cell_ids"])
            trace.append(
                {
                    "permission": permission,
                    "selection_step_within_permission": len(trace),
                    **chosen,
                }
            )
    selected_ids = tuple(sorted(selected))
    used_token_cost, used_resolution_cost = _selected_costs(selected_ids, by_id)
    if not eligible_roots:
        status = "not_evaluable_no_eligible_root_support"
    elif not feasible:
        status = "not_evaluable_lane_budget_below_minimum_partition"
    else:
        status = "materialized"
    return {
        "permission": permission,
        "status": status,
        "eligible_root_cell_ids": list(sorted(eligible_roots)),
        "selected_cell_ids": list(selected_ids),
        "maximum_token_cost": int(maximum_token_cost),
        "minimum_complete_partition_token_cost": int(minimum_token_cost),
        "used_token_cost": int(used_token_cost),
        "maximum_resolution_weighted_eeg_seconds": float(maximum_resolution_cost),
        "minimum_complete_partition_resolution_weighted_eeg_seconds": float(
            minimum_resolution_cost
        ),
        "used_resolution_weighted_eeg_seconds": float(used_resolution_cost),
        "upgrade_trace": trace,
    }


def _permission_partition(
    *,
    permission: str,
    candidates: Sequence[Mapping[str, Any]],
    roots: Sequence[str],
    selected_ids: Sequence[str],
    outer_support_union: Sequence[Sequence[float]],
    policy: BAIEGInnerRaggedRouterPolicyV1,
    lane_selection: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_by_id = {str(item["cell_id"]): item for item in candidates}
    permission_roots = [
        candidate_by_id[item]
        for item in roots
        if candidate_by_id[item]["permission"] == permission
        and _eligible_cell(candidate_by_id[item], policy)
    ]
    selected = [
        candidate_by_id[item]
        for item in selected_ids
        if candidate_by_id[item]["permission"] == permission
    ]
    root_union = _normalize_interval_union(
        [item["nominal_interval_seconds"] for item in permission_roots],
        f"{permission}.root_union",
    )
    selected_nominal_union = _normalize_interval_union(
        [item["nominal_interval_seconds"] for item in selected],
        f"{permission}.selected_nominal_union",
    )
    selected_actual_union = _normalize_interval_union(
        [item["actual_interval_seconds"] for item in selected],
        f"{permission}.selected_actual_union",
    )
    outer_seconds = _union_seconds(outer_support_union)
    root_seconds = _union_seconds(root_union)
    onset_authorized = [
        str(item["cell_id"])
        for item in selected
        if item["permission"] == "onset_causal"
        and item["future_sample_access"] is False
        and item["onset_evidence_authorized"] is True
    ]
    onset_authorized_union = _normalize_interval_union(
        [
            candidate_by_id[item]["nominal_interval_seconds"]
            for item in onset_authorized
        ],
        f"{permission}.onset_authorized_union",
    )
    return {
        "permission": permission,
        "selection_status": str(lane_selection["status"]),
        "maximum_token_cost": int(lane_selection["maximum_token_cost"]),
        "used_token_cost": int(lane_selection["used_token_cost"]),
        "maximum_resolution_weighted_eeg_seconds": float(
            lane_selection["maximum_resolution_weighted_eeg_seconds"]
        ),
        "used_resolution_weighted_eeg_seconds": float(
            lane_selection["used_resolution_weighted_eeg_seconds"]
        ),
        "candidate_cell_count": sum(
            item["permission"] == permission for item in candidates
        ),
        "eligible_root_cell_ids": sorted(
            str(item["cell_id"]) for item in permission_roots
        ),
        "selected_cell_ids": sorted(str(item["cell_id"]) for item in selected),
        "eligible_root_support_union": root_union,
        "selected_nominal_support_union": selected_nominal_union,
        "selected_actual_support_union": selected_actual_union,
        "complete_partition_within_eligible_root_support": _same_interval_union(
            root_union, selected_nominal_union
        ),
        "outer_support_coverage_fraction": (
            0.0 if outer_seconds <= 0.0 else root_seconds / outer_seconds
        ),
        "selected_future_sample_access_present": any(
            bool(item["future_sample_access"]) for item in selected
        ),
        "onset_authorized_selected_cell_ids": sorted(onset_authorized),
        "onset_authorized_support_union": onset_authorized_union,
        "onset_positive_support_eligible": bool(
            permission == "onset_causal"
            and selected
            and _same_interval_union(selected_nominal_union, onset_authorized_union)
        ),
    }


def _route_normalized_candidates(
    *,
    event_id: str,
    canonical_signal_sha256: str,
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]],
    candidates: Sequence[Mapping[str, Any]],
    policy: BAIEGInnerRaggedRouterPolicyV1,
) -> dict[str, Any]:
    by_id, children, roots, score_source, score_receipt = _validate_candidate_tree(
        candidates
    )
    lane_selections = [
        _select_permission_lane(
            permission=permission,
            roots=roots,
            by_id=by_id,
            children=children,
            policy=policy,
        )
        for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
    ]
    eligible_roots = tuple(
        cell_id
        for lane in lane_selections
        for cell_id in lane["eligible_root_cell_ids"]
    )
    selected: set[str] = {
        cell_id for lane in lane_selections for cell_id in lane["selected_cell_ids"]
    }
    upgrade_trace = [item for lane in lane_selections for item in lane["upgrade_trace"]]
    minimum_token_cost = sum(
        int(item["minimum_complete_partition_token_cost"]) for item in lane_selections
    )
    minimum_resolution_cost = sum(
        float(item["minimum_complete_partition_resolution_weighted_eeg_seconds"])
        for item in lane_selections
    )
    infeasible_lanes = [
        str(item["permission"])
        for item in lane_selections
        if item["eligible_root_cell_ids"] and item["status"] != "materialized"
    ]
    budget_feasible = bool(eligible_roots and not infeasible_lanes)

    selected_ids = sorted(
        selected,
        key=lambda cell_id: (
            BA_IEG_INNER_ROUTER_PERMISSIONS.index(str(by_id[cell_id]["permission"])),
            float(by_id[cell_id]["nominal_interval_seconds"][0]),
            float(by_id[cell_id]["nominal_interval_seconds"][1]),
            BA_IEG_INNER_ROUTER_SCALES.index(str(by_id[cell_id]["scale"])),
            cell_id,
        ),
    )
    used_token_cost, used_resolution_cost = _selected_costs(selected_ids, by_id)
    lane_by_permission = {str(item["permission"]): item for item in lane_selections}
    partitions = [
        _permission_partition(
            permission=permission,
            candidates=candidates,
            roots=roots,
            selected_ids=selected_ids,
            outer_support_union=outer_support_union,
            policy=policy,
            lane_selection=lane_by_permission[permission],
        )
        for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
    ]
    complete = all(
        item["complete_partition_within_eligible_root_support"] for item in partitions
    )
    if not candidates:
        status = "not_evaluable_no_candidate_cells"
    elif not eligible_roots:
        status = "not_evaluable_no_eligible_root_support"
    elif not selected_ids and infeasible_lanes:
        status = "not_evaluable_budget_below_minimum_complete_partition"
    elif infeasible_lanes:
        status = "materialized_partial_permission_lane_budget"
    elif any(
        item["outer_support_coverage_fraction"] < 1.0 - 1e-8
        for item in partitions
        if item["candidate_cell_count"]
    ):
        status = "materialized_partial_outer_support_opportunity"
    else:
        status = "materialized_complete_available_support"

    candidate_roster_sha256 = _canonical_sha256(candidates)
    body: dict[str, Any] = {
        "schema_version": BA_IEG_INNER_RAGGED_ROUTER_SCHEMA_VERSION,
        "method_id": BA_IEG_INNER_RAGGED_ROUTER_METHOD_ID,
        "router_id": "CONTENT-ADDRESS-PENDING",
        "route_boundary": {
            "scope": "inside_already_acquired_outer_support_only",
            "new_physical_eeg_acquisition_authorized": False,
            "channel_or_reference_subset_selection_authorized": False,
            "clinical_finding_or_soz_claim_authorized": False,
        },
        "source_binding": {
            "event_id": event_id,
            "canonical_signal_sha256": canonical_signal_sha256,
            "outer_support_receipt_sha256": outer_support_receipt_sha256,
            "candidate_roster_sha256": candidate_roster_sha256,
            "score_source": score_source,
            "score_receipt_sha256": score_receipt,
        },
        "policy": policy.to_dict(),
        "outer_support_union": [list(item) for item in outer_support_union],
        "candidate_cells": [deepcopy(dict(item)) for item in candidates],
        "selected_cells": [deepcopy(dict(by_id[item])) for item in selected_ids],
        "budget": {
            "status": (
                "feasible_for_all_eligible_permission_lanes"
                if budget_feasible
                else (
                    "partial_permission_lane_infeasible"
                    if selected_ids
                    else "infeasible"
                )
            ),
            "maximum_token_cost": int(policy.maximum_token_cost),
            "minimum_complete_partition_token_cost": int(minimum_token_cost),
            "used_token_cost": int(used_token_cost),
            "maximum_resolution_weighted_eeg_seconds": float(
                policy.maximum_resolution_weighted_eeg_seconds
            ),
            "minimum_complete_partition_resolution_weighted_eeg_seconds": float(
                minimum_resolution_cost
            ),
            "used_resolution_weighted_eeg_seconds": float(used_resolution_cost),
            "token_budget_violated": used_token_cost > policy.maximum_token_cost,
            "resolution_budget_violated": (
                used_resolution_cost
                > policy.maximum_resolution_weighted_eeg_seconds + 1e-8
            ),
            "permission_lanes": [
                {
                    key: deepcopy(value)
                    for key, value in item.items()
                    if key != "upgrade_trace"
                }
                for item in lane_selections
            ],
        },
        "coverage": {
            "status": status,
            "outer_support_seconds": float(_union_seconds(outer_support_union)),
            "selected_cell_count": len(selected_ids),
            "complete_partition_within_each_available_permission": bool(complete),
            "candidate_rows_are_channel_neutral_groups": True,
        },
        "permission_partition": partitions,
        "replay": {
            "selection_algorithm": policy.to_dict()["selection_algorithm"],
            "router_score_semantics": policy.to_dict()["router_score_semantics"],
            "candidate_row_order_invariant": True,
            "complete_child_cover_required": True,
            "permission_lane_budget_isolation": True,
            "unused_lane_budget_reallocation_allowed": False,
            "utility_integrated_over_physical_seconds": True,
            "source_token_copy_confidence_gain_authorized": False,
            "offline_to_onset_permission_upgrade_authorized": False,
            "clinical_terms_emitted": False,
            "labels_annotations_spreadsheets_or_clinical_text_used": False,
            "upgrade_trace": upgrade_trace,
        },
        "router_sha256": "CONTENT-ADDRESS-PENDING",
    }
    id_source = deepcopy(body)
    body["router_id"] = "BAIEG-INNER-" + _canonical_sha256(id_source)[:20]
    digest_source = deepcopy(body)
    body["router_sha256"] = _canonical_sha256(digest_source)
    return body


def materialize_ba_ieg_inner_ragged_router_v1(
    *,
    event_id: str,
    canonical_signal_sha256: str,
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]],
    candidate_cells: Sequence[Mapping[str, Any]],
    policy: BAIEGInnerRaggedRouterPolicyV1 = BAIEGInnerRaggedRouterPolicyV1(),
) -> dict[str, Any]:
    """Select an auditable ragged partition inside frozen outer support."""

    event = _identifier(event_id, "event_id")
    canonical = _sha256(canonical_signal_sha256, "canonical_signal_sha256")
    outer_receipt = _sha256(
        outer_support_receipt_sha256, "outer_support_receipt_sha256"
    )
    if not isinstance(policy, BAIEGInnerRaggedRouterPolicyV1):
        raise TypeError("policy must be BAIEGInnerRaggedRouterPolicyV1")
    support = _normalize_interval_union(outer_support_union, "outer_support_union")
    if not support:
        raise ValueError("outer_support_union cannot be empty")
    if isinstance(candidate_cells, (str, bytes)) or not isinstance(
        candidate_cells, Sequence
    ):
        raise TypeError("candidate_cells must be an array")
    normalized = [
        _normalize_candidate(item, outer_support_union=support, policy=policy)
        for item in candidate_cells
    ]
    normalized.sort(
        key=lambda item: (
            str(item["permission"]),
            float(item["nominal_interval_seconds"][0]),
            float(item["nominal_interval_seconds"][1]),
            str(item["scale"]),
            str(item["cell_id"]),
        )
    )
    if not normalized:
        # A typed artifact still needs score provenance.  With no cells there
        # is no legitimate model score receipt, so bind a fixed empty source.
        empty_score_receipt = _canonical_sha256(
            {
                "schema_version": "ba_ieg_inner_router_empty_score_roster_v1",
                "event_id": event,
                "outer_support_receipt_sha256": outer_receipt,
            }
        )
        synthetic_empty = {
            "cell_id": "EMPTY-SENTINEL-NOT-A-CANDIDATE",
            "score_source": "deterministic_signal_policy",
            "score_receipt_sha256": empty_score_receipt,
        }
        # The selector expects score provenance from the roster.  Pass it
        # explicitly through a no-cell branch without manufacturing a cell.
        body = _route_empty_candidates(
            event_id=event,
            canonical_signal_sha256=canonical,
            outer_support_receipt_sha256=outer_receipt,
            outer_support_union=support,
            policy=policy,
            score_source=synthetic_empty["score_source"],
            score_receipt_sha256=synthetic_empty["score_receipt_sha256"],
        )
        return validate_ba_ieg_inner_ragged_router_v1(body)
    body = _route_normalized_candidates(
        event_id=event,
        canonical_signal_sha256=canonical,
        outer_support_receipt_sha256=outer_receipt,
        outer_support_union=support,
        candidates=normalized,
        policy=policy,
    )
    return validate_ba_ieg_inner_ragged_router_v1(body)


def _route_empty_candidates(
    *,
    event_id: str,
    canonical_signal_sha256: str,
    outer_support_receipt_sha256: str,
    outer_support_union: Sequence[Sequence[float]],
    policy: BAIEGInnerRaggedRouterPolicyV1,
    score_source: str,
    score_receipt_sha256: str,
) -> dict[str, Any]:
    if score_source not in BA_IEG_INNER_ROUTER_SCORE_SOURCES:
        raise ValueError("empty route score source is unsupported")
    _sha256(score_receipt_sha256, "empty route score_receipt_sha256")
    partitions = [
        {
            "permission": permission,
            "selection_status": "not_evaluable_no_eligible_root_support",
            "maximum_token_cost": int(policy.permission_token_budgets[permission]),
            "used_token_cost": 0,
            "maximum_resolution_weighted_eeg_seconds": float(
                policy.permission_resolution_budgets[permission]
            ),
            "used_resolution_weighted_eeg_seconds": 0.0,
            "candidate_cell_count": 0,
            "eligible_root_cell_ids": [],
            "selected_cell_ids": [],
            "eligible_root_support_union": [],
            "selected_nominal_support_union": [],
            "selected_actual_support_union": [],
            "complete_partition_within_eligible_root_support": True,
            "outer_support_coverage_fraction": 0.0,
            "selected_future_sample_access_present": False,
            "onset_authorized_selected_cell_ids": [],
            "onset_authorized_support_union": [],
            "onset_positive_support_eligible": False,
        }
        for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
    ]
    body: dict[str, Any] = {
        "schema_version": BA_IEG_INNER_RAGGED_ROUTER_SCHEMA_VERSION,
        "method_id": BA_IEG_INNER_RAGGED_ROUTER_METHOD_ID,
        "router_id": "CONTENT-ADDRESS-PENDING",
        "route_boundary": {
            "scope": "inside_already_acquired_outer_support_only",
            "new_physical_eeg_acquisition_authorized": False,
            "channel_or_reference_subset_selection_authorized": False,
            "clinical_finding_or_soz_claim_authorized": False,
        },
        "source_binding": {
            "event_id": event_id,
            "canonical_signal_sha256": canonical_signal_sha256,
            "outer_support_receipt_sha256": outer_support_receipt_sha256,
            "candidate_roster_sha256": _canonical_sha256([]),
            "score_source": score_source,
            "score_receipt_sha256": score_receipt_sha256,
        },
        "policy": policy.to_dict(),
        "outer_support_union": [list(item) for item in outer_support_union],
        "candidate_cells": [],
        "selected_cells": [],
        "budget": {
            "status": "infeasible",
            "maximum_token_cost": int(policy.maximum_token_cost),
            "minimum_complete_partition_token_cost": 0,
            "used_token_cost": 0,
            "maximum_resolution_weighted_eeg_seconds": float(
                policy.maximum_resolution_weighted_eeg_seconds
            ),
            "minimum_complete_partition_resolution_weighted_eeg_seconds": 0.0,
            "used_resolution_weighted_eeg_seconds": 0.0,
            "token_budget_violated": False,
            "resolution_budget_violated": False,
            "permission_lanes": [
                {
                    "permission": permission,
                    "status": "not_evaluable_no_eligible_root_support",
                    "eligible_root_cell_ids": [],
                    "selected_cell_ids": [],
                    "maximum_token_cost": int(
                        policy.permission_token_budgets[permission]
                    ),
                    "minimum_complete_partition_token_cost": 0,
                    "used_token_cost": 0,
                    "maximum_resolution_weighted_eeg_seconds": float(
                        policy.permission_resolution_budgets[permission]
                    ),
                    "minimum_complete_partition_resolution_weighted_eeg_seconds": 0.0,
                    "used_resolution_weighted_eeg_seconds": 0.0,
                }
                for permission in BA_IEG_INNER_ROUTER_PERMISSIONS
            ],
        },
        "coverage": {
            "status": "not_evaluable_no_candidate_cells",
            "outer_support_seconds": float(_union_seconds(outer_support_union)),
            "selected_cell_count": 0,
            "complete_partition_within_each_available_permission": True,
            "candidate_rows_are_channel_neutral_groups": True,
        },
        "permission_partition": partitions,
        "replay": {
            "selection_algorithm": policy.to_dict()["selection_algorithm"],
            "router_score_semantics": policy.to_dict()["router_score_semantics"],
            "candidate_row_order_invariant": True,
            "complete_child_cover_required": True,
            "permission_lane_budget_isolation": True,
            "unused_lane_budget_reallocation_allowed": False,
            "utility_integrated_over_physical_seconds": True,
            "source_token_copy_confidence_gain_authorized": False,
            "offline_to_onset_permission_upgrade_authorized": False,
            "clinical_terms_emitted": False,
            "labels_annotations_spreadsheets_or_clinical_text_used": False,
            "upgrade_trace": [],
        },
        "router_sha256": "CONTENT-ADDRESS-PENDING",
    }
    id_source = deepcopy(body)
    body["router_id"] = "BAIEG-INNER-" + _canonical_sha256(id_source)[:20]
    digest_source = deepcopy(body)
    body["router_sha256"] = _canonical_sha256(digest_source)
    return body


def validate_ba_ieg_inner_ragged_router_v1(payload: object) -> dict[str, Any]:
    """Replay the complete route and reject self-consistent tampering."""

    if type(payload) is not dict or set(payload) != _ARTIFACT_KEYS:
        raise ValueError("inner-router artifact has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != BA_IEG_INNER_RAGGED_ROUTER_SCHEMA_VERSION:
        raise ValueError("inner-router artifact schema drifted")
    if data["method_id"] != BA_IEG_INNER_RAGGED_ROUTER_METHOD_ID:
        raise ValueError("inner-router method drifted")
    source = data["source_binding"]
    if type(source) is not dict or set(source) != {
        "event_id",
        "canonical_signal_sha256",
        "outer_support_receipt_sha256",
        "candidate_roster_sha256",
        "score_source",
        "score_receipt_sha256",
    }:
        raise ValueError("inner-router source binding is invalid")
    event_id = _identifier(source["event_id"], "source_binding.event_id")
    canonical = _sha256(
        source["canonical_signal_sha256"],
        "source_binding.canonical_signal_sha256",
    )
    outer_receipt = _sha256(
        source["outer_support_receipt_sha256"],
        "source_binding.outer_support_receipt_sha256",
    )
    _sha256(
        source["candidate_roster_sha256"],
        "source_binding.candidate_roster_sha256",
    )
    score_source = str(source["score_source"])
    if score_source not in BA_IEG_INNER_ROUTER_SCORE_SOURCES:
        raise ValueError("inner-router score source is unsupported")
    score_receipt = _sha256(
        source["score_receipt_sha256"],
        "source_binding.score_receipt_sha256",
    )
    policy = BAIEGInnerRaggedRouterPolicyV1.from_dict(data["policy"])
    support = _normalize_interval_union(
        data["outer_support_union"], "outer_support_union"
    )
    if support != data["outer_support_union"]:
        raise ValueError("outer support union is not canonical")
    candidates = data["candidate_cells"]
    if type(candidates) is not list:
        raise TypeError("inner-router candidate_cells must be an array")
    if candidates:
        normalized = [
            _normalize_candidate(item, outer_support_union=support, policy=policy)
            for item in candidates
        ]
        normalized.sort(
            key=lambda item: (
                str(item["permission"]),
                float(item["nominal_interval_seconds"][0]),
                float(item["nominal_interval_seconds"][1]),
                str(item["scale"]),
                str(item["cell_id"]),
            )
        )
        expected = _route_normalized_candidates(
            event_id=event_id,
            canonical_signal_sha256=canonical,
            outer_support_receipt_sha256=outer_receipt,
            outer_support_union=support,
            candidates=normalized,
            policy=policy,
        )
    else:
        expected = _route_empty_candidates(
            event_id=event_id,
            canonical_signal_sha256=canonical,
            outer_support_receipt_sha256=outer_receipt,
            outer_support_union=support,
            policy=policy,
            score_source=score_source,
            score_receipt_sha256=score_receipt,
        )
    if expected != data:
        raise ValueError("inner-router artifact does not replay from its frozen inputs")
    return data
